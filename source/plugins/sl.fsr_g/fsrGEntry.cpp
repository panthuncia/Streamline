/*
 * Community Shaders FSR 3 frame-generation Streamline plugin.
 * Uses a dedicated depth, motion-vector, and HUD-less viewport with an FFX Vulkan swapchain.
 */

#include <cstring>
#include <cstdlib>
#include <atomic>
#include <map>
#include <mutex>
#include <vector>

#include "include/sl.h"
#include "include/sl_consts.h"
#include "source/core/sl.api/internal.h"
#include "source/core/sl.log/log.h"
#include "source/core/sl.plugin/plugin.h"
#include "source/core/sl.param/parameters.h"
#include "source/platforms/sl.chi/compute.h"
#include "source/plugins/sl.fsr_g/versions.h"
#include "source/plugins/sl.common/commonInterface.h"
#include "external/json/include/nlohmann/json.hpp"
#include "_artifacts/json/fsr_g_json.h"
#include "_artifacts/gitVersion.h"

#include "include/sl_fsr_g.h"

// FFX-API (descriptor structs for the prebuilt amd_fidelityfx_vk.dll, resolved at runtime). FFX now
// lives inside the fork under external/fidelityfx-sdk.
#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_api_loader.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/vk/ffx_api_vk.h"

using json = nlohmann::json;

namespace sl
{
namespace fsr
{
struct FrameGenerationFrame
{
    FfxApiResource hudlessResource{};
    uint64_t frameID = 0;
    bool prepared = false;
    bool haveHudless = false;
    bool hudlessCompatible = false;
};

struct FSRContext
{
    SL_PLUGIN_CONTEXT_CREATE_DESTROY(FSRContext);
    void onCreateContext() {};
    void onDestroyContext() {};

    common::PFunRegisterEvaluateCallbacks* registerEvaluateCallbacks{};
    RenderAPI platform = RenderAPI::eVulkan;
    chi::ICompute* compute{};

    // FFX-API runtime.
    HMODULE ffxModule = nullptr;
    ffxFunctions ffxApi{};
    std::atomic<bool> dispatchFaulted{ false };

    // Resolved device functions for the FFX backend + our barriers.
    PFN_vkGetDeviceProcAddr realDeviceProcAddr = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;

    // Guards the frame-gen option writes (slFSRFrameGenerationSetOptions) against the render/present threads.
    std::mutex optionsMutex;

    // Transfers prepared resources from the render thread to the present thread as one frame.
    std::mutex frameMutex;
    FrameGenerationFrame pendingFrame{};
    std::atomic<bool> generatedDispatchSucceeded{ false };
    std::atomic<uint32_t> lastPresentedFrameCount{ 1 };

    // --- Frame generation (FFX FrameInterpolationSwapChainVK proxy) -----------------------------
    // The present hook recreates the swapchain when FSR FG takes or releases ownership. While wrapped,
    // the WSI hooks route swapchain operations to FFX and fsrEndEvaluation records the per-frame prepare.
    bool fgEnabled = false;                  // host requested FSR FG
    bool fgColorHDR = false;
    uint32_t fgDebugFlags = 0;               // FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_* from the host's debug toggles
    bool fgOnlyPresentGenerated = false;     // present only generated frames (host fgShowOnlyGenerated)
    // Defer the first wrapper install until gameplay produces frames. Later enable/disable changes recreate
    // between the FFX wrapper and a plain swapchain.
    std::atomic<bool> fgGameplayReached{ false };
    bool fgBootstrapInFlight = false;
    bool fgWrapFailed = false;
    uint64_t frameID = 0;                    // protected by frameMutex
    int64_t fgLastPrepareQpc = 0;            // QPC of the last FG-prepare, for real frameTimeDelta
    PFN_vkQueuePresentKHR realQueuePresentKHR = nullptr;
    ffxContext fgContext = nullptr;          // interpolation context (ffxCreateContextDescFrameGeneration)
    bool fgContextTeardownRetryDisabled = false;
    ffxContext fgSwapchainContext = nullptr; // swapchain-replacement context (FGSWAPCHAIN_VK)
    VkSwapchainKHR fgWrappedSwapchain = VK_NULL_HANDLE;
    bool fgSwapchainTeardownFailed = false;  // protected by frameMutex
    bool fgSwapchainTeardownRetryDisabled = false; // an SEH may have destroyed the context before faulting
    ffxQueryDescSwapchainReplacementFunctionsVK fgSwapchainFns{};
    uint32_t fgDisplayW = 0, fgDisplayH = 0, fgRenderW = 0, fgRenderH = 0, fgBackBufferFormat = 0;
    bool fgContextHDR = false;  // the HDR flag the current fgContext was actually built with (vs fgColorHDR desired)

    // HUDLessColor is the scene without UI at display resolution. FFX consumes it during present to mask UI,
    // so it must use the swapchain's exact format and current DXVK layout.
    VkFormat fgBackBufferVkFormat = VK_FORMAT_UNDEFINED;

    // The three queues FFX's FG swapchain requires. gameQueue is DXVK's own present queue (captured
    // from the first passthrough present); presentQueue/imageAcquireQueue are the two extra graphics
    // queues the interposer injected on our behalf (declared in updateEmbeddedJSON). All distinct.
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkQueue gameQueue = VK_NULL_HANDLE;
    uint32_t gameQueueFamily = 0;
    VkQueue presentQueue = VK_NULL_HANDLE, imageAcquireQueue = VK_NULL_HANDLE;
    uint32_t extraQueueFamily = 0;
    bool queuesAcquired = false;
};
}

void updateEmbeddedJSON(json& config);

static std::string JSON = std::string(fsr_g_json, &fsr_g_json[fsr_g_json_len]);

SL_PLUGIN_DEFINE("sl.fsr_g", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(0, 0, 1), JSON.c_str(), updateEmbeddedJSON, fsr, FSRContext)

namespace
{
// Skyrim world units -> meters, for FFX's depth-based reactive/reprojection heuristics (~70 units/m).
// Shared by the upscale and FG-prepare dispatches so both describe the scene at the same scale.
constexpr float kSkyrimViewSpaceToMeters = 0.01428222656f;

// --- DXVK ⇄ unmodified prebuilt-FFX VK backend shim (ported from CS FidelityFX.cpp) -------------
// The stock amd_fidelityfx_vk.dll calls two functions that resolve to NULL on DXVK without null-checks:
//   vkGetBufferMemoryRequirements2KHR (DXVK promoted it to core, dropped the KHR alias) and
//   vkCmdWriteBufferMarker(2)AMD (DXVK enumerates VK_AMD_buffer_marker but leaves the entries NULL).
// Route the first to the core entry; stub the markers to a no-op so the UNMODIFIED DLL runs on DXVK.
PFN_vkGetDeviceProcAddr g_realDeviceProcAddr = nullptr;

// Set while FFX creates its inner swapchain inside createFgSwapchain. FFX's inner vkCreateSwapchainKHR
// re-enters the SL interposer, where our CreateSwapchainKHR hook sees this flag and creates the swapchain
// on the genuine driver with Skip=true so the interposer's other frame-generation hooks do not wrap it.
std::atomic<bool> g_creatingFgSwapchain{ false };

VKAPI_ATTR void VKAPI_CALL Noop_vkCmdWriteBufferMarker(
    VkCommandBuffer, VkPipelineStageFlagBits, VkBuffer, VkDeviceSize, uint32_t) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL DxvkFfxGetDeviceProcAddr(VkDevice device, const char* pName)
{
    if (!pName || !g_realDeviceProcAddr)
        return nullptr;
    if (std::strcmp(pName, "vkGetBufferMemoryRequirements2KHR") == 0) {
        if (auto p = g_realDeviceProcAddr(device, "vkGetBufferMemoryRequirements2KHR"))
            return p;
        return g_realDeviceProcAddr(device, "vkGetBufferMemoryRequirements2");
    }
    if (std::strcmp(pName, "vkCmdWriteBufferMarkerAMD") == 0 ||
        std::strcmp(pName, "vkCmdWriteBufferMarker2AMD") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&Noop_vkCmdWriteBufferMarker);
    return g_realDeviceProcAddr(device, pName);
}

// SEH-guarded FFX calls use leaf functions so __try has no C++ objects requiring unwinding
// (C2712). The stock FFX DLL can access-violate on DXVK; latch the feature off instead of crashing.
ffxReturnCode_t ffxCreateContextSEH(ffxFunctions& api, ffxContext* context,
    ffxCreateContextDescHeader* desc, std::atomic<bool>& dispatchFaulted)
{
    if (dispatchFaulted.load(std::memory_order_acquire))
        return (ffxReturnCode_t)0xFFFFFFFF;
    __try { return api.CreateContext(context, desc, nullptr); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        dispatchFaulted.store(true, std::memory_order_release);
        return (ffxReturnCode_t)0xFFFFFFFF;
    }
}
ffxReturnCode_t ffxDispatchSEH(ffxFunctions& api, ffxContext* context,
    ffxDispatchDescHeader* desc, std::atomic<bool>& dispatchFaulted, bool* faulted = nullptr)
{
    if (faulted)
        *faulted = false;
    if (dispatchFaulted.load(std::memory_order_acquire)) {
        if (faulted)
            *faulted = true;
        return (ffxReturnCode_t)0xFFFFFFFF;
    }
    __try { return api.Dispatch(context, desc); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        dispatchFaulted.store(true, std::memory_order_release);
        if (faulted)
            *faulted = true;
        return (ffxReturnCode_t)0xFFFFFFFF;
    }
}
ffxReturnCode_t ffxConfigureSEH(ffxFunctions& api, ffxContext* context,
    ffxConfigureDescHeader* desc, std::atomic<bool>& dispatchFaulted, bool* faulted = nullptr)
{
    if (faulted)
        *faulted = false;
    if (dispatchFaulted.load(std::memory_order_acquire)) {
        if (faulted)
            *faulted = true;
        return (ffxReturnCode_t)0xFFFFFFFF;
    }
    __try { return api.Configure(context, desc); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        dispatchFaulted.store(true, std::memory_order_release);
        if (faulted)
            *faulted = true;
        return (ffxReturnCode_t)0xFFFFFFFF;
    }
}
ffxReturnCode_t ffxQuerySEH(ffxFunctions& api, ffxContext* context,
    ffxQueryDescHeader* desc, std::atomic<bool>& dispatchFaulted)
{
    if (dispatchFaulted.load(std::memory_order_acquire))
        return (ffxReturnCode_t)0xFFFFFFFF;
    __try { return api.Query(context, desc); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        dispatchFaulted.store(true, std::memory_order_release);
        return (ffxReturnCode_t)0xFFFFFFFF;
    }
}
ffxReturnCode_t ffxDestroyContextSEH(ffxFunctions& api, ffxContext* context,
    std::atomic<bool>& dispatchFaulted, bool* faulted = nullptr)
{
    if (faulted)
        *faulted = false;
    if (dispatchFaulted.load(std::memory_order_acquire)) {
        if (faulted)
            *faulted = true;
        return (ffxReturnCode_t)0xFFFFFFFF;
    }
    __try { return api.DestroyContext(context, nullptr); }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        dispatchFaulted.store(true, std::memory_order_release);
        if (faulted)
            *faulted = true;
        return (ffxReturnCode_t)0xFFFFFFFF;
    }
}

// FFX submits composition synchronously from its replacement vkQueuePresentKHR while DXVK holds its own
// queue mutex. Serialize the plugin callback with a separate leaf mutex: acquiring DXVK's queue lock here
// would recursively take the lock already held by the presenter and deadlock.
std::mutex g_ffxGameQueueSubmitMutex;
PFN_vkQueueSubmit g_ffxRealQueueSubmit = nullptr;
VkQueue g_ffxGameQueue = VK_NULL_HANDLE;
VkResult FfxGameQueueSubmit(uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence)
{
    if (!g_ffxRealQueueSubmit || g_ffxGameQueue == VK_NULL_HANDLE)
        return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard<std::mutex> lock(g_ffxGameQueueSubmitMutex);
    return g_ffxRealQueueSubmit(g_ffxGameQueue, submitCount, pSubmits, fence);
}

VkImageLayout ffxStateToLayout(uint32_t state)
{
    switch (state) {
        case FFX_API_RESOURCE_STATE_UNORDERED_ACCESS:
        case FFX_API_RESOURCE_STATE_COMMON:
            return VK_IMAGE_LAYOUT_GENERAL;
        case FFX_API_RESOURCE_STATE_COPY_SRC:
            return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        case FFX_API_RESOURCE_STATE_COPY_DEST:
            return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        default:
            return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

// Route FFX-API context diagnostics into the Streamline log.
static void ffxApiLogMessage(uint32_t type, const wchar_t* message)
{
    if (!message)
        return;
    char buf[1024] = {};
    size_t conv = 0;
    wcstombs_s(&conv, buf, sizeof(buf), message, _TRUNCATE);
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        SL_LOG_ERROR("sl.fsr_g: [FFX] %s", buf);
    } else {
        SL_LOG_WARN("sl.fsr_g: [FFX] %s", buf);
    }
}

// Inverse of ffxStateToLayout: map a resource's current VkImageLayout back to an FFX state enum. Used for
// present-time resources (HUDLessColor) that must be declared to FFX in the layout DXVK already has them in,
// without recording a transition in our evaluate command buffer.
uint32_t layoutToFfxState(VkImageLayout layout)
{
    switch (layout) {
        case VK_IMAGE_LAYOUT_GENERAL:              return FFX_API_RESOURCE_STATE_COMMON;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return FFX_API_RESOURCE_STATE_COPY_SRC;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return FFX_API_RESOURCE_STATE_COPY_DEST;
        default:                                   return FFX_API_RESOURCE_STATE_COMPUTE_READ;
    }
}

std::wstring getPluginDir()
{
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&getPluginDir), &self);
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(self, path, MAX_PATH);
    std::wstring p = path;
    auto slash = p.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : p.substr(0, slash);
}

// Record a layout transition for one image on the SL command buffer; returns the old layout to restore.
struct Barrier { VkImage image; VkImageLayout oldLayout; VkImageLayout newLayout; VkImageAspectFlags aspect; };

void recordBarrier(fsr::FSRContext& ctx, VkCommandBuffer cmd, const Barrier& b)
{
    if (b.oldLayout == b.newLayout || !ctx.vkCmdPipelineBarrier)
        return;
    VkImageMemoryBarrier mb{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    mb.oldLayout = b.oldLayout;
    mb.newLayout = b.newLayout;
    mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    mb.image = b.image;
    mb.subresourceRange = { b.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    ctx.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 0, nullptr, 0, nullptr, 1, &mb);
}

// Build a VkImageCreateInfo from an SL resource description so ffxApiGetImageResourceDescriptionVK works.
VkImageCreateInfo imageInfoFromDesc(const chi::ResourceDescription& desc)
{
    VkImageCreateInfo info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = (VkFormat)desc.nativeFormat;
    info.extent = { desc.width, desc.height, 1 };
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return info;
}

bool loadFFX(fsr::FSRContext& ctx)
{
    if (ctx.ffxModule)
        return true;
    std::wstring dll = getPluginDir() + L"\\amd_fidelityfx_vk.dll";
    ctx.ffxModule = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!ctx.ffxModule) {
        SL_LOG_WARN("sl.fsr_g: amd_fidelityfx_vk.dll not found next to the plugin — FSR disabled");
        return false;
    }
    ffxLoadFunctions(&ctx.ffxApi, ctx.ffxModule);
    if (!ctx.ffxApi.CreateContext || !ctx.ffxApi.DestroyContext || !ctx.ffxApi.Configure || !ctx.ffxApi.Dispatch) {
        SL_LOG_ERROR("sl.fsr_g: amd_fidelityfx_vk.dll missing FFX-API entry points");
        FreeLibrary(ctx.ffxModule);
        ctx.ffxModule = nullptr;
        return false;
    }
    SL_LOG_INFO("sl.fsr_g: loaded amd_fidelityfx_vk.dll");
    return true;
}

// Wrap an SL CommonResource as an FfxApiResource + record its transition to the FFX-expected layout.
FfxApiResource wrapAndTransition(fsr::FSRContext& ctx, VkCommandBuffer cmd, CommonResource& cr, uint32_t ffxState,
    uint32_t usage, bool depth, std::vector<Barrier>& restore)
{
    chi::Resource res = cr;
    if (!res || !res->native)
        return ffxApiGetResourceVK(nullptr, {}, ffxState);
    chi::ResourceDescription desc{};
    ctx.compute->getResourceDescription(res, desc);
    VkImage image = (VkImage)res->native;
    VkImageLayout cur = (VkImageLayout)res->state;
    VkImageLayout want = ffxStateToLayout(ffxState);
    VkImageAspectFlags aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    if (cur != want) {
        recordBarrier(ctx, cmd, { image, cur, want, aspect });
        restore.push_back({ image, want, cur, aspect });
    }
    VkImageCreateInfo info = imageInfoFromDesc(desc);
    FfxApiResourceDescription fdesc = ffxApiGetImageResourceDescriptionVK(image, info, usage);
    return ffxApiGetResourceVK(reinterpret_cast<void*>(image), fdesc, ffxState);
}

// ====================== Frame generation (FFX FrameInterpolationSwapChainVK) =====================

// Serialize prepare on the render thread with interpolation on DXVK's submit thread. FFX mutates shared
// descriptor and command state during both operations, so concurrent dispatch can corrupt GPU work. When both locks
// are required, frameMutex is acquired before this mutex.
std::mutex g_fgDispatchMutex;

bool destroyFgContext(fsr::FSRContext& ctx)
{
    if (ctx.dispatchFaulted.load(std::memory_order_acquire))
        return false;
    std::lock_guard<std::mutex> lock(g_fgDispatchMutex);
    if (!ctx.fgContext)
        return true;
    if (ctx.fgContextTeardownRetryDisabled)
        return false;

    ffxContext context = ctx.fgContext;
    bool faulted = false;
    const ffxReturnCode_t rc = ffxDestroyContextSEH(
        ctx.ffxApi, &context, ctx.dispatchFaulted, &faulted);
    if (rc != FFX_API_RETURN_OK) {
        ctx.dispatchFaulted.store(true, std::memory_order_release);
        ctx.fgContextTeardownRetryDisabled = faulted;
        SL_LOG_ERROR("sl.fsr_g: FG context DestroyContext failed 0x%08X%s",
            (uint32_t)rc, faulted ? " after exception" : "");
        return false;
    }

    ctx.fgContext = nullptr;
    ctx.fgContextTeardownRetryDisabled = false;
    return true;
}

// FFX's swapchain calls this per generated frame to run the interpolation dispatch on our FG context.
ffxReturnCode_t FfxFrameGenDispatchCallback(ffxDispatchDescFrameGeneration* params, void* pUserCtx)
{
    auto* ctx = reinterpret_cast<fsr::FSRContext*>(pUserCtx);
    if (!ctx)
        return FFX_API_RETURN_ERROR;
    std::lock_guard<std::mutex> lock(g_fgDispatchMutex);
    if (!ctx->fgContext)
        return FFX_API_RETURN_ERROR;
    const ffxReturnCode_t result = ffxDispatchSEH(
        ctx->ffxApi, &ctx->fgContext, &params->header, ctx->dispatchFaulted);
    ctx->generatedDispatchSucceeded.store(result == FFX_API_RETURN_OK, std::memory_order_release);
    return result;
}

bool rejectPendingFrame(fsr::FSRContext& ctx, bool* discarded = nullptr)
{
    std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
    if (discarded)
        *discarded = ctx.pendingFrame.prepared;
    ctx.pendingFrame = {};
    return !ctx.dispatchFaulted.load(std::memory_order_acquire);
}

// Grab the three distinct queues FFX needs (gameQueue from DXVK's own present queue captured earlier;
// present/imageAcquire = the two extra graphics queues the interposer injected, located via the
// published kVulkanTable). Lazy: kVulkanTable is only set after the interposer's vkCreateDevice.
bool acquireFgQueues(fsr::FSRContext& ctx)
{
    if (ctx.queuesAcquired)
        return true;
    if (!ctx.vkGetDeviceQueue || ctx.device == VK_NULL_HANDLE)
        return false;

    // The interposer publishes its internal VkTable (layer.h) under kVulkanTable with the injected
    // extra-graphics-queue family + start index. We avoid including the heavy interposer header (it
    // pulls SL's bundled Vulkan headers, which would clash with the FFX ones here) by mirroring only
    // the stable POD head — field order matches layer.h exactly (device, instance, 2 procAddrs, then
    // compute*, then graphics*). graphicsQueueIndex = the start index of the injected extra queues
    // (set by the interposer's vkCreateDevice wrapper); graphicsQueueFamily = their family.
    struct VkTableHead {
        VkDevice device;
        VkInstance instance;
        PFN_vkGetDeviceProcAddr getDeviceProcAddr;
        PFN_vkGetInstanceProcAddr getInstanceProcAddr;
        uint32_t computeQueueIndex, computeQueueFamily, computeQueueCreateFlags;
        uint32_t graphicsQueueIndex, graphicsQueueFamily, graphicsQueueCreateFlags;
    };
    VkTableHead* vk = nullptr;
    if (!param::getPointerParam(api::getContext()->parameters, sl::param::global::kVulkanTable, &vk) || !vk)
        return false;
    const uint32_t family = vk->graphicsQueueFamily;
    const uint32_t extraStart = vk->graphicsQueueIndex;
    if (extraStart == 0) {  // no extra queues injected -> our queue declaration didn't take effect
        SL_LOG_ERROR("sl.fsr_g: interposer injected no extra graphics queues (graphicsQueueIndex==0)");
        return false;
    }
    ctx.extraQueueFamily = family;
    ctx.gameQueueFamily = family;

    ctx.vkGetDeviceQueue(ctx.device, family, extraStart, &ctx.presentQueue);
    ctx.vkGetDeviceQueue(ctx.device, family, extraStart + 1, &ctx.imageAcquireQueue);
    if (ctx.gameQueue == VK_NULL_HANDLE)  // not captured yet -> fall back to the app's index-0 queue
        ctx.vkGetDeviceQueue(ctx.device, family, 0, &ctx.gameQueue);
    if (ctx.presentQueue == VK_NULL_HANDLE || ctx.imageAcquireQueue == VK_NULL_HANDLE) {
        SL_LOG_ERROR("sl.fsr_g: failed to acquire injected FG queues (family %u start %u)", family, extraStart);
        return false;
    }
    ctx.queuesAcquired = true;
    SL_LOG_INFO("sl.fsr_g: FG queues ready (family %u, game=%p present=%p acquire=%p)",
        family, (void*)ctx.gameQueue, (void*)ctx.presentQueue, (void*)ctx.imageAcquireQueue);
    return true;
}

// Create the interpolation context (separate from the swapchain context; the swapchain drives it via
// the dispatch callback). Sized to display+maxRender; recreated on size change.
bool ensureFgContext(fsr::FSRContext& ctx, uint32_t displayW, uint32_t displayH, uint32_t renderW, uint32_t renderH, uint32_t backBufferFormat)
{
    bool colorHDR = false;
    {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        colorHDR = ctx.fgColorHDR;
    }
    if (ctx.fgContext && ctx.fgDisplayW == displayW && ctx.fgDisplayH == displayH &&
        ctx.fgRenderW >= renderW && ctx.fgRenderH >= renderH && ctx.fgBackBufferFormat == backBufferFormat &&
        ctx.fgContextHDR == colorHDR)                    // HDR flag is baked into fgDesc.flags at create
        return true;
    if (ctx.fgContext && !destroyFgContext(ctx))
        return false;
    chi::Device device{}; chi::PhysicalDevice physical{};
    ctx.compute->getDevice(device);
    ctx.compute->getPhysicalDevice(physical);

    ffxCreateBackendVKDesc backendDesc{};
    backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backendDesc.header.pNext = nullptr;
    backendDesc.vkDevice = (VkDevice)device;
    backendDesc.vkPhysicalDevice = (VkPhysicalDevice)physical;
    g_realDeviceProcAddr = ctx.realDeviceProcAddr;
    backendDesc.vkDeviceProcAddr = &DxvkFfxGetDeviceProcAddr;

    ffxCreateContextDescFrameGeneration fgDesc{};
    fgDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    fgDesc.header.pNext = &backendDesc.header;
    fgDesc.flags = colorHDR ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0;
    // Async-compute workloads are disabled (frame generation runs on the game queue).
    // Skyrim's depth is standard [0=near .. 1=far], not reversed-Z, so no FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED.
    fgDesc.displaySize = { displayW, displayH };
    fgDesc.maxRenderSize = { renderW, renderH };
    fgDesc.backBufferFormat = backBufferFormat;

    // Register FFX's global debug callback so context-creation failures include the failing subsystem.
    ffxConfigureDescGlobalDebug1 ffxDbg{};
    ffxDbg.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
    ffxDbg.fpMessage = &ffxApiLogMessage;
    ffxDbg.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS;
    if (ctx.ffxApi.Configure)
        ffxConfigureSEH(ctx.ffxApi, nullptr, &ffxDbg.header, ctx.dispatchFaulted);

    ffxContext newContext = nullptr;
    ffxReturnCode_t rc = ffxCreateContextSEH(
        ctx.ffxApi, &newContext, &fgDesc.header, ctx.dispatchFaulted);
    if (rc != FFX_API_RETURN_OK || !newContext) {
        SL_LOG_ERROR("sl.fsr_g: ffxCreateContext(FG) failed 0x%08X", (uint32_t)rc);
        if (newContext) {
            ctx.fgContext = newContext;
            destroyFgContext(ctx);
        }
        return false;
    }
    ctx.fgContext = newContext;
    ctx.fgContextTeardownRetryDisabled = false;
    ctx.fgDisplayW = displayW; ctx.fgDisplayH = displayH; ctx.fgRenderW = renderW; ctx.fgRenderH = renderH;
    ctx.fgBackBufferFormat = backBufferFormat;
    ctx.fgContextHDR = colorHDR;
    {
        std::lock_guard<std::mutex> lock(ctx.frameMutex);
        ctx.frameID = 0;
        ctx.pendingFrame = {};
    }
    ctx.lastPresentedFrameCount.store(1, std::memory_order_release);
    SL_LOG_INFO("sl.fsr_g: FFX FG context created (display %ux%u render %ux%u)", displayW, displayH, renderW, renderH);
    return true;
}

// Tear down the FG swapchain context (stops its present + interpolation threads, destroys the real
// VkSwapchainKHR it owns) and unlink the interpolation context.
bool destroyFgSwapchain(fsr::FSRContext& ctx)
{
    if (ctx.dispatchFaulted.load(std::memory_order_acquire))
        return false;
    std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
    if (ctx.fgSwapchainTeardownRetryDisabled)
        return false;

    if (!ctx.fgSwapchainContext) {
        if (ctx.fgWrappedSwapchain != VK_NULL_HANDLE) {
            ctx.fgSwapchainTeardownFailed = true;
            ctx.fgSwapchainTeardownRetryDisabled = true;
            SL_LOG_ERROR("sl.fsr_g: cannot destroy wrapped swapchain without its FFX context");
            return false;
        }
        ctx.fgSwapchainTeardownFailed = false;
        return true;
    }

    const uint64_t frameID = ctx.frameID;
    ctx.pendingFrame = {};
    ffxConfigureDescFrameGeneration cfg{};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain = nullptr;
    cfg.frameGenerationEnabled = false;
    cfg.frameID = frameID;
    if (ctx.fgContext) {
        bool faulted = false;
        ffxReturnCode_t rc = FFX_API_RETURN_ERROR;
        std::lock_guard<std::mutex> lock(g_fgDispatchMutex);
        if (ctx.fgContext)
            rc = ffxConfigureSEH(
                ctx.ffxApi, &ctx.fgContext, &cfg.header, ctx.dispatchFaulted, &faulted);
        if (rc != FFX_API_RETURN_OK) {
            ctx.fgSwapchainTeardownFailed = true;
            ctx.fgSwapchainTeardownRetryDisabled = faulted;
            SL_LOG_ERROR("sl.fsr_g: failed to unlink FG swapchain 0x%08X%s",
                (uint32_t)rc, faulted ? " after exception" : "");
            return false;
        }
    }

    ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK waitDesc{};
    waitDesc.header.type = FFX_API_DISPATCH_DESC_TYPE_FGSWAPCHAIN_WAIT_FOR_PRESENTS_VK;
    bool faulted = false;
    ffxReturnCode_t rc = ctx.ffxApi.Dispatch ?
        ffxDispatchSEH(ctx.ffxApi, &ctx.fgSwapchainContext, &waitDesc.header,
            ctx.dispatchFaulted, &faulted) :
        FFX_API_RETURN_ERROR;
    if (rc != FFX_API_RETURN_OK) {
        ctx.fgSwapchainTeardownFailed = true;
        ctx.fgSwapchainTeardownRetryDisabled = faulted;
        SL_LOG_ERROR("sl.fsr_g: WaitForPresents failed 0x%08X%s",
            (uint32_t)rc, faulted ? " after exception" : "");
        return false;
    }

    ffxContext swapchainContext = ctx.fgSwapchainContext;
    faulted = false;
    rc = ctx.ffxApi.DestroyContext ?
        ffxDestroyContextSEH(ctx.ffxApi, &swapchainContext, ctx.dispatchFaulted, &faulted) :
        FFX_API_RETURN_ERROR;
    if (rc != FFX_API_RETURN_OK) {
        ctx.fgSwapchainTeardownFailed = true;
        ctx.fgSwapchainTeardownRetryDisabled = faulted;
        SL_LOG_ERROR("sl.fsr_g: FG swapchain DestroyContext failed 0x%08X%s",
            (uint32_t)rc, faulted ? " after exception" : "");
        return false;
    }

    ctx.fgSwapchainContext = nullptr;
    ctx.fgSwapchainFns = {};
    ctx.fgWrappedSwapchain = VK_NULL_HANDLE;
    ctx.fgSwapchainTeardownFailed = false;
    ctx.fgSwapchainTeardownRetryDisabled = false;
    ctx.lastPresentedFrameCount.store(1, std::memory_order_release);
    return true;
}

// Replace DXVK's swapchain with an FFX FrameInterpolationSwapChain. Writes the wrapped handle into
// *pSwapchain; returns false to fall back to the normal swapchain on any failure.
bool createFgSwapchain(fsr::FSRContext& ctx, VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    if (!ctx.ffxApi.CreateContext || !acquireFgQueues(ctx))
        return false;
    {
        std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
        if (ctx.fgSwapchainContext || ctx.fgWrappedSwapchain != VK_NULL_HANDLE)
            return false;
        ctx.fgSwapchainTeardownFailed = false;
        ctx.fgSwapchainTeardownRetryDisabled = false;
    }

    const uint32_t displayW = pCreateInfo->imageExtent.width, displayH = pCreateInfo->imageExtent.height;
    const uint32_t backBufferFormat = ffxApiGetSurfaceFormatVK(pCreateInfo->imageFormat);
    ctx.fgBackBufferVkFormat = pCreateInfo->imageFormat;
    // Render size is unknown until the first upscale; size FG to display (FFX clamps maxRenderSize).
    if (!ensureFgContext(ctx, displayW, displayH, displayW, displayH, backBufferFormat))
        return false;

    *pSwapchain = VK_NULL_HANDLE;
    if (!g_ffxRealQueueSubmit && ctx.realDeviceProcAddr)
        g_ffxRealQueueSubmit = reinterpret_cast<PFN_vkQueueSubmit>(ctx.realDeviceProcAddr(device, "vkQueueSubmit"));
    g_ffxGameQueue = ctx.gameQueue;

    VkQueueInfoFFXAPI gameQ{};
    gameQ.queue = ctx.gameQueue;
    gameQ.familyIndex = ctx.gameQueueFamily;
    gameQ.submitFunc = g_ffxRealQueueSubmit ? &FfxGameQueueSubmit : nullptr;
    VkQueueInfoFFXAPI presentQ{};
    presentQ.queue = ctx.presentQueue;
    presentQ.familyIndex = ctx.extraQueueFamily;
    VkQueueInfoFFXAPI acquireQ{};
    acquireQ.queue = ctx.imageAcquireQueue;
    acquireQ.familyIndex = ctx.extraQueueFamily;

    ffxCreateContextDescFrameGenerationSwapChainModeVK mode{};
    mode.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_MODE_VK;
    mode.header.pNext = nullptr;
    mode.composeOnPresentQueue = false;  // compose on the game queue (FFX default)

    ffxCreateContextDescFrameGenerationSwapChainVK desc{};
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    desc.header.pNext = &mode.header;
    desc.physicalDevice = ctx.physicalDevice;
    desc.device = device;
    desc.swapchain = pSwapchain;
    desc.allocator = const_cast<VkAllocationCallbacks*>(pAllocator);
    desc.createInfo = *pCreateInfo;
    desc.createInfo.pNext = nullptr;  // FFX rejects DXVK's extension chain (maintenance1 etc.)
    // FFX builds a fresh swapchain; don't hand it DXVK's oldSwapchain (DXVK retires that itself via our
    // DestroySwapchainKHR hook — passing it to FFX too would double-retire it).
    desc.createInfo.oldSwapchain = VK_NULL_HANDLE;
    desc.gameQueue = gameQ;
    desc.presentQueue = presentQ;
    desc.imageAcquireQueue = acquireQ;

    // FFX creates its inner real swapchain here; intercept that re-entrant CreateSwapchainKHR (see the hook)
    // so it bypasses the interposer's other plugins.
    g_creatingFgSwapchain.store(true, std::memory_order_release);
    ffxContext swapchainContext = nullptr;
    ffxReturnCode_t rc = ffxCreateContextSEH(
        ctx.ffxApi, &swapchainContext, &desc.header, ctx.dispatchFaulted);
    g_creatingFgSwapchain.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
        ctx.fgSwapchainContext = swapchainContext;
        ctx.fgWrappedSwapchain = *pSwapchain;
    }
    if (rc != FFX_API_RETURN_OK || !swapchainContext || *pSwapchain == VK_NULL_HANDLE) {
        SL_LOG_ERROR("sl.fsr_g: FG swapchain CreateContext failed 0x%08X", (uint32_t)rc);
        if (swapchainContext) {
            if (destroyFgSwapchain(ctx))
                *pSwapchain = VK_NULL_HANDLE;
        } else if (*pSwapchain != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
            ctx.fgSwapchainTeardownFailed = true;
            ctx.fgSwapchainTeardownRetryDisabled = true;
        }
        return false;
    }

    ctx.fgSwapchainFns = {};
    ctx.fgSwapchainFns.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    ctx.fgSwapchainFns.header.pNext = nullptr;
    if (ffxQuerySEH(ctx.ffxApi, &ctx.fgSwapchainContext, &ctx.fgSwapchainFns.header,
            ctx.dispatchFaulted) != FFX_API_RETURN_OK ||
        !ctx.fgSwapchainFns.pOutQueuePresentKHR) {
        SL_LOG_ERROR("sl.fsr_g: FG swapchain replacement-function query failed");
        if (destroyFgSwapchain(ctx))
            *pSwapchain = VK_NULL_HANDLE;
        return false;
    }

    // Link the interpolation context to this swapchain + register our dispatch callback. Start
    // DISABLED — enabling before an FG-prepare has run interpolates on empty inputs (GPU device-lost);
    // the present hook enables it per prepared frame.
    ffxConfigureDescFrameGeneration cfg{};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain = reinterpret_cast<void*>(ctx.fgWrappedSwapchain);
    cfg.frameGenerationCallback = &FfxFrameGenDispatchCallback;
    cfg.frameGenerationCallbackUserContext = &ctx;
    cfg.frameGenerationEnabled = false;
    cfg.allowAsyncWorkloads = false;
    cfg.generationRect = { 0, 0, (int32_t)displayW, (int32_t)displayH };
    {
        std::lock_guard<std::mutex> lock(ctx.frameMutex);
        cfg.frameID = ctx.frameID;
    }
    ffxReturnCode_t configureRc = FFX_API_RETURN_ERROR;
    {
        std::lock_guard<std::mutex> lock(g_fgDispatchMutex);
        if (ctx.fgContext)
            configureRc = ffxConfigureSEH(
                ctx.ffxApi, &ctx.fgContext, &cfg.header, ctx.dispatchFaulted);
    }
    if (configureRc != FFX_API_RETURN_OK) {
        SL_LOG_ERROR("sl.fsr_g: FG interpolation-context link failed 0x%08X", (uint32_t)configureRc);
        if (!ctx.dispatchFaulted.load(std::memory_order_acquire) && destroyFgSwapchain(ctx))
            *pSwapchain = VK_NULL_HANDLE;
        return false;
    }

    // Frame-pacing tuning on the FG SWAPCHAIN context (distinct from the FG interpolation context above).
    // safetyMargin/varianceFactor shift the pacing algorithm's target frametime; set to FFX's own
    // defaults (0.1ms / 0.1) explicitly.
    {
        FfxApiSwapchainFramePacingTuning pacing{};
        pacing.safetyMarginInMs = 0.1f;
        pacing.varianceFactor   = 0.1f;
        ffxConfigureDescFrameGenerationSwapChainKeyValueVK kv{};
        kv.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_KEYVALUE_VK;
        kv.header.pNext = nullptr;
        kv.key = FFX_API_CONFIGURE_FG_SWAPCHAIN_KEY_FRAMEPACINGTUNING;
        kv.ptr = &pacing;
        ffxReturnCode_t prc = ffxConfigureSEH(
            ctx.ffxApi, &ctx.fgSwapchainContext, &kv.header, ctx.dispatchFaulted);
        SL_LOG_INFO("sl.fsr_g: FG frame-pacing tuning (rc=0x%08X safety=%.3fms variance=%.3f)",
            (uint32_t)prc, pacing.safetyMarginInMs, pacing.varianceFactor);
        if (ctx.dispatchFaulted.load(std::memory_order_acquire))
            return false;
    }

    SL_LOG_INFO("sl.fsr_g: FFX FG swapchain created (%ux%u, wrapped 0x%llX)", displayW, displayH, (unsigned long long)ctx.fgWrappedSwapchain);
    return true;
}

}

sl::Result fsrBeginEvaluation(chi::CommandList /*pCmdList*/, const common::EventData& /*evd*/, const sl::BaseStructure** /*inputs*/, uint32_t /*numInputs*/)
{
    return Result::eOk;
}

sl::Result fsrEndEvaluation(chi::CommandList cmdList, const common::EventData& evd, const sl::BaseStructure** inputs, uint32_t numInputs)
{
    auto& ctx = (*fsr::getContext());
    struct EvaluationGuard
    {
        fsr::FSRContext& ctx;
        bool accepted = false;
        ~EvaluationGuard() { if (!accepted) rejectPendingFrame(ctx); }
    } evaluationGuard{ ctx };

    if (ctx.dispatchFaulted.load(std::memory_order_acquire) || ctx.platform != RenderAPI::eVulkan)
        return Result::eErrorNotInitialized;
    if (!ctx.ffxModule && !loadFFX(ctx))  // lazy load (never in slOnPluginStartup — vkCreateDevice re-entrancy)
        return Result::eErrorNotInitialized;

    sl::Constants* consts{};
    if (!common::getConsts(evd, &consts))
        return Result::eErrorMissingConstants;

    CommonResource depth{}, mvec{};
    // The FG viewport carries no color. The host tags depth, motion vectors, and HUDLessColor on a dedicated
    // frame-generation viewport every frame so FSR frame generation
    // works under any upscaler. Render size is derived from the depth extent.
    getTaggedResource(kBufferTypeDepth, depth, evd.frame, evd.id, false, inputs, numInputs);
    getTaggedResource(kBufferTypeMotionVectors, mvec, evd.frame, evd.id, false, inputs, numInputs);
    CommonResource hudless{};
    getTaggedResource(kBufferTypeHUDLessColor, hudless, evd.frame, evd.id, true, inputs, numInputs);
    if (!depth || !mvec || !hudless)
        return Result::eErrorMissingInputParameter;

    const auto& depthExt = depth.getExtent();
    const uint32_t renderW = depthExt.width;
    const uint32_t renderH = depthExt.height;
    if (!renderW || !renderH)
        return Result::eErrorMissingInputParameter;

    VkCommandBuffer cmd = (VkCommandBuffer)cmdList;
    std::vector<Barrier> restore;

    // Depth + MV feed the FG-prepare; wrap+transition once.
    FfxApiResource depthRes = wrapAndTransition(ctx, cmd, depth, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_DEPTHTARGET, true, restore);
    FfxApiResource mvecRes = wrapAndTransition(ctx, cmd, mvec, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY, false, restore);
    // The motion-vector resource must carry the same dimensions and subrect as depth because FFX reconstructs it against
    // the depth grid, so a size mismatch (e.g. a full-res MV texture vs a render-res depth) misaligns the
    // dilation and leaves MV not covering the frame. Force MV's description to depth's so they're sampled
    // identically (renderSize below already defines the shared valid subrect).
    mvecRes.description.width = depthRes.description.width;
    mvecRes.description.height = depthRes.description.height;
    const FfxApiFloatCoords2D motionVectorScale = { (float)renderW, (float)renderH };
    // consts->jitterOffset is already negated by the host to match FFX's expected -jitter;
    // pass it straight through — negating again was a double-negation that ghosts under motion.
    const FfxApiFloatCoords2D jitterOffset = { consts->jitterOffset.x, consts->jitterOffset.y };

    // FG-prepare: runs whenever the host wants FG, regardless of which upscaler produced the frame. It
    // records this frame's dilated depth/MV + camera/render size for the present-time interpolation.
    // Setting fgGameplayReached lets the present hook bootstrap-install the FFX swapchain; the prepare
    // dispatch itself only runs once that swapchain (and its FG context) exists.
    bool fgEnabled = false;
    {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        fgEnabled = ctx.fgEnabled;
    }
    Result evaluationResult = fgEnabled ? Result::eErrorNotInitialized : Result::eErrorInvalidState;
    if (fgEnabled) {
        ctx.fgGameplayReached.store(true, std::memory_order_release);
        fsr::FrameGenerationFrame frame{};
        bool hudlessWrapped = false;

        // Capture HUDLessColor for present-time UI extraction. Its exact format and current DXVK layout must
        // match the frame-generation backbuffer contract.
        if (chi::Resource hres = hudless; hres && hres->native) {
            chi::ResourceDescription hdesc{};
            ctx.compute->getResourceDescription(hres, hdesc);
            VkImage himage = (VkImage)hres->native;
            const VkFormat hfmt = (VkFormat)hdesc.nativeFormat;
            const bool compatible = ctx.fgBackBufferVkFormat != VK_FORMAT_UNDEFINED &&
                hfmt == ctx.fgBackBufferVkFormat;
            if (compatible) {
                VkImageCreateInfo hinfo = imageInfoFromDesc(hdesc);
                FfxApiResourceDescription hfdesc = ffxApiGetImageResourceDescriptionVK(himage, hinfo, FFX_API_RESOURCE_USAGE_READ_ONLY);
                frame.hudlessResource = ffxApiGetResourceVK(
                    reinterpret_cast<void*>(himage), hfdesc, layoutToFfxState((VkImageLayout)hres->state));
                frame.haveHudless = frame.hudlessResource.resource != nullptr;
                if (frame.haveHudless) {
                    frame.hudlessCompatible = true;
                    hudlessWrapped = true;
                }
            } else {
                SL_LOG_WARN("sl.fsr_g: HUD-less format %d does not match backbuffer format %d",
                    (int)hfmt, (int)ctx.fgBackBufferVkFormat);
            }
        }

        if (!hudlessWrapped)
            evaluationResult = Result::eErrorComputeFailed;

        if (hudlessWrapped && ctx.fgContext && ctx.fgSwapchainContext && ctx.fgWrappedSwapchain != VK_NULL_HANDLE) {
            ffxDispatchDescFrameGenerationPrepare prep{};
            prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            prep.flags = 0;
            prep.commandList = reinterpret_cast<void*>(cmd);
            prep.renderSize = { renderW, renderH };
            prep.jitterOffset = jitterOffset;
            prep.motionVectorScale = motionVectorScale;
            // Real frame delta (QPC between prepares), not a hardcoded 60Hz value:
            // FFX scales its motion field by this, so at uncapped/high FPS a wrong
            // constant mis-scales interpolation motion.
            {
                LARGE_INTEGER qpcNow;
                QueryPerformanceCounter(&qpcNow);
                float deltaMs = 16.6f;
                if (ctx.fgLastPrepareQpc) {
                    LARGE_INTEGER qpcFreq;
                    QueryPerformanceFrequency(&qpcFreq);
                    const double ms = double(qpcNow.QuadPart - ctx.fgLastPrepareQpc) * 1000.0 / double(qpcFreq.QuadPart);
                    deltaMs = (float)std::min(std::max(ms, 1.0), 100.0);
                }
                ctx.fgLastPrepareQpc = qpcNow.QuadPart;
                prep.frameTimeDelta = deltaMs;
            }
            prep.unused_reset = false;
            prep.cameraNear = consts->cameraNear;
            prep.cameraFar = consts->cameraFar;
            prep.cameraFovAngleVertical = consts->cameraFOV;
            prep.viewSpaceToMetersFactor = kSkyrimViewSpaceToMeters;
            prep.depth = depthRes;
            prep.motionVectors = mvecRes;
            {
                std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
                ctx.pendingFrame = {};
                prep.frameID = ctx.frameID;
                frame.frameID = prep.frameID;
                ffxReturnCode_t prepareRc = FFX_API_RETURN_ERROR;
                {
                    std::lock_guard<std::mutex> dispatchLock(g_fgDispatchMutex);
                    if (ctx.fgContext && ctx.fgSwapchainContext && ctx.fgWrappedSwapchain != VK_NULL_HANDLE)
                        prepareRc = ffxDispatchSEH(
                            ctx.ffxApi, &ctx.fgContext, &prep.header, ctx.dispatchFaulted);
                }
                if (prepareRc == FFX_API_RETURN_OK) {
                    frame.prepared = true;
                    ctx.pendingFrame = frame;
                    evaluationResult = Result::eOk;
                    evaluationGuard.accepted = true;
                } else {
                    SL_LOG_ERROR("sl.fsr_g: FrameGenerationPrepare failed 0x%08X", (uint32_t)prepareRc);
                    evaluationResult = Result::eErrorComputeFailed;
                }
            }
        }
    }

    for (auto it = restore.rbegin(); it != restore.rend(); ++it)
        recordBarrier(ctx, cmd, { it->image, it->oldLayout, it->newLayout, it->aspect });

    return evaluationResult;
}

sl::Result slFSRFrameGenerationSetOptions(const sl::ViewportHandle& viewport, const sl::FSRFrameGenOptions& options)
{
    auto& ctx = (*fsr::getContext());
    if (ctx.dispatchFaulted.load(std::memory_order_acquire))
        return Result::eErrorComputeFailed;

    const bool want = options.enabled == Boolean::eTrue;
    bool changed = false;
    {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        changed = want != ctx.fgEnabled;
        if (changed) {
            // A real on/off edge: let the present hook (re)trigger the wrap/unwrap recreate this frame.
            ctx.fgBootstrapInFlight = false;
            ctx.fgWrapFailed = false;
        }
        ctx.fgEnabled = want;
        ctx.fgColorHDR = options.colorBuffersHDR == Boolean::eTrue;
        // Debug overlays applied per-present via ffxConfigureDescFrameGeneration (see slHookVkQueuePresentKHR).
        ctx.fgDebugFlags =
            (options.debugView == Boolean::eTrue ? (uint32_t)FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW : 0u) |
            (options.debugTearLines == Boolean::eTrue ? (uint32_t)FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES : 0u) |
            (options.debugPacingLines == Boolean::eTrue ? (uint32_t)FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_PACING_LINES : 0u);
        ctx.fgOnlyPresentGenerated = options.onlyPresentGenerated == Boolean::eTrue;
    }
    if (changed && !rejectPendingFrame(ctx)) {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        ctx.fgEnabled = false;
        ctx.fgBootstrapInFlight = false;
        ctx.fgWrapFailed = true;
        if (auto* p = api::getContext()->parameters)
            p->set("sl.fsr.fgActive", false);
        return Result::eErrorComputeFailed;
    }
    if (!want)
        ctx.lastPresentedFrameCount.store(1, std::memory_order_release);
    // Publish to the shared param store so the interposer's vkCreateSwapchainKHR dispatch suppresses
    // sl.dlss_g's swapchain hook while FSR FG owns present (prevents the two-present-owner device-lost,
    // enabling in-game FSR-FG ↔ DLSS-G switching).
    if (auto* p = api::getContext()->parameters)
        p->set("sl.fsr.fgActive", want);
    return Result::eOk;
}

sl::Result slFSRGetFrameGenState(const sl::ViewportHandle& /*viewport*/, sl::FSRFrameGenState& state)
{
    auto& ctx = (*fsr::getContext());
    state.status = 0;
    state.numFramesActuallyPresented = ctx.lastPresentedFrameCount.load(std::memory_order_acquire);
    state.estimatedVRAMUsageInBytes = 0;
    return Result::eOk;
}

sl::Result slFSRFrameGenerationDiscardPreparedFrame(const sl::ViewportHandle& /*viewport*/)
{
    auto& ctx = (*fsr::getContext());
    if (ctx.dispatchFaulted.load(std::memory_order_acquire))
        return Result::eErrorComputeFailed;
    bool discarded = false;
    if (!rejectPendingFrame(ctx, &discarded))
        return Result::eErrorComputeFailed;
    return discarded ? Result::eOk : Result::eErrorInvalidState;
}

bool slFSRFrameGenerationOwnsSwapchain(VkSwapchainKHR swapchain)
{
    auto& ctx = (*fsr::getContext());
    std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
    return swapchain != VK_NULL_HANDLE && ctx.fgWrappedSwapchain == swapchain;
}

bool slFSRFrameGenerationCompleteSwapchainTeardown(bool releaseFeatureContext)
{
    auto& ctx = *fsr::getContext();
    if (!destroyFgSwapchain(ctx))
        return false;
    return !releaseFeatureContext || destroyFgContext(ctx);
}

// ============================ Vulkan WSI before-hooks (FG swapchain proxy) =======================
// Registered via fsr.json "hooks". Each runs inside the interposer's vk* wrapper BEFORE the real call;
// setting Skip=true (and returning VK_SUCCESS) replaces the real call with FFX's. All swapchain hooks
// no-op (Skip=false) unless the handle is our wrapped FG swapchain, so a normal (FG-off) swapchain and
// DLSS-G's present path are untouched.

VkResult slHookVkCreateSwapchainKHR(VkDevice Device, const VkSwapchainCreateInfoKHR* CreateInfo,
    const VkAllocationCallbacks* Allocator, VkSwapchainKHR* Swapchain, bool& Skip)
{
    auto& ctx = (*fsr::getContext());
    ctx.device = Device;
    // Re-entrant FFX inner swapchain creation uses the genuine driver and skips the interposer's
    // remaining hooks so sl.dlss_g never wraps FFX's swapchain (the device-lost collision).
    if (g_creatingFgSwapchain.load(std::memory_order_acquire) && ctx.realDeviceProcAddr) {
        auto realCreate = reinterpret_cast<PFN_vkCreateSwapchainKHR>(ctx.realDeviceProcAddr(Device, "vkCreateSwapchainKHR"));
        if (realCreate) {
            VkResult r = realCreate(Device, CreateInfo, Allocator, Swapchain);
            Skip = true;
            SL_LOG_INFO("sl.fsr_g: FFX inner CreateSwapchainKHR routed to genuine loader (rc=%d)", (int)r);
            return r;
        }
    }
    bool fgEnabled = false;
    {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        fgEnabled = ctx.fgEnabled;
    }
    SL_LOG_INFO("sl.fsr_g: CreateSwapchainKHR (fgEnabled=%d) — %s",
        (int)fgEnabled,
        (ctx.platform == RenderAPI::eVulkan && fgEnabled) ? "will wrap" : "pass-through");
    if (ctx.physicalDevice == VK_NULL_HANDLE) {
        chi::PhysicalDevice physical{};
        if (ctx.compute) { ctx.compute->getPhysicalDevice(physical); ctx.physicalDevice = (VkPhysicalDevice)physical; }
    }
    // Wrap with the FFX FG swapchain only while FSR FG is the active method. When FSR FG is
    // off (disabled, or DLSS-G selected) we leave DXVK's plain swapchain so present flows back through the
    // SL interposer — that is the path DLSS-G needs. The present hook drives the recreate that re-enters
    // this hook on every on/off transition, so FSR FG ↔ DLSS-G ↔ disabled switch in-game. Gating on
    // fgEnabled (never set at cold init / menus) also avoids the no-frame-production crash.
    if (ctx.platform != RenderAPI::eVulkan || !fgEnabled) {
        Skip = false;
        return VK_SUCCESS;
    }
    if (ctx.dispatchFaulted.load(std::memory_order_acquire)) {
        Skip = true;
        return VK_ERROR_DEVICE_LOST;
    }
    if (!ctx.ffxModule && !loadFFX(ctx)) {
        Skip = false;
        return VK_SUCCESS;
    }
    {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        ctx.fgBootstrapInFlight = false;
    }
    if (createFgSwapchain(ctx, Device, CreateInfo, Allocator, Swapchain)) {
        Skip = true;  // *Swapchain now holds FFX's wrapped handle
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        ctx.fgWrapFailed = false;
    } else {
        bool unsafeFallback = false;
        {
            std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
            unsafeFallback = ctx.fgSwapchainTeardownFailed || ctx.fgSwapchainContext ||
                ctx.fgWrappedSwapchain != VK_NULL_HANDLE;
        }
        if (unsafeFallback) {
            SL_LOG_ERROR("sl.fsr_g: FG swapchain creation failed with live FFX ownership");
            Skip = true;
            return VK_NOT_READY;
        }
        SL_LOG_WARN("sl.fsr_g: FG swapchain proxy failed; falling back to a normal swapchain");
        Skip = false;
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        ctx.fgWrapFailed = true;  // don't loop on the bootstrap recreate
    }
    return VK_SUCCESS;
}

void slHookVkDestroySwapchainKHR(VkDevice /*Device*/, VkSwapchainKHR Swapchain, const VkAllocationCallbacks* /*Allocator*/, bool& Skip)
{
    auto& ctx = (*fsr::getContext());
    bool wrapped = false;
    {
        std::lock_guard<std::mutex> frameLock(ctx.frameMutex);
        wrapped = Swapchain != VK_NULL_HANDLE && Swapchain == ctx.fgWrappedSwapchain;
    }
    if (wrapped) {
        if (!destroyFgSwapchain(ctx))
            SL_LOG_ERROR("sl.fsr_g: retaining failed FG swapchain teardown");
        Skip = true;
    } else {
        Skip = false;
    }
}

VkResult slHookVkGetSwapchainImagesKHR(VkDevice Device, VkSwapchainKHR Swapchain, uint32_t* Count, VkImage* Images, bool& Skip)
{
    auto& ctx = (*fsr::getContext());
    if (Swapchain == ctx.fgWrappedSwapchain && ctx.dispatchFaulted.load(std::memory_order_acquire)) {
        Skip = true;
        return VK_ERROR_DEVICE_LOST;
    }
    if (Swapchain == ctx.fgWrappedSwapchain && ctx.fgSwapchainFns.pOutGetSwapchainImagesKHR) {
        Skip = true;
        return ctx.fgSwapchainFns.pOutGetSwapchainImagesKHR(Device, Swapchain, Count, Images);
    }
    Skip = false;
    return VK_SUCCESS;
}

VkResult slHookVkAcquireNextImageKHR(VkDevice Device, VkSwapchainKHR Swapchain, uint64_t Timeout,
    VkSemaphore Semaphore, VkFence Fence, uint32_t* ImageIndex, bool& Skip)
{
    auto& ctx = (*fsr::getContext());
    if (Swapchain == ctx.fgWrappedSwapchain && ctx.dispatchFaulted.load(std::memory_order_acquire)) {
        Skip = true;
        return VK_ERROR_DEVICE_LOST;
    }
    if (Swapchain == ctx.fgWrappedSwapchain && ctx.fgSwapchainFns.pOutAcquireNextImageKHR) {
        Skip = true;
        return ctx.fgSwapchainFns.pOutAcquireNextImageKHR(Device, Swapchain, Timeout, Semaphore, Fence, ImageIndex);
    }
    Skip = false;
    return VK_SUCCESS;
}

VkResult slHookVkQueuePresentKHR(VkQueue Queue, const VkPresentInfoKHR* PresentInfo, bool& Skip)
{
    auto& ctx = (*fsr::getContext());

    bool wrapped = false;
    if (ctx.fgWrappedSwapchain != VK_NULL_HANDLE && PresentInfo) {
        for (uint32_t i = 0; i < PresentInfo->swapchainCount; ++i)
            if (PresentInfo->pSwapchains[i] == ctx.fgWrappedSwapchain) { wrapped = true; break; }
    }

    if (wrapped && ctx.dispatchFaulted.load(std::memory_order_acquire)) {
        Skip = true;
        return VK_ERROR_DEVICE_LOST;
    }

    // Capture DXVK's real present queue once for FFX's gameQueue (needed before we ever wrap).
    if (ctx.gameQueue == VK_NULL_HANDLE)
        ctx.gameQueue = Queue;

    bool fgEnabled = false;
    uint32_t fgDebugFlags = 0;
    bool fgOnlyPresentGenerated = false;
    bool needRecreate = false;
    {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        fgEnabled = ctx.fgEnabled;
        fgDebugFlags = ctx.fgDebugFlags;
        fgOnlyPresentGenerated = ctx.fgOnlyPresentGenerated;
        needRecreate = ctx.fgGameplayReached.load(std::memory_order_acquire) &&
            !ctx.fgBootstrapInFlight && !ctx.fgWrapFailed &&
            ((fgEnabled && ctx.fgWrappedSwapchain == VK_NULL_HANDLE && !wrapped) ||
             (!fgEnabled && wrapped));
        if (needRecreate)
            ctx.fgBootstrapInFlight = true;
    }

    // Bidirectional wrap/unwrap allows FSR FG, DLSS-G, and disabled states to switch in game. The desired state is
    // wrapped exactly while FSR FG is enabled; the actual state is whether this swapchain is our FFX handle. On a
    // mismatch, present this frame on its real path then return VK_SUBOPTIMAL_KHR so DXVK recreates its
    // swapchain — re-entering CreateSwapchainKHR, which wraps (fgEnabled) or leaves it plain (returning
    // present to SL for DLSS-G). Presenting first avoids the dangling-fence hang of a no-present OUT_OF_DATE.
    // fgBootstrapInFlight de-bounces to one trigger per recreate; fgWrapFailed stops a wrap-fail loop.
    if (needRecreate) {
        Skip = true;
        // Present this frame on the path that owns it now: FFX's present if currently wrapped (turning OFF),
        // else the genuine driver present (turning ON, still a plain swapchain).
        VkResult pr;
        if (wrapped && ctx.fgSwapchainFns.pOutQueuePresentKHR)
            pr = ctx.fgSwapchainFns.pOutQueuePresentKHR(Queue, PresentInfo);
        else if (ctx.realQueuePresentKHR)
            pr = ctx.realQueuePresentKHR(Queue, PresentInfo);
        else
            pr = VK_SUCCESS;
        ctx.lastPresentedFrameCount.store(1, std::memory_order_release);
        SL_LOG_INFO("sl.fsr_g: recreate to %s FG (present rc=%d)", fgEnabled ? "ENABLE" : "DISABLE", (int)pr);
        return pr < 0 ? pr : VK_SUBOPTIMAL_KHR;
    }

    // Not our wrapped handle (FSR FG off, or an overlay's own swapchain) — pass through untouched.
    if (!wrapped) {
        Skip = false;
        return VK_SUCCESS;
    }

    fsr::FrameGenerationFrame frame{};
    uint64_t frameID = 0;
    {
        std::lock_guard<std::mutex> lock(ctx.frameMutex);
        frameID = ctx.frameID++;
        if (ctx.pendingFrame.prepared && ctx.pendingFrame.frameID == frameID)
            frame = ctx.pendingFrame;
        ctx.pendingFrame = {};
    }

    // Interpolate only after evaluation prepared and accepted every resource for this frame.
    // Other frames remain on the wrapped swapchain's 1:1 present path.
    bool generatedFrameConfigured = false;
    if (ctx.fgContext) {
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.swapChain = reinterpret_cast<void*>(ctx.fgWrappedSwapchain);
        cfg.frameGenerationCallback = &FfxFrameGenDispatchCallback;
        cfg.frameGenerationCallbackUserContext = &ctx;
        cfg.frameGenerationEnabled = fgEnabled && frame.prepared;
        cfg.allowAsyncWorkloads = false;
        cfg.flags = fgDebugFlags;
        cfg.onlyPresentGenerated = fgOnlyPresentGenerated;
        cfg.generationRect = { 0, 0, (int32_t)ctx.fgDisplayW, (int32_t)ctx.fgDisplayH };
        // HUDLessColor: the scene without UI, captured this frame in fsrEndEvaluation. With it, FFX extracts the
        // UI (backbuffer - hudless) and only interpolates the scene, so the HUD doesn't ghost. Only pass it when
        // the resource is in the exact format baked into the interpolation context.
        if (frame.haveHudless && frame.hudlessCompatible)
            cfg.HUDLessColor = frame.hudlessResource;
        cfg.frameID = frameID;
        {
            std::lock_guard<std::mutex> lock(g_fgDispatchMutex);
            const ffxReturnCode_t configureResult = ctx.fgContext ?
                ffxConfigureSEH(ctx.ffxApi, &ctx.fgContext, &cfg.header,
                    ctx.dispatchFaulted) : FFX_API_RETURN_ERROR;
            generatedFrameConfigured = cfg.frameGenerationEnabled && configureResult == FFX_API_RETURN_OK;
        }
    }

    if (ctx.dispatchFaulted.load(std::memory_order_acquire)) {
        Skip = true;
        return VK_ERROR_DEVICE_LOST;
    }

    ctx.generatedDispatchSucceeded.store(false, std::memory_order_release);
    ctx.lastPresentedFrameCount.store(1, std::memory_order_release);
    VkResult r = ctx.fgSwapchainFns.pOutQueuePresentKHR ?
        ctx.fgSwapchainFns.pOutQueuePresentKHR(Queue, PresentInfo) : VK_ERROR_INITIALIZATION_FAILED;

    if (generatedFrameConfigured && r >= 0 &&
        ctx.generatedDispatchSucceeded.load(std::memory_order_acquire))
        ctx.lastPresentedFrameCount.store(2, std::memory_order_release);
    Skip = true;
    return r;
}

bool slOnPluginStartup(const char* jsonConfig, void* device)
{
    SL_PLUGIN_COMMON_STARTUP();

    auto& ctx = (*fsr::getContext());
    auto parameters = api::getContext()->parameters;

    if (!param::getPointerParam(parameters, param::common::kPFunRegisterEvaluateCallbacks, &ctx.registerEvaluateCallbacks))
    {
        SL_LOG_ERROR("sl.fsr_g: failed to obtain kPFunRegisterEvaluateCallbacks");
        return false;
    }
    ctx.registerEvaluateCallbacks(kFeatureFSR_G, fsrBeginEvaluation, fsrEndEvaluation);

    json& config = *(json*)api::getContext()->loaderConfig;
    uint32_t deviceType{};
    int appId{};
    config.at("appId").get_to(appId);
    config.at("deviceType").get_to(deviceType);
    ctx.platform = (RenderAPI)deviceType;

    if (!param::getPointerParam(parameters, sl::param::common::kComputeAPI, &ctx.compute))
    {
        SL_LOG_ERROR("sl.fsr_g: failed to obtain compute interface");
        return false;
    }

    // Resolve un-interposed device functions for the FFX backend and barriers from the genuine
    // driver loader in System32. GetModuleHandle("vulkan-1.dll") returns
    // sl.interposer.dll. FFX's FrameInterpolationSwapChain creates + presents its swapchain through these
    // function pointers; routing them to the genuine loader makes the FFX swapchain invisible to the SL
    // interposer's hook dispatch (so sl.dlss_g's swapchain hook never touches it), which is what lets DLSS-G
    // and FSR FG coexist and be switched in-game. The interposer already self-loaded this same genuine
    // module, so LoadLibrary just bumps its refcount.
    wchar_t sysDir[MAX_PATH]{};
    std::wstring genuineVkPath = L"vulkan-1.dll";
    if (GetSystemDirectoryW(sysDir, MAX_PATH))
        genuineVkPath = std::wstring(sysDir) + L"\\vulkan-1.dll";
    if (HMODULE vk = LoadLibraryW(genuineVkPath.c_str())) {
        auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(vk, "vkGetInstanceProcAddr")));
        chi::Instance instance{};
        ctx.compute->getInstance(instance);
        if (gipa) {
            ctx.realDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa((VkInstance)instance, "vkGetDeviceProcAddr"));
            chi::Device dev{};
            ctx.compute->getDevice(dev);
            if (ctx.realDeviceProcAddr) {
                ctx.vkCmdPipelineBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(ctx.realDeviceProcAddr((VkDevice)dev, "vkCmdPipelineBarrier"));
                ctx.vkGetDeviceQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(ctx.realDeviceProcAddr((VkDevice)dev, "vkGetDeviceQueue"));
                // Real ICD present, for the present-then-SUBOPTIMAL recreate trigger (see the present hook).
                ctx.realQueuePresentKHR = reinterpret_cast<PFN_vkQueuePresentKHR>(ctx.realDeviceProcAddr((VkDevice)dev, "vkQueuePresentKHR"));
            }
        }
    }

    SL_LOG_INFO("sl.fsr_g plugin started");
    return true;
}

void slOnPluginShutdown()
{
    auto& ctx = (*fsr::getContext());
    if (!destroyFgSwapchain(ctx) || !destroyFgContext(ctx)) {
        SL_LOG_ERROR("sl.fsr_g: retaining FFX runtime after failed context teardown");
        return;
    }
    if (ctx.ffxModule)
        FreeLibrary(ctx.ffxModule);
    ctx.ffxModule = nullptr;
    ctx.registerEvaluateCallbacks(kFeatureFSR_G, nullptr, nullptr);
    plugin::onShutdown(api::getContext());
}

void updateEmbeddedJSON(json& config)
{
    common::SystemCaps* caps = {};
    param::getPointerParam(api::getContext()->parameters, sl::param::common::kSystemCaps, &caps);
    common::PFunUpdateCommonEmbeddedJSONConfig* updateCommonEmbeddedJSONConfig{};
    param::getPointerParam(api::getContext()->parameters, sl::param::common::kPFunUpdateCommonEmbeddedJSONConfig, &updateCommonEmbeddedJSONConfig);
    if (caps && updateCommonEmbeddedJSONConfig)
    {
        common::PluginInfo info{};
        // SHA is mandatory: updateCommonEmbeddedJSONConfig writes config["sha"]=info.SHA, and the plugin
        // manager later does config.at("sha").get_to(string) — a null SHA makes that throw and kills slInit.
        info.SHA = GIT_LAST_COMMIT_SHORT;
        info.minOS = sl::Version(10, 0, 0);
        info.minGPUArchitecture = 0;
        updateCommonEmbeddedJSONConfig(&config, info);
    }
    // Request 2 EXTRA graphics-family queues from the interposer's vkCreateDevice (which sums this
    // over loaded plugins' configs). FFX's FrameInterpolationSwapChain needs distinct
    // present + image-acquire queues, exclusive of DXVK's own. Set AFTER updateCommonEmbeddedJSONConfig
    // so it isn't overwritten; the plugin reads the injected family/start-index back from kVulkanTable.
    config["external"]["vk"]["device"]["queues"]["graphics"]["count"] = 2;
}

SL_EXPORT void* slGetPluginFunction(const char* functionName)
{
    SL_EXPORT_FUNCTION(slOnPluginLoad);
    SL_EXPORT_FUNCTION(slOnPluginShutdown);
    SL_EXPORT_FUNCTION(slOnPluginStartup);

    SL_EXPORT_FUNCTION(slFSRFrameGenerationSetOptions);
    SL_EXPORT_FUNCTION(slFSRGetFrameGenState);
    SL_EXPORT_FUNCTION(slFSRFrameGenerationDiscardPreparedFrame);
    SL_EXPORT_FUNCTION(slFSRFrameGenerationOwnsSwapchain);
    SL_EXPORT_FUNCTION(slFSRFrameGenerationCompleteSwapchainTeardown);

    // Vulkan WSI before-hooks (declared in fsr.json) — the FG swapchain proxy.
    SL_EXPORT_FUNCTION(slHookVkCreateSwapchainKHR);
    SL_EXPORT_FUNCTION(slHookVkDestroySwapchainKHR);
    SL_EXPORT_FUNCTION(slHookVkGetSwapchainImagesKHR);
    SL_EXPORT_FUNCTION(slHookVkAcquireNextImageKHR);
    SL_EXPORT_FUNCTION(slHookVkQueuePresentKHR);

    return nullptr;
}

}
