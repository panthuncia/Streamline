/*
* Community Shaders sl.fsr plugin — AMD FidelityFX FSR3 upscaling (+ frame generation later).
*
* Drives FFX-API (amd_fidelityfx_vk.dll) on DXVK's VkDevice through Streamline's evaluate mechanism.
* The host tags ScalingInputColor / ScalingOutputColor / Depth / MotionVectors + sets common Constants
* (jitter, camera) and calls slEvaluateFeature(kFeatureFSR); fsrEndEvaluation records ffxDispatch(upscale)
* onto the SL-provided command buffer. Frame generation (present-hook insertion) is added in a later phase.
*/

#include <cstring>
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
#include "source/plugins/sl.fsr/versions.h"
#include "source/plugins/sl.common/commonInterface.h"
#include "external/json/include/nlohmann/json.hpp"
#include "_artifacts/json/fsr_json.h"
#include "_artifacts/gitVersion.h"

#include "include/sl_fsr.h"

// FFX-API (descriptor structs for the prebuilt amd_fidelityfx_vk.dll, resolved at runtime). FFX now
// lives inside the fork under external/fidelityfx-sdk.
#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_api_loader.h"
#include "ffx_api/ffx_upscale.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/vk/ffx_api_vk.h"

using json = nlohmann::json;

namespace sl
{
namespace fsr
{
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
    ffxContext upscaleContext = nullptr;
    uint32_t ctxRenderW = 0, ctxRenderH = 0, ctxDisplayW = 0, ctxDisplayH = 0;
    bool dispatchFaulted = false;

    // Resolved device functions for the FFX backend + our barriers.
    PFN_vkGetDeviceProcAddr realDeviceProcAddr = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;

    std::mutex optionsMutex;
    std::map<uint32_t, FSROptions> upscaleOptions;
    std::map<uint32_t, FSRFrameGenOptions> fgOptions;

    // --- Frame generation (FFX FrameInterpolationSwapChainVK proxy) -----------------------------
    // The host enables FSR FG (slFSRFrameGenerationSetOptions) and triggers a DXVK swapchain
    // recreate; our eVulkan_CreateSwapchainKHR before-hook then replaces DXVK's swapchain with an
    // FFX FrameInterpolationSwapChain that interpolates + double-presents internally. The other WSI
    // before-hooks (GetSwapchainImages/AcquireNextImage/Present/DestroySwapchain) route the wrapped
    // handle to FFX's replacement functions. Per-frame depth/MV/camera is recorded by an FG-prepare
    // dispatch appended to the upscale (fsrEndEvaluation).
    bool fgEnabled = false;                  // host requested FSR FG for the active viewport
    bool fgPreparedThisFrame = false;        // an FG-prepare ran this frame -> safe to interpolate
    bool fgColorHDR = false;
    bool fgRecreateInFlight = false;         // a wrap/unwrap recreate was triggered, awaiting the hook
    bool fgWrapFailed = false;               // createFgSwapchain failed; suppress retrigger until toggled
    bool fgUpscaleSinceLastPresent = false;  // an upscale ran this present cycle -> in gameplay (gate the
                                             // initial wrap: building the FFX swapchain at a menu/loading
                                             // screen, with no frame production, deadlocks its init)
    uint64_t frameID = 0;
    PFN_vkQueuePresentKHR realQueuePresentKHR = nullptr;
    ffxContext fgContext = nullptr;          // interpolation context (ffxCreateContextDescFrameGeneration)
    ffxContext fgSwapchainContext = nullptr; // swapchain-replacement context (FGSWAPCHAIN_VK)
    VkSwapchainKHR fgWrappedSwapchain = VK_NULL_HANDLE;
    ffxQueryDescSwapchainReplacementFunctionsVK fgSwapchainFns{};
    uint32_t fgDisplayW = 0, fgDisplayH = 0, fgRenderW = 0, fgRenderH = 0, fgBackBufferFormat = 0;

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

static std::string JSON = std::string(fsr_json, &fsr_json[fsr_json_len]);

SL_PLUGIN_DEFINE("sl.fsr", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(0, 0, 1), JSON.c_str(), updateEmbeddedJSON, fsr, FSRContext)

namespace
{
// --- DXVK ⇄ unmodified prebuilt-FFX VK backend shim (ported from CS FidelityFX.cpp) -------------
// The stock amd_fidelityfx_vk.dll calls two functions that resolve to NULL on DXVK without null-checks:
//   vkGetBufferMemoryRequirements2KHR (DXVK promoted it to core, dropped the KHR alias) and
//   vkCmdWriteBufferMarker(2)AMD (DXVK enumerates VK_AMD_buffer_marker but leaves the entries NULL).
// Route the first to the core entry; stub the markers to a no-op so the UNMODIFIED DLL runs on DXVK.
PFN_vkGetDeviceProcAddr g_realDeviceProcAddr = nullptr;

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

// SEH-guarded FFX calls — kept in their OWN functions so __try has no C++ objects requiring unwinding
// (C2712). The stock FFX DLL can access-violate on DXVK; latch the feature off instead of crashing.
ffxReturnCode_t ffxCreateContextSEH(ffxFunctions& api, ffxContext* context, ffxCreateContextDescHeader* desc)
{
    __try { return api.CreateContext(context, desc, nullptr); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return (ffxReturnCode_t)0xFFFFFFFF; }
}
ffxReturnCode_t ffxDispatchSEH(ffxFunctions& api, ffxContext* context, ffxDispatchDescHeader* desc)
{
    __try { return api.Dispatch(context, desc); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return (ffxReturnCode_t)0xFFFFFFFF; }
}
ffxReturnCode_t ffxConfigureSEH(ffxFunctions& api, ffxContext* context, ffxConfigureDescHeader* desc)
{
    __try { return api.Configure(context, desc); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return (ffxReturnCode_t)0xFFFFFFFF; }
}
ffxReturnCode_t ffxQuerySEH(ffxFunctions& api, ffxContext* context, ffxQueryDescHeader* desc)
{
    __try { return api.Query(context, desc); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return (ffxReturnCode_t)0xFFFFFFFF; }
}
ffxReturnCode_t ffxDestroyContextSEH(ffxFunctions& api, ffxContext* context)
{
    __try { return api.DestroyContext(context, nullptr); }
    __except (EXCEPTION_EXECUTE_HANDLER) { return (ffxReturnCode_t)0xFFFFFFFF; }
}

// FFX's FG swapchain submits composition work on the game queue from its own present thread, racing
// DXVK's submits on the same queue. Serialize FFX's own access through a leaf mutex (matches CS
// FidelityFX::FfxGameQueueSubmit — deliberately NOT routed through any DXVK lock, which would deadlock
// when called from FFX's present thread).
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
        SL_LOG_WARN("sl.fsr: amd_fidelityfx_vk.dll not found next to the plugin — FSR disabled");
        return false;
    }
    ffxLoadFunctions(&ctx.ffxApi, ctx.ffxModule);
    if (!ctx.ffxApi.CreateContext || !ctx.ffxApi.DestroyContext || !ctx.ffxApi.Configure || !ctx.ffxApi.Dispatch) {
        SL_LOG_ERROR("sl.fsr: amd_fidelityfx_vk.dll missing FFX-API entry points");
        FreeLibrary(ctx.ffxModule);
        ctx.ffxModule = nullptr;
        return false;
    }
    SL_LOG_INFO("sl.fsr: loaded amd_fidelityfx_vk.dll");
    return true;
}

bool ensureUpscaleContext(fsr::FSRContext& ctx, uint32_t renderW, uint32_t renderH, uint32_t displayW, uint32_t displayH)
{
    if (ctx.upscaleContext && ctx.ctxRenderW >= renderW && ctx.ctxRenderH >= renderH &&
        ctx.ctxDisplayW == displayW && ctx.ctxDisplayH == displayH)
        return true;
    if (ctx.upscaleContext) {
        ctx.ffxApi.DestroyContext(&ctx.upscaleContext, nullptr);
        ctx.upscaleContext = nullptr;
    }

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

    ffxCreateContextDescUpscale upscaleDesc{};
    upscaleDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    upscaleDesc.header.pNext = &backendDesc.header;
    upscaleDesc.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    upscaleDesc.maxRenderSize = { renderW, renderH };
    upscaleDesc.maxUpscaleSize = { displayW, displayH };
    upscaleDesc.fpMessage = nullptr;

    ffxReturnCode_t rc = ffxCreateContextSEH(ctx.ffxApi, &ctx.upscaleContext, &upscaleDesc.header);
    if (rc != FFX_API_RETURN_OK) {
        SL_LOG_ERROR("sl.fsr: ffxCreateContext(upscale) failed 0x%08X", (uint32_t)rc);
        ctx.upscaleContext = nullptr;
        return false;
    }
    ctx.ctxRenderW = renderW; ctx.ctxRenderH = renderH; ctx.ctxDisplayW = displayW; ctx.ctxDisplayH = displayH;
    SL_LOG_INFO("sl.fsr: FFX upscale context created (render %ux%u display %ux%u)", renderW, renderH, displayW, displayH);
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

// FFX's swapchain calls this per generated frame to run the interpolation dispatch on our FG context.
ffxReturnCode_t FfxFrameGenDispatchCallback(ffxDispatchDescFrameGeneration* params, void* pUserCtx)
{
    auto* ctx = reinterpret_cast<fsr::FSRContext*>(pUserCtx);
    if (!ctx || !ctx->fgContext)
        return FFX_API_RETURN_ERROR;
    return ffxDispatchSEH(ctx->ffxApi, &ctx->fgContext, &params->header);
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
    // (wrapper.cpp:900); graphicsQueueFamily = their family.
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
        SL_LOG_ERROR("sl.fsr: interposer injected no extra graphics queues (graphicsQueueIndex==0)");
        return false;
    }
    ctx.extraQueueFamily = family;
    ctx.gameQueueFamily = family;

    ctx.vkGetDeviceQueue(ctx.device, family, extraStart, &ctx.presentQueue);
    ctx.vkGetDeviceQueue(ctx.device, family, extraStart + 1, &ctx.imageAcquireQueue);
    if (ctx.gameQueue == VK_NULL_HANDLE)  // not captured yet -> fall back to the app's index-0 queue
        ctx.vkGetDeviceQueue(ctx.device, family, 0, &ctx.gameQueue);
    if (ctx.presentQueue == VK_NULL_HANDLE || ctx.imageAcquireQueue == VK_NULL_HANDLE) {
        SL_LOG_ERROR("sl.fsr: failed to acquire injected FG queues (family %u start %u)", family, extraStart);
        return false;
    }
    ctx.queuesAcquired = true;
    SL_LOG_INFO("sl.fsr: FG queues ready (family %u, game=%p present=%p acquire=%p)",
        family, (void*)ctx.gameQueue, (void*)ctx.presentQueue, (void*)ctx.imageAcquireQueue);
    return true;
}

// Create the interpolation context (separate from the swapchain context; the swapchain drives it via
// the dispatch callback). Sized to display+maxRender; recreated on size change.
bool ensureFgContext(fsr::FSRContext& ctx, uint32_t displayW, uint32_t displayH, uint32_t renderW, uint32_t renderH, uint32_t backBufferFormat)
{
    if (ctx.fgContext && ctx.fgDisplayW == displayW && ctx.fgDisplayH == displayH &&
        ctx.fgRenderW >= renderW && ctx.fgRenderH >= renderH && ctx.fgBackBufferFormat == backBufferFormat)
        return true;
    if (ctx.fgContext) {
        ffxDestroyContextSEH(ctx.ffxApi, &ctx.fgContext);
        ctx.fgContext = nullptr;
    }
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
    fgDesc.flags = ctx.fgColorHDR ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0;
    fgDesc.displaySize = { displayW, displayH };
    fgDesc.maxRenderSize = { renderW, renderH };
    fgDesc.backBufferFormat = backBufferFormat;

    ffxReturnCode_t rc = ffxCreateContextSEH(ctx.ffxApi, &ctx.fgContext, &fgDesc.header);
    if (rc != FFX_API_RETURN_OK) {
        SL_LOG_ERROR("sl.fsr: ffxCreateContext(FG) failed 0x%08X", (uint32_t)rc);
        ctx.fgContext = nullptr;
        return false;
    }
    ctx.fgDisplayW = displayW; ctx.fgDisplayH = displayH; ctx.fgRenderW = renderW; ctx.fgRenderH = renderH;
    ctx.fgBackBufferFormat = backBufferFormat;
    ctx.frameID = 0;
    SL_LOG_INFO("sl.fsr: FFX FG context created (display %ux%u render %ux%u)", displayW, displayH, renderW, renderH);
    return true;
}

// Tear down the FG swapchain context (stops its present + interpolation threads, destroys the real
// VkSwapchainKHR it owns) and unlink the interpolation context.
void destroyFgSwapchain(fsr::FSRContext& ctx)
{
    if (ctx.fgContext) {
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.swapChain = nullptr;
        cfg.frameGenerationEnabled = false;
        cfg.frameID = ctx.frameID;
        ffxConfigureSEH(ctx.ffxApi, &ctx.fgContext, &cfg.header);
    }
    if (ctx.fgSwapchainContext) {
        ffxDestroyContextSEH(ctx.ffxApi, &ctx.fgSwapchainContext);
        ctx.fgSwapchainContext = nullptr;
    }
    ctx.fgSwapchainFns = {};
    ctx.fgWrappedSwapchain = VK_NULL_HANDLE;
}

// Replace DXVK's swapchain with an FFX FrameInterpolationSwapChain. Writes the wrapped handle into
// *pSwapchain; returns false to fall back to the normal swapchain on any failure.
bool createFgSwapchain(fsr::FSRContext& ctx, VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    if (!ctx.ffxApi.CreateContext || !acquireFgQueues(ctx))
        return false;

    const uint32_t displayW = pCreateInfo->imageExtent.width, displayH = pCreateInfo->imageExtent.height;
    const uint32_t backBufferFormat = ffxApiGetSurfaceFormatVK(pCreateInfo->imageFormat);
    ctx.fgBackBufferFormat = backBufferFormat;
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
    mode.composeOnPresentQueue = false;

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

    ffxReturnCode_t rc = ffxCreateContextSEH(ctx.ffxApi, &ctx.fgSwapchainContext, &desc.header);
    if (rc != FFX_API_RETURN_OK || !ctx.fgSwapchainContext) {
        SL_LOG_ERROR("sl.fsr: FG swapchain CreateContext failed 0x%08X", (uint32_t)rc);
        ctx.fgSwapchainContext = nullptr;
        return false;
    }

    ctx.fgSwapchainFns = {};
    ctx.fgSwapchainFns.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    ctx.fgSwapchainFns.header.pNext = nullptr;
    if (ffxQuerySEH(ctx.ffxApi, &ctx.fgSwapchainContext, &ctx.fgSwapchainFns.header) != FFX_API_RETURN_OK ||
        !ctx.fgSwapchainFns.pOutQueuePresentKHR) {
        SL_LOG_ERROR("sl.fsr: FG swapchain replacement-function query failed");
        ffxDestroyContextSEH(ctx.ffxApi, &ctx.fgSwapchainContext);
        ctx.fgSwapchainContext = nullptr;
        return false;
    }
    ctx.fgWrappedSwapchain = *pSwapchain;

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
    cfg.frameID = ctx.frameID;
    ffxConfigureSEH(ctx.ffxApi, &ctx.fgContext, &cfg.header);

    SL_LOG_INFO("sl.fsr: FFX FG swapchain created (%ux%u, wrapped 0x%llX)", displayW, displayH, (unsigned long long)ctx.fgWrappedSwapchain);
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
    if (ctx.dispatchFaulted || ctx.platform != RenderAPI::eVulkan)
        return Result::eErrorNotInitialized;
    if (!ctx.ffxModule && !loadFFX(ctx))  // lazy load (never in slOnPluginStartup — vkCreateDevice re-entrancy)
        return Result::eErrorNotInitialized;

    sl::Constants* consts{};
    if (!common::getConsts(evd, &consts))
        return Result::eErrorMissingConstants;

    CommonResource colorIn{}, colorOut{}, depth{}, mvec{};
    getTaggedResource(kBufferTypeScalingInputColor, colorIn, evd.frame, evd.id, false, inputs, numInputs);
    getTaggedResource(kBufferTypeScalingOutputColor, colorOut, evd.frame, evd.id, false, inputs, numInputs);
    getTaggedResource(kBufferTypeDepth, depth, evd.frame, evd.id, false, inputs, numInputs);
    getTaggedResource(kBufferTypeMotionVectors, mvec, evd.frame, evd.id, false, inputs, numInputs);
    if (!colorIn || !colorOut || !depth || !mvec)
        return Result::eErrorMissingInputParameter;

    const auto& inExt = colorIn.getExtent();
    const auto& outExt = colorOut.getExtent();
    const uint32_t renderW = inExt.width, renderH = inExt.height;
    const uint32_t outputW = outExt.width, outputH = outExt.height;
    if (!renderW || !renderH || !outputW || !outputH)
        return Result::eErrorMissingInputParameter;

    float sharpness = 0.0f;
    {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        auto it = ctx.upscaleOptions.find((uint32_t)evd.id);
        if (it != ctx.upscaleOptions.end()) {
            if (it->second.mode == FSRMode::eOff)
                return Result::eOk;
            sharpness = it->second.sharpness;
        }
    }

    if (!ensureUpscaleContext(ctx, renderW, renderH, outputW, outputH)) {
        ctx.dispatchFaulted = true;
        return Result::eErrorExceptionHandler;
    }

    VkCommandBuffer cmd = (VkCommandBuffer)cmdList;
    std::vector<Barrier> restore;

    ffxDispatchDescUpscale dp{};
    dp.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dp.commandList = reinterpret_cast<void*>(cmd);
    dp.color = wrapAndTransition(ctx, cmd, colorIn, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY, false, restore);
    dp.depth = wrapAndTransition(ctx, cmd, depth, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_DEPTHTARGET, true, restore);
    dp.motionVectors = wrapAndTransition(ctx, cmd, mvec, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY, false, restore);
    dp.reactive = ffxApiGetResourceVK(nullptr, {}, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dp.transparencyAndComposition = ffxApiGetResourceVK(nullptr, {}, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dp.exposure = ffxApiGetResourceVK(nullptr, {}, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dp.output = wrapAndTransition(ctx, cmd, colorOut, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV, false, restore);

    dp.motionVectorScale = { (float)renderW, (float)renderH };
    dp.renderSize = { renderW, renderH };
    dp.upscaleSize = { outputW, outputH };
    // consts->jitterOffset is ALREADY negated by the host (CS EvaluateFSR), matching the SL/DLSS
    // convention and FFX's expected -jitter. Pass it straight through — do NOT negate again (that was
    // a double-negation giving +jitter → ghosting/instability under motion).
    dp.jitterOffset = { consts->jitterOffset.x, consts->jitterOffset.y };
    dp.frameTimeDelta = 16.6f;
    dp.cameraFar = consts->cameraFar;
    dp.cameraNear = consts->cameraNear;
    dp.enableSharpening = sharpness > 0.0f;
    dp.sharpness = sharpness;
    dp.cameraFovAngleVertical = consts->cameraFOV;
    dp.viewSpaceToMetersFactor = 0.01428222656f;
    dp.reset = false;
    dp.preExposure = 1.0f;
    dp.flags = 0;

    ffxReturnCode_t rc = ffxDispatchSEH(ctx.ffxApi, &ctx.upscaleContext, &dp.header);

    // An upscale ran -> we're in gameplay (post-processing), so the present hook may now safely build the
    // FFX FG swapchain. Set regardless of wrap state so the very first wrap is gated on real frame production.
    if (rc == FFX_API_RETURN_OK && ctx.fgEnabled)
        ctx.fgUpscaleSinceLastPresent = true;

    // FG-prepare on the interpolation context: records this frame's dilated depth/MV + camera/render
    // size that the present-time interpolation reads. Reuses the upscale's already-transitioned
    // depth/MV resources (still in COMPUTE_READ here, before the restore below), the same frameID the
    // following present consumes, and the same SL command buffer. Only when the FG swapchain is live.
    if (rc == FFX_API_RETURN_OK && ctx.fgEnabled && ctx.fgContext && ctx.fgWrappedSwapchain != VK_NULL_HANDLE) {
        ffxDispatchDescFrameGenerationPrepare prep{};
        prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        prep.frameID = ctx.frameID;
        prep.flags = 0;
        prep.commandList = reinterpret_cast<void*>(cmd);
        prep.renderSize = dp.renderSize;
        prep.jitterOffset = dp.jitterOffset;
        prep.motionVectorScale = dp.motionVectorScale;
        prep.frameTimeDelta = dp.frameTimeDelta;
        prep.unused_reset = false;
        prep.cameraNear = dp.cameraNear;
        prep.cameraFar = dp.cameraFar;
        prep.cameraFovAngleVertical = dp.cameraFovAngleVertical;
        prep.viewSpaceToMetersFactor = dp.viewSpaceToMetersFactor;
        prep.depth = dp.depth;
        prep.motionVectors = dp.motionVectors;
        ctx.fgPreparedThisFrame = (ffxDispatchSEH(ctx.ffxApi, &ctx.fgContext, &prep.header) == FFX_API_RETURN_OK);
    }

    for (auto it = restore.rbegin(); it != restore.rend(); ++it)
        recordBarrier(ctx, cmd, { it->image, it->oldLayout, it->newLayout, it->aspect });

    static ffxReturnCode_t s_logged = ~0u;
    if (rc != s_logged) {
        s_logged = rc;
        SL_LOG_INFO("sl.fsr: ffxDispatch(upscale) rc=0x%08X render=%ux%u output=%ux%u", (uint32_t)rc, renderW, renderH, outputW, outputH);
    }
    if (rc != FFX_API_RETURN_OK)
        return Result::eErrorComputeFailed;
    return Result::eOk;
}

sl::Result slFSRSetOptions(const sl::ViewportHandle& viewport, const sl::FSROptions& options)
{
    auto& ctx = (*fsr::getContext());
    std::lock_guard<std::mutex> lock(ctx.optionsMutex);
    ctx.upscaleOptions[(uint32_t)viewport] = options;
    return Result::eOk;
}

sl::Result slFSRGetState(const sl::ViewportHandle& /*viewport*/, sl::FSRState& state)
{
    state.estimatedVRAMUsageInBytes = 0;
    return Result::eOk;
}

sl::Result slFSRGetOptimalSettings(const sl::FSROptions& options, sl::FSROptimalSettings& settings)
{
    float ratio = 0.667f;
    switch (options.mode)
    {
        case FSRMode::eMaxPerformance:   ratio = 0.5f;   break;
        case FSRMode::eBalanced:         ratio = 0.59f;  break;
        case FSRMode::eMaxQuality:       ratio = 0.667f; break;
        case FSRMode::eUltraPerformance: ratio = 0.333f; break;
        case FSRMode::eNativeAA:         ratio = 1.0f;   break;
        default: break;
    }
    settings.optimalRenderWidth = (uint32_t)(options.outputWidth * ratio);
    settings.optimalRenderHeight = (uint32_t)(options.outputHeight * ratio);
    settings.optimalSharpness = options.sharpness;
    return Result::eOk;
}

sl::Result slFSRFrameGenerationSetOptions(const sl::ViewportHandle& viewport, const sl::FSRFrameGenOptions& options)
{
    auto& ctx = (*fsr::getContext());
    std::lock_guard<std::mutex> lock(ctx.optionsMutex);
    ctx.fgOptions[(uint32_t)viewport] = options;
    // Drives the CreateSwapchainKHR hook: the host enables/disables FSR FG here and then triggers a
    // DXVK swapchain recreate so the hook (re)wraps or releases the FFX FG swapchain. We do NOT create
    // or destroy the FFX swapchain here (no VkSwapchainCreateInfoKHR yet) — only latch the request.
    const bool want = options.enabled == Boolean::eTrue;
    if (want != ctx.fgEnabled) {
        ctx.fgEnabled = want;
        ctx.fgWrapFailed = false;       // fresh toggle: allow the present hook to (re)trigger the recreate
        ctx.fgRecreateInFlight = false;
    }
    ctx.fgColorHDR = options.colorBuffersHDR == Boolean::eTrue;
    return Result::eOk;
}

sl::Result slFSRGetFrameGenState(const sl::ViewportHandle& /*viewport*/, sl::FSRFrameGenState& state)
{
    auto& ctx = (*fsr::getContext());
    state.status = 0;
    // 2 == frame generation is actually doubling (FFX swapchain live + an FG-prepare ran).
    state.numFramesActuallyPresented = (ctx.fgEnabled && ctx.fgWrappedSwapchain != VK_NULL_HANDLE) ? 2 : 1;
    state.estimatedVRAMUsageInBytes = 0;
    return Result::eOk;
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
    if (ctx.physicalDevice == VK_NULL_HANDLE) {
        chi::PhysicalDevice physical{};
        if (ctx.compute) { ctx.compute->getPhysicalDevice(physical); ctx.physicalDevice = (VkPhysicalDevice)physical; }
    }
    // Only proxy when the host has FSR FG enabled (it triggers a swapchain recreate on toggle so this
    // hook re-runs). Otherwise let the interposer create the normal swapchain.
    if (!ctx.fgEnabled || ctx.platform != RenderAPI::eVulkan) {
        Skip = false;
        return VK_SUCCESS;
    }
    if (!ctx.ffxModule && !loadFFX(ctx)) {
        Skip = false;
        return VK_SUCCESS;
    }
    ctx.fgRecreateInFlight = false;  // the recreate we asked for has arrived
    if (createFgSwapchain(ctx, Device, CreateInfo, Allocator, Swapchain)) {
        Skip = true;  // *Swapchain now holds FFX's wrapped handle
        ctx.fgWrapFailed = false;
    } else {
        SL_LOG_WARN("sl.fsr: FG swapchain proxy failed; falling back to a normal swapchain");
        Skip = false;
        ctx.fgWrapFailed = true;  // suppress retrigger until the host toggles FG again
    }
    return VK_SUCCESS;
}

void slHookVkDestroySwapchainKHR(VkDevice /*Device*/, VkSwapchainKHR Swapchain, const VkAllocationCallbacks* /*Allocator*/, bool& Skip)
{
    auto& ctx = (*fsr::getContext());
    if (Swapchain != VK_NULL_HANDLE && Swapchain == ctx.fgWrappedSwapchain) {
        destroyFgSwapchain(ctx);  // FFX's DestroyContext destroys the real VkSwapchainKHR it owns
        ctx.fgRecreateInFlight = false;
        Skip = true;
    } else {
        Skip = false;
    }
}

VkResult slHookVkGetSwapchainImagesKHR(VkDevice Device, VkSwapchainKHR Swapchain, uint32_t* Count, VkImage* Images, bool& Skip)
{
    auto& ctx = (*fsr::getContext());
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

    // Capture DXVK's real present queue once for FFX's gameQueue (needed before we ever wrap).
    if (ctx.gameQueue == VK_NULL_HANDLE)
        ctx.gameQueue = Queue;

    // --- Self-triggered swapchain recreate (the ONLY recreate path that works under full interposition;
    // DxvkWsiHook is bypassed). When the desired wrap state (fgEnabled) differs from the actual state
    // (is this swapchain wrapped?), present this frame normally THEN return VK_SUBOPTIMAL_KHR: DXVK
    // recreates its swapchain, which re-enters our CreateSwapchainKHR hook to (un)wrap. Presenting first
    // avoids the dangling-fence hang of a no-present OUT_OF_DATE. fgRecreateInFlight de-bounces it to one
    // trigger per recreate; fgWrapFailed suppresses an infinite loop if the wrap can't be built.
    const bool wantWrap = ctx.fgEnabled;
    const bool inGameplay = ctx.fgUpscaleSinceLastPresent;
    ctx.fgUpscaleSinceLastPresent = false;  // consume; re-set by the next upscale
    const bool needRecreate = !ctx.fgRecreateInFlight && !ctx.fgWrapFailed &&
        ((wantWrap && ctx.fgWrappedSwapchain == VK_NULL_HANDLE && inGameplay) ||  // start wrapping (gameplay only)
         (!wantWrap && wrapped));                                                 // stop wrapping (safe anytime)
    if (needRecreate) {
        // Present the frame on its real path: FFX's present if this is the wrapped swapchain (FG turning
        // off), else the real ICD present (FG turning on, still a normal swapchain).
        VkResult pr;
        if (wrapped && ctx.fgSwapchainFns.pOutQueuePresentKHR)
            pr = ctx.fgSwapchainFns.pOutQueuePresentKHR(Queue, PresentInfo);
        else if (ctx.realQueuePresentKHR)
            pr = ctx.realQueuePresentKHR(Queue, PresentInfo);
        else
            pr = VK_SUCCESS;
        ctx.fgRecreateInFlight = true;
        Skip = true;
        SL_LOG_INFO("sl.fsr: requesting swapchain recreate to %s FG (present rc=%d)", wantWrap ? "ENABLE" : "disable", (int)pr);
        return pr < 0 ? pr : VK_SUBOPTIMAL_KHR;  // SUBOPTIMAL -> DXVK recreates -> CreateSwapchainKHR hook
    }

    if (!wrapped) {
        Skip = false;
        return VK_SUCCESS;
    }

    // Enable interpolation only on frames an FG-prepare actually ran for (else FFX reads empty inputs
    // -> GPU device-lost). Configure per-present, matching the canonical FFX sample.
    if (ctx.fgContext) {
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.swapChain = reinterpret_cast<void*>(ctx.fgWrappedSwapchain);
        cfg.frameGenerationCallback = &FfxFrameGenDispatchCallback;
        cfg.frameGenerationCallbackUserContext = &ctx;
        cfg.frameGenerationEnabled = ctx.fgPreparedThisFrame;
        cfg.allowAsyncWorkloads = false;
        cfg.generationRect = { 0, 0, (int32_t)ctx.fgDisplayW, (int32_t)ctx.fgDisplayH };
        cfg.frameID = ctx.frameID;
        ffxConfigureSEH(ctx.ffxApi, &ctx.fgContext, &cfg.header);
    }

    VkResult r = ctx.fgSwapchainFns.pOutQueuePresentKHR ?
        ctx.fgSwapchainFns.pOutQueuePresentKHR(Queue, PresentInfo) : VK_ERROR_INITIALIZATION_FAILED;

    ++ctx.frameID;
    ctx.fgPreparedThisFrame = false;
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
        SL_LOG_ERROR("sl.fsr: failed to obtain kPFunRegisterEvaluateCallbacks");
        return false;
    }
    ctx.registerEvaluateCallbacks(kFeatureFSR, fsrBeginEvaluation, fsrEndEvaluation);

    json& config = *(json*)api::getContext()->loaderConfig;
    uint32_t deviceType{};
    int appId{};
    config.at("appId").get_to(appId);
    config.at("deviceType").get_to(deviceType);
    ctx.platform = (RenderAPI)deviceType;

    if (!param::getPointerParam(parameters, sl::param::common::kComputeAPI, &ctx.compute))
    {
        SL_LOG_ERROR("sl.fsr: failed to obtain compute interface");
        return false;
    }

    // Resolve the REAL (un-interposed) device functions for the FFX backend + our barriers, from the
    // genuine vulkan-1.dll (the interposer self-loads it). Done here (no device work) — safe at startup.
    if (HMODULE vk = GetModuleHandleW(L"vulkan-1.dll")) {
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

    SL_LOG_INFO("sl.fsr plugin started");
    return true;
}

void slOnPluginShutdown()
{
    auto& ctx = (*fsr::getContext());
    destroyFgSwapchain(ctx);
    if (ctx.fgContext && ctx.ffxApi.DestroyContext)
        ctx.ffxApi.DestroyContext(&ctx.fgContext, nullptr);
    ctx.fgContext = nullptr;
    if (ctx.upscaleContext && ctx.ffxApi.DestroyContext)
        ctx.ffxApi.DestroyContext(&ctx.upscaleContext, nullptr);
    ctx.upscaleContext = nullptr;
    if (ctx.ffxModule)
        FreeLibrary(ctx.ffxModule);
    ctx.ffxModule = nullptr;
    ctx.registerEvaluateCallbacks(kFeatureFSR, nullptr, nullptr);
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
    // Request 2 EXTRA graphics-family queues from the interposer's vkCreateDevice (wrapper.cpp:357-360
    // sums this over loaded plugins' configs). FFX's FrameInterpolationSwapChain needs distinct
    // present + image-acquire queues, exclusive of DXVK's own. Set AFTER updateCommonEmbeddedJSONConfig
    // so it isn't overwritten; the plugin reads the injected family/start-index back from kVulkanTable.
    config["external"]["vk"]["device"]["queues"]["graphics"]["count"] = 2;
}

SL_EXPORT void* slGetPluginFunction(const char* functionName)
{
    SL_EXPORT_FUNCTION(slOnPluginLoad);
    SL_EXPORT_FUNCTION(slOnPluginShutdown);
    SL_EXPORT_FUNCTION(slOnPluginStartup);

    SL_EXPORT_FUNCTION(slFSRSetOptions);
    SL_EXPORT_FUNCTION(slFSRGetState);
    SL_EXPORT_FUNCTION(slFSRGetOptimalSettings);
    SL_EXPORT_FUNCTION(slFSRFrameGenerationSetOptions);
    SL_EXPORT_FUNCTION(slFSRGetFrameGenState);

    // Vulkan WSI before-hooks (declared in fsr.json) — the FG swapchain proxy.
    SL_EXPORT_FUNCTION(slHookVkCreateSwapchainKHR);
    SL_EXPORT_FUNCTION(slHookVkDestroySwapchainKHR);
    SL_EXPORT_FUNCTION(slHookVkGetSwapchainImagesKHR);
    SL_EXPORT_FUNCTION(slHookVkAcquireNextImageKHR);
    SL_EXPORT_FUNCTION(slHookVkQueuePresentKHR);

    return nullptr;
}

}
