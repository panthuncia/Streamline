/*
* Community Shaders sl.fsr plugin — AMD FidelityFX FSR3 upscaling (+ frame generation later).
*
* Drives FFX-API (amd_fidelityfx_vk.dll) on DXVK's VkDevice through Streamline's evaluate mechanism.
* The host tags ScalingInputColor / ScalingOutputColor / Depth / MotionVectors + sets common Constants
* (jitter, camera) and calls slEvaluateFeature(kFeatureFSR); fsrEndEvaluation records ffxDispatch(upscale)
* onto the SL-provided command buffer. Frame generation (present-hook insertion) is added in a later phase.
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

// Embedded SPIR-V for the HUDLessColor format bridge (hudless_copy.comp -> SPIR-V via glslangValidator).
#include "source/plugins/sl.fsr/hudless_copy_spv.h"

using json = nlohmann::json;

namespace sl
{
namespace fsr
{
// HUDLessColor format bridge: a self-contained compute pass (descriptor set + pipeline + an
// R10G10B10A2 target image) that copies the host's RGBA16F hudless into the backbuffer format so its
// FFX precision group matches and FFX accepts it as HUDLessColor under HDR. All VK objects are
// device-lifetime; the target image is recreated on display-size change. Function pointers are resolved
// from the GENUINE driver (ctx.realDeviceProcAddr), never through the FFX procaddr shim.
struct HudlessConvert
{
    bool failed = false;

    PFN_vkCreateImage createImage = nullptr;
    PFN_vkDestroyImage destroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements getImageMemReq = nullptr;
    PFN_vkAllocateMemory allocateMemory = nullptr;
    PFN_vkFreeMemory freeMemory = nullptr;
    PFN_vkBindImageMemory bindImageMemory = nullptr;
    PFN_vkCreateImageView createImageView = nullptr;
    PFN_vkDestroyImageView destroyImageView = nullptr;
    PFN_vkCreateSampler createSampler = nullptr;
    PFN_vkDestroySampler destroySampler = nullptr;
    PFN_vkCreateShaderModule createShaderModule = nullptr;
    PFN_vkDestroyShaderModule destroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout createDescSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroyDescSetLayout = nullptr;
    PFN_vkCreatePipelineLayout createPipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout destroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines createComputePipelines = nullptr;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool createDescPool = nullptr;
    PFN_vkDestroyDescriptorPool destroyDescPool = nullptr;
    PFN_vkAllocateDescriptorSets allocDescSets = nullptr;
    PFN_vkUpdateDescriptorSets updateDescSets = nullptr;
    PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmdBindDescSets = nullptr;
    PFN_vkCmdDispatch cmdDispatch = nullptr;

    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    VkImage dstImage = VK_NULL_HANDLE;
    VkDeviceMemory dstMemory = VK_NULL_HANDLE;
    VkImageView dstView = VK_NULL_HANDLE;
    uint32_t dstW = 0, dstH = 0;
    VkFormat dstFormat = VK_FORMAT_UNDEFINED;

    VkImage srcImage = VK_NULL_HANDLE;
    VkImageView srcView = VK_NULL_HANDLE;
    VkFormat srcFormat = VK_FORMAT_UNDEFINED;
    bool descDirty = true;
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
    ffxContext upscaleContext = nullptr;
    uint32_t ctxRenderW = 0, ctxRenderH = 0, ctxDisplayW = 0, ctxDisplayH = 0;
    bool dispatchFaulted = false;

    // Resolved device functions for the FFX backend + our barriers.
    PFN_vkGetDeviceProcAddr realDeviceProcAddr = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;

    std::mutex optionsMutex;
    std::map<uint32_t, FSROptions> upscaleOptions;

    // --- Frame generation (FFX FrameInterpolationSwapChainVK proxy) -----------------------------
    // DXVK's swapchain is ALWAYS replaced with an FFX FrameInterpolationSwapChain in the
    // eVulkan_CreateSwapchainKHR before-hook, independent of whether FG is enabled. With FG disabled the
    // FFX swapchain is a 1:1 passthrough (frameGenerationEnabled=false), so toggling FG is just an
    // ffxConfigure in the present hook — never a swapchain recreate (which was the source of every FG
    // toggle freeze/deadlock). The other WSI before-hooks (GetSwapchainImages/AcquireNextImage/Present/
    // DestroySwapchain) route the wrapped handle to FFX's replacement functions. Per-frame depth/MV/camera
    // is recorded by an FG-prepare dispatch appended to the upscale (fsrEndEvaluation).
    bool fgEnabled = false;                  // host requested FSR FG; gates frameGenerationEnabled only
    bool fgPreparedThisFrame = false;        // an FG-prepare ran this frame -> safe to interpolate
    bool fgColorHDR = false;
    uint32_t fgDebugFlags = 0;               // FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_* from the host's debug toggles
    bool fgOnlyPresentGenerated = false;     // present only generated frames (host fgShowOnlyGenerated)
    bool fgGameplayReached = false;          // an upscale has run -> frames are flowing -> safe to install
                                             // the FFX FG swapchain. Building it at cold init / a no-frame
                                             // loading screen crashes its setup, so the FIRST wrap is deferred
                                             // to here. Latched on; never cleared. NOTE: the swapchain does NOT
                                             // stay wrapped for the session — the present hook below
                                             // bidirectionally recreates via VK_SUBOPTIMAL on `wrapped !=
                                             // ctx.fgEnabled`, so disabling FSR FG drops the FFX wrapper back to
                                             // a plain swapchain (no passthrough overhead when FG is off).
    bool fgBootstrapInFlight = false;        // de-bounce the single present->SUBOPTIMAL recreate that
                                             // installs the FFX swapchain the first time gameplay is reached.
    bool fgWrapFailed = false;               // createFgSwapchain failed -> stop retriggering the bootstrap
                                             // (else present->recreate->fail loops every frame).
    uint64_t frameID = 0;
    int64_t fgLastPrepareQpc = 0;            // QPC of the last FG-prepare, for real frameTimeDelta
    int64_t upscaleLastQpc = 0;              // QPC of the last upscale dispatch, for real frameTimeDelta
    PFN_vkQueuePresentKHR realQueuePresentKHR = nullptr;
    ffxContext fgContext = nullptr;          // interpolation context (ffxCreateContextDescFrameGeneration)
    ffxContext fgSwapchainContext = nullptr; // swapchain-replacement context (FGSWAPCHAIN_VK)
    VkSwapchainKHR fgWrappedSwapchain = VK_NULL_HANDLE;
    ffxQueryDescSwapchainReplacementFunctionsVK fgSwapchainFns{};
    uint32_t fgDisplayW = 0, fgDisplayH = 0, fgRenderW = 0, fgRenderH = 0, fgBackBufferFormat = 0;
    bool fgContextHDR = false;  // the HDR flag the current fgContext was actually built with (vs fgColorHDR desired)

    // HUDLessColor: the scene WITHOUT UI, at display (back-buffer) res, tagged by the host on the FG viewport
    // every gameplay frame. FFX's FrameInterpolationSwapChain uses it to extract the UI (backbuffer - hudless)
    // so interpolated frames don't ghost the HUD. fsrEndEvaluation captures the wrapped resource declared in
    // its CURRENT DXVK layout (NO transition here — it is consumed on FFX's own present cmd, not our evaluate
    // cmd; FFX transitions from/back to it per its contract, keeping DXVK's layout tracking consistent) and the
    // present hook feeds it to ffxConfigure. fgHudlessVkFormat (derived from the tagged resource) is baked into
    // the FG context via the optional hudless create-struct when it differs from the backbuffer (e.g. an
    // RGBA16F scene buffer vs an RGBA8 backbuffer) — without it FFX subtracts the hudless using the wrong format.
    FfxApiResource fgHudlessResource{};
    bool fgHaveHudless = false;
    bool fgHudlessCompatible = false;   // hudless format is in the SAME FFX precision group as the backbuffer
    uint32_t fgHudlessVkFormat = 0;     // VkFormat of the host's hudless texture; 0 => assume == backbuffer
    uint32_t fgCtxHudlessFormat = 0;    // VkFormat the current fgContext was actually built with (recreate on change)
    VkFormat fgBackBufferVkFormat = VK_FORMAT_UNDEFINED;  // the swapchain's VkFormat (R10G10B10A2 in HDR); the
                                        // format the hudless bridge targets so it shares the backbuffer group.

    // HUDLessColor format bridge (RGBA16F hudless -> backbuffer-format R10G10B10A2 so FFX accepts it in HDR).
    HudlessConvert hc;
    PFN_vkGetPhysicalDeviceMemoryProperties pfnGetMemProps = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties pfnGetFormatProps = nullptr;
    VkPhysicalDeviceMemoryProperties memProps{};
    bool memPropsValid = false;

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
// Skyrim world units -> meters, for FFX's depth-based reactive/reprojection heuristics (~70 units/m).
// Shared by the upscale and FG-prepare dispatches so both describe the scene at the same scale.
constexpr float kSkyrimViewSpaceToMeters = 0.01428222656f;

// --- DXVK ⇄ unmodified prebuilt-FFX VK backend shim (ported from CS FidelityFX.cpp) -------------
// The stock amd_fidelityfx_vk.dll calls two functions that resolve to NULL on DXVK without null-checks:
//   vkGetBufferMemoryRequirements2KHR (DXVK promoted it to core, dropped the KHR alias) and
//   vkCmdWriteBufferMarker(2)AMD (DXVK enumerates VK_AMD_buffer_marker but leaves the entries NULL).
// Route the first to the core entry; stub the markers to a no-op so the UNMODIFIED DLL runs on DXVK.
PFN_vkGetDeviceProcAddr g_realDeviceProcAddr = nullptr;

// Set while FFX creates its OWN (inner) swapchain inside createFgSwapchain. FFX's inner vkCreateSwapchainKHR
// re-enters the SL interposer, where our CreateSwapchainKHR hook sees this flag and creates the swapchain
// on the GENUINE driver with Skip=true — so the interposer's other plugin hooks (sl.dlss_g, which would
// otherwise also wrap FFX's swapchain → two FG present owners → device-lost) never touch it. This is what
// lets FSR FG and DLSS-G stay loaded together and be switched in-game.
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

// Route FFX-API internal messages (notably context-creation failures) into the SL log. FFX's providers
// are otherwise silent — ffxCreateContext just returns a code — so without this we cannot see WHY a FG
// context fails. Registered globally via ffxConfigure(nullptr, GLOBALDEBUG1) before the FG create.
static void ffxApiLogMessage(uint32_t type, const wchar_t* message)
{
    if (!message)
        return;
    char buf[1024] = {};
    size_t conv = 0;
    wcstombs_s(&conv, buf, sizeof(buf), message, _TRUNCATE);
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        SL_LOG_ERROR("sl.fsr: [FFX] %s", buf);
    } else {
        SL_LOG_WARN("sl.fsr: [FFX] %s", buf);
    }
}

// Inverse of ffxStateToLayout: map a resource's CURRENT VkImageLayout back to an FFX state enum. Used for
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
    // Skyrim's depth is standard [0=near .. 1=far], not reversed-Z, so no FFX_UPSCALE_ENABLE_DEPTH_INVERTED.
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

// Serializes ALL ffxDispatch calls on the FG context. The FG-prepare dispatches on the game's
// render thread (inside slEvaluateFeature) while the interpolation dispatch runs on DXVK's submit
// thread (present -> FFX queuePresent -> FfxFrameGenDispatchCallback) — TWO threads driving ONE
// FFX context. FFX has no internal lock on this edge and mutates non-atomic per-context state on
// every dispatch (dynamic-resource/descriptor index allocators, barrier scratch arrays, the
// backend GPU-job list), so a concurrent prepare + generate corrupts descriptors / the command
// stream -> sporadic VK_ERROR_DEVICE_LOST minutes into steady frame generation. AMD's samples
// never hit this because they present on the game thread (prepare and callback are naturally
// same-thread); DXVK presenting from its submit thread created the concurrency. Leaf mutex:
// nothing else is locked while held.
std::mutex g_fgDispatchMutex;

// FFX's swapchain calls this per generated frame to run the interpolation dispatch on our FG context.
ffxReturnCode_t FfxFrameGenDispatchCallback(ffxDispatchDescFrameGeneration* params, void* pUserCtx)
{
    auto* ctx = reinterpret_cast<fsr::FSRContext*>(pUserCtx);
    if (!ctx || !ctx->fgContext)
        return FFX_API_RETURN_ERROR;
    std::lock_guard<std::mutex> lock(g_fgDispatchMutex);
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

// Mirror ffx_frameinterpolation.cpp's GetFormatPrecisionGroup over FfxApiSurfaceFormat values. FFX's
// ffxFrameInterpolationContextCreate rejects (FFX_ERROR_INVALID_ARGUMENT) a HUDLessColor whose format is in a
// different precision group than the backbuffer, so we use it only when the groups match. -1 = unaccepted.
static int ffxApiFormatPrecisionGroup(uint32_t fmt)
{
    switch (fmt) {
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT:
    case FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT:        return 0;
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS:
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT:     return 1;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM:
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS:
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM:         return 2;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM:         return 3;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB:
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB:          return 4;
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT:        return 5;
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS:
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM:      return 6;
    case FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP:     return 7;
    default:                                            return -1;
    }
}

// Create the interpolation context (separate from the swapchain context; the swapchain drives it via
// the dispatch callback). Sized to display+maxRender; recreated on size change.
bool ensureFgContext(fsr::FSRContext& ctx, uint32_t displayW, uint32_t displayH, uint32_t renderW, uint32_t renderH, uint32_t backBufferFormat)
{
    if (ctx.fgContext && ctx.fgDisplayW == displayW && ctx.fgDisplayH == displayH &&
        ctx.fgRenderW >= renderW && ctx.fgRenderH >= renderH && ctx.fgBackBufferFormat == backBufferFormat &&
        ctx.fgContextHDR == ctx.fgColorHDR &&            // HDR flag is baked into fgDesc.flags at create
        ctx.fgCtxHudlessFormat == ctx.fgHudlessVkFormat)  // hudless format is baked via the hudless create-struct
        return true;
    if (ctx.fgContext) {
        // Wait out any in-flight prepare/interpolation dispatch before destroying the
        // context it runs on (render thread vs present thread; see g_fgDispatchMutex).
        std::lock_guard<std::mutex> lock(g_fgDispatchMutex);
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
    // Async-compute workloads are disabled (frame generation runs on the game queue).
    // Skyrim's depth is standard [0=near .. 1=far], not reversed-Z, so no FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED.
    fgDesc.displaySize = { displayW, displayH };
    fgDesc.maxRenderSize = { renderW, renderH };
    fgDesc.backBufferFormat = backBufferFormat;

    // The host's HUDLessColor (the upscaled scene without UI) lets FFX extract the HUD so it isn't ghosted by
    // interpolation. BUT ffxFrameInterpolationContextCreate REQUIRES the hudless and backbuffer formats to be in
    // the same "precision group" (ffx_frameinterpolation.cpp GetFormatPrecisionGroup) or it returns
    // FFX_ERROR_INVALID_ARGUMENT — which is exactly the FSR-FG failure: the hudless is the RGBA16F scene buffer
    // (group 1) while the HDR backbuffer is R10G10B10A2 (group 6). So only chain + use the hudless when its
    // precision group matches the backbuffer's; otherwise run with an empty HUDLessColor (as the dev branch did).
    ffxCreateContextDescFrameGenerationHudless hudlessDesc{};
    const uint32_t hudlessFormat = ctx.fgHudlessVkFormat ? ffxApiGetSurfaceFormatVK((VkFormat)ctx.fgHudlessVkFormat) : 0;
    const int hudlessGroup = ffxApiFormatPrecisionGroup(hudlessFormat);
    const int backBufferGroup = ffxApiFormatPrecisionGroup(backBufferFormat);
    ctx.fgHudlessCompatible = hudlessFormat && hudlessGroup >= 0 && hudlessGroup == backBufferGroup;
    if (ctx.fgHudlessCompatible && hudlessFormat != backBufferFormat) {
        hudlessDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;
        hudlessDesc.header.pNext = &backendDesc.header;
        hudlessDesc.hudlessBackBufferFormat = hudlessFormat;
        fgDesc.header.pNext = &hudlessDesc.header;  // chain: fgDesc -> hudless -> backend
        SL_LOG_INFO("sl.fsr: FG context using hudless format %u (backbuffer %u)", hudlessFormat, backBufferFormat);
    } else if (hudlessFormat && !ctx.fgHudlessCompatible) {
        SL_LOG_INFO("sl.fsr: hudless format %u (group %d) incompatible with backbuffer %u (group %d) — empty HUDLessColor",
            hudlessFormat, hudlessGroup, backBufferFormat, backBufferGroup);
    }

    // Register FFX's global debug message callback so a FG-context-create failure logs WHY (optical-flow /
    // frame-interpolation pipeline or resource that failed), not just the opaque return code.
    ffxConfigureDescGlobalDebug1 ffxDbg{};
    ffxDbg.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1;
    ffxDbg.fpMessage = &ffxApiLogMessage;
    ffxDbg.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS;
    if (ctx.ffxApi.Configure)
        ctx.ffxApi.Configure(nullptr, &ffxDbg.header);

    ffxReturnCode_t rc = ffxCreateContextSEH(ctx.ffxApi, &ctx.fgContext, &fgDesc.header);
    if (rc != FFX_API_RETURN_OK) {
        SL_LOG_ERROR("sl.fsr: ffxCreateContext(FG) failed 0x%08X", (uint32_t)rc);
        ctx.fgContext = nullptr;
        return false;
    }
    ctx.fgDisplayW = displayW; ctx.fgDisplayH = displayH; ctx.fgRenderW = renderW; ctx.fgRenderH = renderH;
    ctx.fgBackBufferFormat = backBufferFormat;
    ctx.fgContextHDR = ctx.fgColorHDR;
    ctx.fgCtxHudlessFormat = ctx.fgHudlessVkFormat;
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
    ctx.fgBackBufferVkFormat = pCreateInfo->imageFormat;  // target format for the hudless bridge
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
    ffxReturnCode_t rc = ffxCreateContextSEH(ctx.ffxApi, &ctx.fgSwapchainContext, &desc.header);
    g_creatingFgSwapchain.store(false, std::memory_order_release);
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
        ffxReturnCode_t prc = ffxConfigureSEH(ctx.ffxApi, &ctx.fgSwapchainContext, &kv.header);
        SL_LOG_INFO("sl.fsr: FG frame-pacing tuning (rc=0x%08X safety=%.3fms variance=%.3f)",
            (uint32_t)prc, pacing.safetyMarginInMs, pacing.varianceFactor);
    }

    SL_LOG_INFO("sl.fsr: FFX FG swapchain created (%ux%u, wrapped 0x%llX)", displayW, displayH, (unsigned long long)ctx.fgWrappedSwapchain);
    return true;
}

// ---- HUDLessColor format bridge (RGBA16F hudless -> backbuffer-format R10G10B10A2 via compute) -------

// Resolve the bridge's device functions from the genuine driver. Latches failure so a missing entry
// disables the bridge (HUD extraction off) rather than retrying every frame.
bool hcEnsureResolved(fsr::FSRContext& ctx)
{
    auto& hc = ctx.hc;
    if (hc.failed) return false;
    if (hc.cmdDispatch) return true;
    if (!ctx.realDeviceProcAddr || ctx.device == VK_NULL_HANDLE) return false;
    auto R = [&](const char* n) { return ctx.realDeviceProcAddr(ctx.device, n); };
    hc.createImage            = (PFN_vkCreateImage)R("vkCreateImage");
    hc.destroyImage           = (PFN_vkDestroyImage)R("vkDestroyImage");
    hc.getImageMemReq         = (PFN_vkGetImageMemoryRequirements)R("vkGetImageMemoryRequirements");
    hc.allocateMemory         = (PFN_vkAllocateMemory)R("vkAllocateMemory");
    hc.freeMemory             = (PFN_vkFreeMemory)R("vkFreeMemory");
    hc.bindImageMemory        = (PFN_vkBindImageMemory)R("vkBindImageMemory");
    hc.createImageView        = (PFN_vkCreateImageView)R("vkCreateImageView");
    hc.destroyImageView       = (PFN_vkDestroyImageView)R("vkDestroyImageView");
    hc.createSampler          = (PFN_vkCreateSampler)R("vkCreateSampler");
    hc.destroySampler         = (PFN_vkDestroySampler)R("vkDestroySampler");
    hc.createShaderModule     = (PFN_vkCreateShaderModule)R("vkCreateShaderModule");
    hc.destroyShaderModule    = (PFN_vkDestroyShaderModule)R("vkDestroyShaderModule");
    hc.createDescSetLayout    = (PFN_vkCreateDescriptorSetLayout)R("vkCreateDescriptorSetLayout");
    hc.destroyDescSetLayout   = (PFN_vkDestroyDescriptorSetLayout)R("vkDestroyDescriptorSetLayout");
    hc.createPipelineLayout   = (PFN_vkCreatePipelineLayout)R("vkCreatePipelineLayout");
    hc.destroyPipelineLayout  = (PFN_vkDestroyPipelineLayout)R("vkDestroyPipelineLayout");
    hc.createComputePipelines = (PFN_vkCreateComputePipelines)R("vkCreateComputePipelines");
    hc.destroyPipeline        = (PFN_vkDestroyPipeline)R("vkDestroyPipeline");
    hc.createDescPool         = (PFN_vkCreateDescriptorPool)R("vkCreateDescriptorPool");
    hc.destroyDescPool        = (PFN_vkDestroyDescriptorPool)R("vkDestroyDescriptorPool");
    hc.allocDescSets          = (PFN_vkAllocateDescriptorSets)R("vkAllocateDescriptorSets");
    hc.updateDescSets         = (PFN_vkUpdateDescriptorSets)R("vkUpdateDescriptorSets");
    hc.cmdBindPipeline        = (PFN_vkCmdBindPipeline)R("vkCmdBindPipeline");
    hc.cmdBindDescSets        = (PFN_vkCmdBindDescriptorSets)R("vkCmdBindDescriptorSets");
    hc.cmdDispatch            = (PFN_vkCmdDispatch)R("vkCmdDispatch");
    const bool ok = hc.createImage && hc.destroyImage && hc.getImageMemReq && hc.allocateMemory &&
        hc.freeMemory && hc.bindImageMemory && hc.createImageView && hc.destroyImageView && hc.createSampler &&
        hc.destroySampler && hc.createShaderModule && hc.destroyShaderModule && hc.createDescSetLayout &&
        hc.destroyDescSetLayout && hc.createPipelineLayout && hc.destroyPipelineLayout &&
        hc.createComputePipelines && hc.destroyPipeline && hc.createDescPool && hc.destroyDescPool &&
        hc.allocDescSets && hc.updateDescSets && hc.cmdBindPipeline && hc.cmdBindDescSets && hc.cmdDispatch;
    if (!ok) {
        SL_LOG_WARN("sl.fsr: hudless bridge — missing device function(s); HUD extraction disabled");
        hc.failed = true;
    }
    return ok;
}

// Build the compute pipeline + descriptor set once.
bool hcEnsurePipeline(fsr::FSRContext& ctx)
{
    auto& hc = ctx.hc;
    if (hc.pipeline != VK_NULL_HANDLE) return true;

    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sizeof(g_hudlessCopySpv);
    smci.pCode = g_hudlessCopySpv;
    if (hc.createShaderModule(ctx.device, &smci, nullptr, &hc.shader) != VK_SUCCESS) { hc.failed = true; return false; }

    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    hc.createSampler(ctx.device, &sci, nullptr, &hc.sampler);

    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo slci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    slci.bindingCount = 2; slci.pBindings = b;
    if (hc.createDescSetLayout(ctx.device, &slci, nullptr, &hc.setLayout) != VK_SUCCESS) { hc.failed = true; return false; }

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &hc.setLayout;
    if (hc.createPipelineLayout(ctx.device, &plci, nullptr, &hc.pipeLayout) != VK_SUCCESS) { hc.failed = true; return false; }

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = hc.shader;
    cpci.stage.pName = "main";
    cpci.layout = hc.pipeLayout;
    if (hc.createComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &hc.pipeline) != VK_SUCCESS) {
        hc.pipeline = VK_NULL_HANDLE; hc.failed = true; return false;
    }

    VkDescriptorPoolSize ps[2]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ps[0].descriptorCount = 1;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          ps[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
    if (hc.createDescPool(ctx.device, &dpci, nullptr, &hc.descPool) != VK_SUCCESS) { hc.failed = true; return false; }
    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = hc.descPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &hc.setLayout;
    if (hc.allocDescSets(ctx.device, &dsai, &hc.descSet) != VK_SUCCESS) { hc.failed = true; return false; }
    return true;
}

// (Re)create the R10G10B10A2 target image at display resolution. Requires STORAGE_IMAGE support for the
// backbuffer format (true on NVIDIA/DXVK with shaderStorageImageExtendedFormats); disables the bridge if not.
bool hcEnsureTarget(fsr::FSRContext& ctx, uint32_t w, uint32_t h, VkFormat fmt)
{
    auto& hc = ctx.hc;
    if (hc.dstImage != VK_NULL_HANDLE && hc.dstW == w && hc.dstH == h && hc.dstFormat == fmt)
        return true;
    if (hc.dstView)   { hc.destroyImageView(ctx.device, hc.dstView, nullptr);   hc.dstView = VK_NULL_HANDLE; }
    if (hc.dstImage)  { hc.destroyImage(ctx.device, hc.dstImage, nullptr);       hc.dstImage = VK_NULL_HANDLE; }
    if (hc.dstMemory) { hc.freeMemory(ctx.device, hc.dstMemory, nullptr);        hc.dstMemory = VK_NULL_HANDLE; }

    if (ctx.physicalDevice == VK_NULL_HANDLE && ctx.compute) {
        chi::PhysicalDevice physical{};
        ctx.compute->getPhysicalDevice(physical);
        ctx.physicalDevice = (VkPhysicalDevice)physical;
    }
    if (ctx.pfnGetFormatProps && ctx.physicalDevice != VK_NULL_HANDLE) {
        VkFormatProperties fp{};
        ctx.pfnGetFormatProps(ctx.physicalDevice, fmt, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
            SL_LOG_WARN("sl.fsr: hudless bridge — backbuffer format %d lacks STORAGE_IMAGE; HUD extraction disabled", (int)fmt);
            hc.failed = true; return false;
        }
    }
    if (!ctx.memPropsValid && ctx.pfnGetMemProps && ctx.physicalDevice != VK_NULL_HANDLE) {
        ctx.pfnGetMemProps(ctx.physicalDevice, &ctx.memProps);
        ctx.memPropsValid = true;
    }

    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = { w, h, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (hc.createImage(ctx.device, &ici, nullptr, &hc.dstImage) != VK_SUCCESS) { hc.dstImage = VK_NULL_HANDLE; hc.failed = true; return false; }

    VkMemoryRequirements mr{};
    hc.getImageMemReq(ctx.device, hc.dstImage, &mr);
    uint32_t typeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < ctx.memProps.memoryTypeCount; ++i)
        if ((mr.memoryTypeBits & (1u << i)) && (ctx.memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { typeIndex = i; break; }
    if (typeIndex == UINT32_MAX) { SL_LOG_WARN("sl.fsr: hudless bridge — no device-local memory type"); hc.failed = true; return false; }
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size; mai.memoryTypeIndex = typeIndex;
    if (hc.allocateMemory(ctx.device, &mai, nullptr, &hc.dstMemory) != VK_SUCCESS) { hc.dstMemory = VK_NULL_HANDLE; hc.failed = true; return false; }
    hc.bindImageMemory(ctx.device, hc.dstImage, hc.dstMemory, 0);

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = hc.dstImage; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (hc.createImageView(ctx.device, &vci, nullptr, &hc.dstView) != VK_SUCCESS) { hc.dstView = VK_NULL_HANDLE; hc.failed = true; return false; }

    hc.dstW = w; hc.dstH = h; hc.dstFormat = fmt; hc.descDirty = true;
    SL_LOG_INFO("sl.fsr: hudless bridge target %ux%u fmt=%d", w, h, (int)fmt);
    return true;
}

// Cache an image view of the host's hudless source; rebuild only when the source image/format changes.
bool hcEnsureSrcView(fsr::FSRContext& ctx, VkImage src, VkFormat fmt)
{
    auto& hc = ctx.hc;
    if (hc.srcView != VK_NULL_HANDLE && hc.srcImage == src && hc.srcFormat == fmt)
        return true;
    if (hc.srcView) { hc.destroyImageView(ctx.device, hc.srcView, nullptr); hc.srcView = VK_NULL_HANDLE; }
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = src; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = fmt;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (hc.createImageView(ctx.device, &vci, nullptr, &hc.srcView) != VK_SUCCESS) { hc.srcView = VK_NULL_HANDLE; return false; }
    hc.srcImage = src; hc.srcFormat = fmt; hc.descDirty = true;
    return true;
}

void hcUpdateDescriptors(fsr::FSRContext& ctx)
{
    auto& hc = ctx.hc;
    VkDescriptorImageInfo srcInfo{};
    srcInfo.sampler = hc.sampler;
    srcInfo.imageView = hc.srcView;
    srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo dstInfo{};
    dstInfo.imageView = hc.dstView;
    dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet w[2]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = hc.descSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &srcInfo;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = hc.descSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &dstInfo;
    hc.updateDescSets(ctx.device, 2, w, 0, nullptr);
    hc.descDirty = false;
}

// Record the hudless->backbuffer-format copy on the SL evaluate command buffer and point
// ctx.fgHudlessResource at the converted (FFX-compatible) image. Returns false on any setup failure.
bool convertHudless(fsr::FSRContext& ctx, VkCommandBuffer cmd, VkImage src, VkFormat srcFmt,
    VkImageLayout srcLayout, uint32_t w, uint32_t h)
{
    auto& hc = ctx.hc;
    if (ctx.fgBackBufferVkFormat == VK_FORMAT_UNDEFINED || !ctx.vkCmdPipelineBarrier)
        return false;
    if (!hcEnsureResolved(ctx) || !hcEnsurePipeline(ctx) ||
        !hcEnsureTarget(ctx, w, h, ctx.fgBackBufferVkFormat) || !hcEnsureSrcView(ctx, src, srcFmt))
        return false;
    if (hc.descDirty)
        hcUpdateDescriptors(ctx);

    const VkImageSubresourceRange color = { VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS };
    VkImageMemoryBarrier pre[2]{};
    pre[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pre[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    pre[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    pre[0].oldLayout = srcLayout;
    pre[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    pre[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre[0].image = src; pre[0].subresourceRange = color;
    pre[1] = pre[0];
    pre[1].srcAccessMask = 0;
    pre[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;   // we overwrite every texel; old contents discardable
    pre[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[1].image = hc.dstImage;
    ctx.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 2, pre);

    hc.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hc.pipeline);
    hc.cmdBindDescSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hc.pipeLayout, 0, 1, &hc.descSet, 0, nullptr);
    hc.cmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);

    VkImageMemoryBarrier post[2]{};
    post[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    post[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    post[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    post[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;   // FFX consumes it as COMMON (GENERAL)
    post[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; post[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post[0].image = hc.dstImage; post[0].subresourceRange = color;
    post[1] = post[0];
    post[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    post[1].dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
    post[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    post[1].newLayout = (srcLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_IMAGE_LAYOUT_GENERAL : srcLayout;
    post[1].image = src;
    ctx.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 0, nullptr, 0, nullptr, 2, post);

    VkImageCreateInfo info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = ctx.fgBackBufferVkFormat;
    info.extent = { w, h, 1 };
    info.mipLevels = 1; info.arrayLayers = 1; info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    FfxApiResourceDescription fdesc = ffxApiGetImageResourceDescriptionVK(hc.dstImage, info, FFX_API_RESOURCE_USAGE_READ_ONLY);
    ctx.fgHudlessResource = ffxApiGetResourceVK(reinterpret_cast<void*>(hc.dstImage), fdesc, FFX_API_RESOURCE_STATE_COMMON);
    return true;
}

// Destroy all bridge VK objects (device-lifetime; called at plugin shutdown).
void hcDestroy(fsr::FSRContext& ctx)
{
    auto& hc = ctx.hc;
    if (ctx.device == VK_NULL_HANDLE) { hc = {}; return; }
    if (hc.srcView)    hc.destroyImageView(ctx.device, hc.srcView, nullptr);
    if (hc.dstView)    hc.destroyImageView(ctx.device, hc.dstView, nullptr);
    if (hc.dstImage)   hc.destroyImage(ctx.device, hc.dstImage, nullptr);
    if (hc.dstMemory)  hc.freeMemory(ctx.device, hc.dstMemory, nullptr);
    if (hc.pipeline)   hc.destroyPipeline(ctx.device, hc.pipeline, nullptr);
    if (hc.pipeLayout) hc.destroyPipelineLayout(ctx.device, hc.pipeLayout, nullptr);
    if (hc.setLayout)  hc.destroyDescSetLayout(ctx.device, hc.setLayout, nullptr);
    if (hc.descPool)   hc.destroyDescPool(ctx.device, hc.descPool, nullptr);
    if (hc.sampler)    hc.destroySampler(ctx.device, hc.sampler, nullptr);
    if (hc.shader)     hc.destroyShaderModule(ctx.device, hc.shader, nullptr);
    hc = {};
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
    // Color is OPTIONAL: this single kFeatureFSR evaluate serves both FSR upscale (color present) and the
    // standalone FG-prepare the host drives on a dedicated viewport (depth+MV only, no color). Query color
    // with optional=true so the absence on the FG-prepare viewport doesn't spam tag-not-found errors every
    // frame. Depth+MV stay required (optional=false) — their absence is a real error.
    getTaggedResource(kBufferTypeScalingInputColor, colorIn, evd.frame, evd.id, true, inputs, numInputs);
    getTaggedResource(kBufferTypeScalingOutputColor, colorOut, evd.frame, evd.id, true, inputs, numInputs);
    getTaggedResource(kBufferTypeDepth, depth, evd.frame, evd.id, false, inputs, numInputs);
    getTaggedResource(kBufferTypeMotionVectors, mvec, evd.frame, evd.id, false, inputs, numInputs);
    // HUDLessColor is OPTIONAL and only present on the FG-prepare viewport (the host tags it for FFX UI
    // extraction). Query it optional so the upscale viewport's absence doesn't spam tag-not-found.
    CommonResource hudless{};
    getTaggedResource(kBufferTypeHUDLessColor, hudless, evd.frame, evd.id, true, inputs, numInputs);
    // This single kFeatureFSR evaluate serves two roles, distinguished by what the host tagged:
    //   * color + depth + MV  -> FSR UPSCALE (+ FG-prepare if FG is on and using FSR upscale)
    //   * depth + MV only      -> standalone FG-PREPARE, which the host drives EVERY frame so FSR frame
    //                             generation works under ANY upscaler (XeSS/DLSS/none), not just FSR.
    // Depth + MV are mandatory for both.
    if (!depth || !mvec)
        return Result::eErrorMissingInputParameter;
    const bool haveColor = (colorIn && colorOut);

    const auto& depthExt = depth.getExtent();
    const uint32_t renderW = haveColor ? colorIn.getExtent().width  : depthExt.width;
    const uint32_t renderH = haveColor ? colorIn.getExtent().height : depthExt.height;
    const uint32_t outputW = haveColor ? colorOut.getExtent().width  : renderW;
    const uint32_t outputH = haveColor ? colorOut.getExtent().height : renderH;
    if (!renderW || !renderH)
        return Result::eErrorMissingInputParameter;

    VkCommandBuffer cmd = (VkCommandBuffer)cmdList;
    std::vector<Barrier> restore;

    // Depth + MV feed both the upscale and the FG-prepare; wrap+transition once and share.
    FfxApiResource depthRes = wrapAndTransition(ctx, cmd, depth, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_DEPTHTARGET, true, restore);
    FfxApiResource mvecRes = wrapAndTransition(ctx, cmd, mvec, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY, false, restore);
    // The motion-vector resource MUST carry the SAME dimensions/subrect as depth — FFX reconstructs MV against
    // the depth grid, so a size mismatch (e.g. a full-res MV texture vs a render-res depth) misaligns the
    // dilation and leaves MV not covering the frame. Force MV's description to depth's so they're sampled
    // identically (renderSize below already defines the shared valid subrect).
    mvecRes.description.width = depthRes.description.width;
    mvecRes.description.height = depthRes.description.height;
    const FfxApiFloatCoords2D motionVectorScale = { (float)renderW, (float)renderH };
    // consts->jitterOffset is ALREADY negated by the host (matching SL/DLSS + FFX's expected -jitter);
    // pass it straight through — negating again was a double-negation that ghosts under motion.
    const FfxApiFloatCoords2D jitterOffset = { consts->jitterOffset.x, consts->jitterOffset.y };

    ffxReturnCode_t rc = FFX_API_RETURN_OK;

    if (haveColor) {
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

        ffxDispatchDescUpscale dp{};
        dp.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dp.commandList = reinterpret_cast<void*>(cmd);
        dp.color = wrapAndTransition(ctx, cmd, colorIn, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY, false, restore);
        dp.depth = depthRes;
        dp.motionVectors = mvecRes;
        dp.reactive = ffxApiGetResourceVK(nullptr, {}, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dp.transparencyAndComposition = ffxApiGetResourceVK(nullptr, {}, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dp.exposure = ffxApiGetResourceVK(nullptr, {}, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dp.output = wrapAndTransition(ctx, cmd, colorOut, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV, false, restore);
        dp.motionVectorScale = motionVectorScale;
        dp.renderSize = { renderW, renderH };
        dp.upscaleSize = { outputW, outputH };
        dp.jitterOffset = jitterOffset;
        // Real frame delta (QPC), not a hardcoded 60Hz value: FSR's reactive/exposure adaptation scales
        // with frameTimeDelta, so a fixed 16.6 mis-adapts at uncapped/high FPS.
        {
            LARGE_INTEGER qpcNow;
            QueryPerformanceCounter(&qpcNow);
            float deltaMs = 16.6f;
            if (ctx.upscaleLastQpc) {
                LARGE_INTEGER qpcFreq;
                QueryPerformanceFrequency(&qpcFreq);
                const double ms = double(qpcNow.QuadPart - ctx.upscaleLastQpc) * 1000.0 / double(qpcFreq.QuadPart);
                deltaMs = (float)std::min(std::max(ms, 1.0), 100.0);
            }
            ctx.upscaleLastQpc = qpcNow.QuadPart;
            dp.frameTimeDelta = deltaMs;
        }
        dp.cameraFar = consts->cameraFar;
        dp.cameraNear = consts->cameraNear;
        dp.enableSharpening = sharpness > 0.0f;
        dp.sharpness = sharpness;
        dp.cameraFovAngleVertical = consts->cameraFOV;
        dp.viewSpaceToMetersFactor = kSkyrimViewSpaceToMeters;
        // Honor the host's camera-cut/reset signal (teleport, fast-travel, menu->game) so FSR flushes its
        // temporal history for one frame instead of smearing the discontinuity.
        dp.reset = (consts->reset == Boolean::eTrue);
        dp.preExposure = 1.0f;
        dp.flags = 0;
        rc = ffxDispatchSEH(ctx.ffxApi, &ctx.upscaleContext, &dp.header);

        static ffxReturnCode_t s_logged = ~0u;
        if (rc != s_logged) {
            s_logged = rc;
            SL_LOG_INFO("sl.fsr: ffxDispatch(upscale) rc=0x%08X render=%ux%u output=%ux%u", (uint32_t)rc, renderW, renderH, outputW, outputH);
        }
    }

    // FG-prepare: runs whenever the host wants FG, regardless of which upscaler produced the frame. It
    // records this frame's dilated depth/MV + camera/render size for the present-time interpolation.
    // Setting fgGameplayReached lets the present hook bootstrap-install the FFX swapchain; the prepare
    // dispatch itself only runs once that swapchain (and its FG context) exists.
    if (ctx.fgEnabled) {
        ctx.fgGameplayReached = true;

        // Capture HUDLessColor for present-time UI extraction. NO transition here — FFX consumes it on its own
        // present command buffer, so we declare it in its CURRENT DXVK layout (FFX transitions from/back to it)
        // and stash the wrapped resource for the present hook. Also record its format so the FG context bakes
        // the matching hudless create-struct (see ensureFgContext) before the swapchain is first wrapped.
        // The FG context was built (createFgSwapchain) expecting hudless in the BACKBUFFER format (fgHudlessVkFormat
        // was 0 at create -> no hudless create-struct chained). So if the host's hudless is already that format, feed
        // it directly; otherwise (HDR: RGBA16F scene vs R10G10B10A2 backbuffer) bridge it through the compute copy so
        // its precision group matches and FFX accepts it for HUD extraction instead of dropping it.
        ctx.fgHaveHudless = false;
        ctx.fgHudlessCompatible = false;
        if (chi::Resource hres = hudless; hres && hres->native) {
            chi::ResourceDescription hdesc{};
            ctx.compute->getResourceDescription(hres, hdesc);
            VkImage himage = (VkImage)hres->native;
            const VkFormat hfmt = (VkFormat)hdesc.nativeFormat;
            if (ctx.fgBackBufferVkFormat != VK_FORMAT_UNDEFINED && hfmt == ctx.fgBackBufferVkFormat) {
                VkImageCreateInfo hinfo = imageInfoFromDesc(hdesc);
                FfxApiResourceDescription hfdesc = ffxApiGetImageResourceDescriptionVK(himage, hinfo, FFX_API_RESOURCE_USAGE_READ_ONLY);
                ctx.fgHudlessResource = ffxApiGetResourceVK(reinterpret_cast<void*>(himage), hfdesc, layoutToFfxState((VkImageLayout)hres->state));
                ctx.fgHudlessVkFormat = hfmt;
                ctx.fgHudlessCompatible = true;
                ctx.fgHaveHudless = true;
            } else if (convertHudless(ctx, cmd, himage, hfmt, (VkImageLayout)hres->state, hdesc.width, hdesc.height)) {
                ctx.fgHudlessVkFormat = ctx.fgBackBufferVkFormat;  // bridged image is now the backbuffer format
                ctx.fgHudlessCompatible = true;
                ctx.fgHaveHudless = true;
            }
        }

        if (ctx.fgContext && ctx.fgWrappedSwapchain != VK_NULL_HANDLE) {
            ffxDispatchDescFrameGenerationPrepare prep{};
            prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            prep.frameID = ctx.frameID;
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
                // Render thread; races the present-thread interpolation dispatch on the same
                // FG context without this lock (see g_fgDispatchMutex).
                std::lock_guard<std::mutex> lock(g_fgDispatchMutex);
                ctx.fgPreparedThisFrame = (ffxDispatchSEH(ctx.ffxApi, &ctx.fgContext, &prep.header) == FFX_API_RETURN_OK);
            }
        }
    }

    for (auto it = restore.rbegin(); it != restore.rend(); ++it)
        recordBarrier(ctx, cmd, { it->image, it->oldLayout, it->newLayout, it->aspect });

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
    const bool want = options.enabled == Boolean::eTrue;
    if (want != ctx.fgEnabled) {
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
    // Re-entrant FFX inner swapchain creation: create it on the GENUINE driver and Skip the interposer's
    // remaining hooks so sl.dlss_g never wraps FFX's swapchain (the device-lost collision).
    if (g_creatingFgSwapchain.load(std::memory_order_acquire) && ctx.realDeviceProcAddr) {
        auto realCreate = reinterpret_cast<PFN_vkCreateSwapchainKHR>(ctx.realDeviceProcAddr(Device, "vkCreateSwapchainKHR"));
        if (realCreate) {
            VkResult r = realCreate(Device, CreateInfo, Allocator, Swapchain);
            Skip = true;
            SL_LOG_INFO("sl.fsr: FFX inner CreateSwapchainKHR routed to genuine loader (rc=%d)", (int)r);
            return r;
        }
    }
    SL_LOG_INFO("sl.fsr: CreateSwapchainKHR (fgEnabled=%d) — %s",
        (int)ctx.fgEnabled,
        (ctx.platform == RenderAPI::eVulkan && ctx.fgEnabled) ? "will wrap" : "pass-through");
    if (ctx.physicalDevice == VK_NULL_HANDLE) {
        chi::PhysicalDevice physical{};
        if (ctx.compute) { ctx.compute->getPhysicalDevice(physical); ctx.physicalDevice = (VkPhysicalDevice)physical; }
    }
    // Wrap with the FFX FG swapchain ONLY while FSR FG is the active method (ctx.fgEnabled). When FSR FG is
    // off (disabled, or DLSS-G selected) we leave DXVK's plain swapchain so present flows back through the
    // SL interposer — that is the path DLSS-G needs. The present hook drives the recreate that re-enters
    // this hook on every on/off transition, so FSR FG ↔ DLSS-G ↔ disabled switch in-game. Gating on
    // fgEnabled (never set at cold init / menus) also avoids the no-frame-production crash.
    if (ctx.platform != RenderAPI::eVulkan || !ctx.fgEnabled) {
        Skip = false;
        return VK_SUCCESS;
    }
    if (!ctx.ffxModule && !loadFFX(ctx)) {
        Skip = false;
        return VK_SUCCESS;
    }
    ctx.fgBootstrapInFlight = false;  // the bootstrap recreate we asked for has arrived
    if (createFgSwapchain(ctx, Device, CreateInfo, Allocator, Swapchain)) {
        Skip = true;  // *Swapchain now holds FFX's wrapped handle
        ctx.fgWrapFailed = false;
    } else {
        SL_LOG_WARN("sl.fsr: FG swapchain proxy failed; falling back to a normal swapchain");
        Skip = false;
        ctx.fgWrapFailed = true;  // don't loop on the bootstrap recreate
    }
    return VK_SUCCESS;
}

void slHookVkDestroySwapchainKHR(VkDevice /*Device*/, VkSwapchainKHR Swapchain, const VkAllocationCallbacks* /*Allocator*/, bool& Skip)
{
    auto& ctx = (*fsr::getContext());
    if (Swapchain != VK_NULL_HANDLE && Swapchain == ctx.fgWrappedSwapchain) {
        destroyFgSwapchain(ctx);  // FFX's DestroyContext destroys the real VkSwapchainKHR it owns
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

    // Bidirectional wrap/unwrap so FSR FG ↔ DLSS-G ↔ disabled switch in-game. The desired state is "wrapped
    // iff FSR FG is on" (ctx.fgEnabled); the actual state is whether THIS swapchain is our FFX handle. On a
    // mismatch, present this frame on its real path then return VK_SUBOPTIMAL_KHR so DXVK recreates its
    // swapchain — re-entering CreateSwapchainKHR, which wraps (fgEnabled) or leaves it plain (returning
    // present to SL for DLSS-G). Presenting first avoids the dangling-fence hang of a no-present OUT_OF_DATE.
    // fgBootstrapInFlight de-bounces to one trigger per recreate; fgWrapFailed stops a wrap-fail loop.
    const bool wantWrap = ctx.fgEnabled;
    const bool needRecreate = ctx.fgGameplayReached && !ctx.fgBootstrapInFlight && !ctx.fgWrapFailed &&
        ((wantWrap && ctx.fgWrappedSwapchain == VK_NULL_HANDLE && !wrapped) ||  // turn ON: install FFX wrap
         (!wantWrap && wrapped));                                               // turn OFF: drop back to plain
    if (needRecreate) {
        ctx.fgBootstrapInFlight = true;
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
        SL_LOG_INFO("sl.fsr: recreate to %s FG (present rc=%d)", wantWrap ? "ENABLE" : "DISABLE", (int)pr);
        return pr < 0 ? pr : VK_SUBOPTIMAL_KHR;
    }

    // Not our wrapped handle (FSR FG off, or an overlay's own swapchain) — pass through untouched.
    if (!wrapped) {
        Skip = false;
        return VK_SUCCESS;
    }

    // Interpolate ONLY when the host wants FG AND an FG-prepare ran this frame (else FFX reads empty
    // inputs -> GPU device-lost). FG-off or non-gameplay (menu/loading) frames present 1:1. Configure
    // per-present, matching the canonical FFX sample.
    if (ctx.fgContext) {
        ffxConfigureDescFrameGeneration cfg{};
        cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cfg.swapChain = reinterpret_cast<void*>(ctx.fgWrappedSwapchain);
        cfg.frameGenerationCallback = &FfxFrameGenDispatchCallback;
        cfg.frameGenerationCallbackUserContext = &ctx;
        cfg.frameGenerationEnabled = ctx.fgEnabled && ctx.fgPreparedThisFrame;
        cfg.allowAsyncWorkloads = false;
        cfg.flags = ctx.fgDebugFlags;                       // host debug overlays (tear/pacing lines, debug view)
        cfg.onlyPresentGenerated = ctx.fgOnlyPresentGenerated;
        cfg.generationRect = { 0, 0, (int32_t)ctx.fgDisplayW, (int32_t)ctx.fgDisplayH };
        // HUDLessColor: the scene without UI, captured this frame in fsrEndEvaluation. With it, FFX extracts the
        // UI (backbuffer - hudless) and only interpolates the scene, so the HUD doesn't ghost. Only pass it when
        // the FG context was built with a compatible hudless format (same precision group as the backbuffer) —
        // otherwise FFX would read it with the wrong format; empty HUDLessColor in that case (e.g. HDR).
        if (ctx.fgHaveHudless && ctx.fgHudlessCompatible)
            cfg.HUDLessColor = ctx.fgHudlessResource;
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

    // Resolve the REAL (un-interposed) device functions for the FFX backend + our barriers from the GENUINE
    // driver loader in System32 — NOT GetModuleHandle("vulkan-1.dll"), which under full interposition IS
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
            // Instance-level functions for the hudless bridge (physical-device memory + format queries).
            ctx.pfnGetMemProps = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(reinterpret_cast<void*>(gipa((VkInstance)instance, "vkGetPhysicalDeviceMemoryProperties")));
            ctx.pfnGetFormatProps = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(reinterpret_cast<void*>(gipa((VkInstance)instance, "vkGetPhysicalDeviceFormatProperties")));
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
    hcDestroy(ctx);
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
