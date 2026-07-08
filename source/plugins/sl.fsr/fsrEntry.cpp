/*
* Community Shaders sl.fsr plugin — AMD FidelityFX FSR3 upscaling ONLY (feature kFeatureFSR).
*
* Drives FFX-API (amd_fidelityfx_vk.dll) on DXVK's VkDevice through Streamline's evaluate mechanism.
* The host tags ScalingInputColor / ScalingOutputColor / Depth / MotionVectors + sets common Constants
* (jitter, camera) and calls slEvaluateFeature(kFeatureFSR); fsrEndEvaluation records ffxDispatch(upscale)
* onto the SL-provided command buffer. FSR3 FRAME GENERATION lives in a SEPARATE, independently-loadable
* plugin (kFeatureFSR_G / sl.fsr_g), a structural twin of sl.dlss_g — see source/plugins/sl.fsr_g.
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

    std::mutex optionsMutex;
    std::map<uint32_t, FSROptions> upscaleOptions;
    int64_t upscaleLastQpc = 0;              // QPC of the last upscale dispatch, for real frameTimeDelta
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
    // FSR UPSCALE tags: color in/out + depth + MV, all required. (Frame generation is a separate feature —
    // kFeatureFSR_G / sl.fsr_g — which tags depth/MV/hudless on its own viewport.)
    getTaggedResource(kBufferTypeScalingInputColor, colorIn, evd.frame, evd.id, false, inputs, numInputs);
    getTaggedResource(kBufferTypeScalingOutputColor, colorOut, evd.frame, evd.id, false, inputs, numInputs);
    getTaggedResource(kBufferTypeDepth, depth, evd.frame, evd.id, false, inputs, numInputs);
    getTaggedResource(kBufferTypeMotionVectors, mvec, evd.frame, evd.id, false, inputs, numInputs);
    if (!colorIn || !colorOut || !depth || !mvec)
        return Result::eErrorMissingInputParameter;

    const uint32_t renderW = colorIn.getExtent().width;
    const uint32_t renderH = colorIn.getExtent().height;
    const uint32_t outputW = colorOut.getExtent().width;
    const uint32_t outputH = colorOut.getExtent().height;
    if (!renderW || !renderH)
        return Result::eErrorMissingInputParameter;

    // Resolve this viewport's options and the FFX context BEFORE recording any layout-transition
    // barriers. Both early-outs below (mode == eOff, and a context-creation failure) return without
    // running the epilogue restore loop, so doing them first is what keeps the depth/MV transitions
    // from leaking and desyncing DXVK's image-layout tracking.
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

    // Depth + MV feed the upscale; wrap+transition once.
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

    {
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
    // sl.interposer.dll. The DxvkFfxGetDeviceProcAddr shim routes the FFX upscale backend through
    // realDeviceProcAddr; vkCmdPipelineBarrier drives our resource transitions. The interposer already
    // self-loaded this same genuine module, so LoadLibrary just bumps its refcount.
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
            if (ctx.realDeviceProcAddr)
                ctx.vkCmdPipelineBarrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(ctx.realDeviceProcAddr((VkDevice)dev, "vkCmdPipelineBarrier"));
        }
    }

    SL_LOG_INFO("sl.fsr plugin started");
    return true;
}

void slOnPluginShutdown()
{
    auto& ctx = (*fsr::getContext());
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
}

SL_EXPORT void* slGetPluginFunction(const char* functionName)
{
    SL_EXPORT_FUNCTION(slOnPluginLoad);
    SL_EXPORT_FUNCTION(slOnPluginShutdown);
    SL_EXPORT_FUNCTION(slOnPluginStartup);

    SL_EXPORT_FUNCTION(slFSRSetOptions);
    SL_EXPORT_FUNCTION(slFSRGetState);
    SL_EXPORT_FUNCTION(slFSRGetOptimalSettings);

    return nullptr;
}

}
