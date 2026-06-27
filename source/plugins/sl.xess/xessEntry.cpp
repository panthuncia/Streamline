/*
* Community Shaders sl.xess plugin — Intel XeSS super-resolution upscaling.
*
* Drives XeSS (libxess.dll, Vulkan backend) on DXVK's VkDevice through Streamline's evaluate mechanism.
* The host tags ScalingInputColor / ScalingOutputColor / Depth / MotionVectors + sets common Constants
* (jitter) and calls slEvaluateFeature(kFeatureXeSS); xessEndEvaluation records xessVKExecute onto the
* SL-provided command list. Upscale-only (no frame generation).
*/

#include <map>
#include <mutex>

#include "include/sl.h"
#include "include/sl_consts.h"
#include "source/core/sl.api/internal.h"
#include "source/core/sl.log/log.h"
#include "source/core/sl.plugin/plugin.h"
#include "source/core/sl.param/parameters.h"
#include "source/platforms/sl.chi/compute.h"
#include "source/plugins/sl.xess/versions.h"
#include "source/plugins/sl.common/commonInterface.h"
#include "external/json/include/nlohmann/json.hpp"
#include "_artifacts/json/xess_json.h"
#include "_artifacts/gitVersion.h"

#include "include/sl_xess.h"

// Intel XeSS public headers (xess.h / xess_vk.h; libxess.dll resolved at runtime). XeSS now lives inside
// the fork under external/xess-sdk.
#define XESS_TYPES_ONLY
#include "xess/xess.h"
#undef XESS_TYPES_ONLY
#include "xess/xess_vk.h"

using json = nlohmann::json;

namespace sl
{
namespace xess
{
//! libxess.dll entry points resolved via GetProcAddress.
struct XeSSApi
{
    HMODULE module = nullptr;
    xess_result_t (*getVersion)(xess_version_t*) = nullptr;
    xess_result_t (*vkCreateContext)(VkInstance, VkPhysicalDevice, VkDevice, xess_context_handle_t*) = nullptr;
    xess_result_t (*vkBuildPipelines)(xess_context_handle_t, VkPipelineCache, bool, uint32_t) = nullptr;
    xess_result_t (*vkInit)(xess_context_handle_t, const xess_vk_init_params_t*) = nullptr;
    xess_result_t (*vkExecute)(xess_context_handle_t, VkCommandBuffer, const xess_vk_execute_params_t*) = nullptr;
    xess_result_t (*setVelocityScale)(xess_context_handle_t, float, float) = nullptr;
    xess_result_t (*destroyContext)(xess_context_handle_t) = nullptr;
};

struct XeSSContext
{
    SL_PLUGIN_CONTEXT_CREATE_DESTROY(XeSSContext);
    void onCreateContext() {};
    void onDestroyContext() {};

    common::PFunRegisterEvaluateCallbacks* registerEvaluateCallbacks{};
    RenderAPI platform = RenderAPI::eVulkan;
    chi::ICompute* compute{};

    XeSSApi api{};
    xess_context_handle_t xessContext = nullptr;
    uint32_t ctxOutputWidth = 0, ctxOutputHeight = 0;
    XeSSMode ctxMode = XeSSMode::eOff;
    bool dispatchFaulted = false;

    std::mutex optionsMutex;
    std::map<uint32_t, XeSSOptions> options;
};
}

void updateEmbeddedJSON(json& config);

static std::string JSON = std::string(xess_json, &xess_json[xess_json_len]);

SL_PLUGIN_DEFINE("sl.xess", Version(VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH), Version(0, 0, 1), JSON.c_str(), updateEmbeddedJSON, xess, XeSSContext)

namespace
{
//! Directory holding this plugin's DLL (libxess.dll ships next to sl.xess.dll in the streamline folder).
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

template <typename T>
bool resolve(HMODULE m, T*& fn, const char* name)
{
    fn = reinterpret_cast<T*>(GetProcAddress(m, name));
    if (!fn)
        SL_LOG_WARN("sl.xess: missing libxess export '%s'", name);
    return fn != nullptr;
}

bool loadXeSS(xess::XeSSApi& api)
{
    if (api.module)
        return true;
    std::wstring dll = getPluginDir() + L"\\libxess.dll";
    api.module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!api.module)
    {
        SL_LOG_WARN("sl.xess: libxess.dll not found next to the plugin — XeSS disabled");
        return false;
    }
    bool ok =
        resolve(api.module, api.getVersion, "xessGetVersion") &&
        resolve(api.module, api.vkCreateContext, "xessVKCreateContext") &&
        resolve(api.module, api.vkBuildPipelines, "xessVKBuildPipelines") &&
        resolve(api.module, api.vkInit, "xessVKInit") &&
        resolve(api.module, api.vkExecute, "xessVKExecute") &&
        resolve(api.module, api.setVelocityScale, "xessSetVelocityScale") &&
        resolve(api.module, api.destroyContext, "xessDestroyContext");
    if (!ok)
    {
        FreeLibrary(api.module);
        api.module = nullptr;
        return false;
    }
    xess_version_t v{};
    api.getVersion(&v);
    SL_LOG_INFO("sl.xess: loaded libxess.dll v%u.%u.%u", v.major, v.minor, v.patch);
    return true;
}

xess_quality_settings_t mapQuality(XeSSMode mode)
{
    switch (mode)
    {
        case XeSSMode::eUltraPerformance: return XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
        case XeSSMode::ePerformance:      return XESS_QUALITY_SETTING_PERFORMANCE;
        case XeSSMode::eBalanced:         return XESS_QUALITY_SETTING_BALANCED;
        case XeSSMode::eQuality:          return XESS_QUALITY_SETTING_QUALITY;
        case XeSSMode::eUltraQuality:     return XESS_QUALITY_SETTING_ULTRA_QUALITY;
        case XeSSMode::eUltraQualityPlus: return XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS;
        case XeSSMode::eNativeAA:         return XESS_QUALITY_SETTING_AA;
        default:                          return XESS_QUALITY_SETTING_QUALITY;
    }
}

//! (Re)create the XeSS context for the requested output size + quality. Returns false on failure.
bool ensureContext(xess::XeSSContext& ctx, uint32_t outputWidth, uint32_t outputHeight, XeSSMode mode)
{
    if (ctx.xessContext && ctx.ctxOutputWidth == outputWidth && ctx.ctxOutputHeight == outputHeight && ctx.ctxMode == mode)
        return true;

    if (ctx.xessContext)
    {
        ctx.api.destroyContext(ctx.xessContext);
        ctx.xessContext = nullptr;
    }

    chi::Device device{}; chi::Instance instance{}; chi::PhysicalDevice physical{};
    ctx.compute->getInstance(instance);
    ctx.compute->getDevice(device);
    ctx.compute->getPhysicalDevice(physical);

    xess_result_t r = ctx.api.vkCreateContext(
        (VkInstance)instance, (VkPhysicalDevice)physical, (VkDevice)device, &ctx.xessContext);
    if (r != XESS_RESULT_SUCCESS)
    {
        SL_LOG_ERROR("sl.xess: xessVKCreateContext failed (%d)", (int)r);
        ctx.xessContext = nullptr;
        return false;
    }

    // Init-flag conventions, matched to the two working CS upscalers (DLSS + the FSR plugin) which consume
    // the SAME Skyrim depth + motion-vector buffers:
    //   * NO XESS_INIT_FLAG_JITTERED_MV — Skyrim's MVs are non-jittered (geometric motion only; the TAA
    //     jitter lives in the projection matrix). DLSS uses motionVectorsJittered=false and FSR sets no FFX
    //     jitter-cancellation flag. Setting it makes XeSS subtract a jitter that isn't there → static shimmer.
    //   * NO XESS_INIT_FLAG_INVERTED_DEPTH — Skyrim's depth (as tagged here) is standard [0..1], NOT
    //     reversed-Z. DLSS uses depthInverted=false and FSR sets no FFX_UPSCALE_ENABLE_DEPTH_INVERTED. With
    //     the flag wrongly set, XeSS reads the depth convention backwards and fails disocclusion rejection,
    //     so revealed geometry on large depth discontinuities keeps stale history → fuzzy/smeared.
    const uint32_t initFlags = XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE;
    ctx.api.vkBuildPipelines(ctx.xessContext, VK_NULL_HANDLE, true, initFlags);

    xess_vk_init_params_t init{};
    init.outputResolution = { outputWidth, outputHeight };
    init.qualitySetting = mapQuality(mode);
    init.initFlags = initFlags;
    r = ctx.api.vkInit(ctx.xessContext, &init);
    if (r != XESS_RESULT_SUCCESS)
    {
        SL_LOG_ERROR("sl.xess: xessVKInit failed (%d)", (int)r);
        ctx.api.destroyContext(ctx.xessContext);
        ctx.xessContext = nullptr;
        return false;
    }
    ctx.ctxOutputWidth = outputWidth;
    ctx.ctxOutputHeight = outputHeight;
    ctx.ctxMode = mode;
    SL_LOG_INFO("sl.xess: context created (%ux%u, quality %u)", outputWidth, outputHeight, (uint32_t)mode);
    return true;
}

xess_vk_image_view_info makeViewInfo(chi::ICompute* compute, CommonResource& cr, uint32_t w, uint32_t h, bool depth)
{
    xess_vk_image_view_info info{};
    chi::Resource res = cr;  // sl::Resource* with ->native (VkImage) + ->view (VkImageView)
    chi::ResourceDescription desc{};
    compute->getResourceDescription(res, desc);
    info.imageView = (VkImageView)res->view;
    info.image = (VkImage)res->native;
    info.format = (VkFormat)desc.nativeFormat;
    info.width = w;
    info.height = h;
    info.subresourceRange.aspectMask = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.layerCount = 1;
    return info;
}
}

sl::Result xessBeginEvaluation(chi::CommandList /*pCmdList*/, const common::EventData& /*evd*/, const sl::BaseStructure** /*inputs*/, uint32_t /*numInputs*/)
{
    return Result::eOk;
}

sl::Result xessEndEvaluation(chi::CommandList cmdList, const common::EventData& evd, const sl::BaseStructure** inputs, uint32_t numInputs)
{
    auto& ctx = (*xess::getContext());
    if (ctx.dispatchFaulted || ctx.platform != RenderAPI::eVulkan)
        return Result::eErrorNotInitialized;
    // Load libxess.dll LAZILY on the first evaluate — NEVER during slOnPluginStartup, which runs inside the
    // interposed vkCreateDevice; libxess enumerating Vulkan on load would re-enter vkCreateDevice ->
    // initializePlugins -> recursive lock on m_mtxPluginConfig (std::system_error "resource deadlock").
    if (!ctx.api.module && !loadXeSS(ctx.api))
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

    XeSSMode mode = XeSSMode::eQuality;
    {
        std::lock_guard<std::mutex> lock(ctx.optionsMutex);
        auto it = ctx.options.find((uint32_t)evd.id);
        if (it != ctx.options.end())
            mode = it->second.mode;
    }
    if (mode == XeSSMode::eOff)
        return Result::eOk;

    if (!ensureContext(ctx, outputW, outputH, mode))
    {
        ctx.dispatchFaulted = true;
        return Result::eErrorExceptionHandler;
    }

    // Skyrim's motion vectors are normalized (screen-relative UV deltas), NOT pixel-space. XeSS runs in
    // pixel-velocity mode (no XESS_INIT_FLAG_USE_NDC_VELOCITY), so scale the MVs up to render-resolution
    // pixels — exactly what the working FSR plugin does with ffx motionVectorScale = {renderW, renderH}.
    // Set per-frame (not at context creation) because the render size varies with dynamic resolution while
    // the context, keyed on output size + quality, is not recreated.
    ctx.api.setVelocityScale(ctx.xessContext, (float)renderW, (float)renderH);

    xess_vk_execute_params_t params{};
    params.colorTexture = makeViewInfo(ctx.compute, colorIn, renderW, renderH, false);
    params.depthTexture = makeViewInfo(ctx.compute, depth, renderW, renderH, true);
    params.velocityTexture = makeViewInfo(ctx.compute, mvec, renderW, renderH, false);
    params.outputTexture = makeViewInfo(ctx.compute, colorOut, outputW, outputH, false);
    // consts->jitterOffset is ALREADY negated by the host (CS EvaluateXeSS) — pass it straight through.
    // Negating again here was a double-negation (+jitter) that ghosts under motion.
    params.jitterOffsetX = consts->jitterOffset.x;
    params.jitterOffsetY = consts->jitterOffset.y;
    params.exposureScale = 1.0f;
    params.resetHistory = 0;
    params.inputWidth = renderW;
    params.inputHeight = renderH;

    xess_result_t r = ctx.api.vkExecute(ctx.xessContext, (VkCommandBuffer)cmdList, &params);
    if (r != XESS_RESULT_SUCCESS)
    {
        SL_LOG_ERROR("sl.xess: xessVKExecute failed (%d) — disabling for this session", (int)r);
        ctx.dispatchFaulted = true;
        return Result::eErrorExceptionHandler;
    }
    return Result::eOk;
}

sl::Result slXeSSSetOptions(const sl::ViewportHandle& viewport, const sl::XeSSOptions& options)
{
    auto& ctx = (*xess::getContext());
    std::lock_guard<std::mutex> lock(ctx.optionsMutex);
    ctx.options[(uint32_t)viewport] = options;
    return Result::eOk;
}

sl::Result slXeSSGetState(const sl::ViewportHandle& /*viewport*/, sl::XeSSState& state)
{
    state.estimatedVRAMUsageInBytes = 0;
    return Result::eOk;
}

sl::Result slXeSSGetOptimalSettings(const sl::XeSSOptions& options, sl::XeSSOptimalSettings& settings)
{
    float ratio = 0.667f;
    switch (options.mode)
    {
        case XeSSMode::eUltraPerformance: ratio = 0.333f; break;
        case XeSSMode::ePerformance:      ratio = 0.5f;   break;
        case XeSSMode::eBalanced:         ratio = 0.58f;  break;
        case XeSSMode::eQuality:          ratio = 0.667f; break;
        case XeSSMode::eUltraQuality:     ratio = 0.769f; break;
        case XeSSMode::eUltraQualityPlus: ratio = 0.833f; break;
        case XeSSMode::eNativeAA:         ratio = 1.0f;   break;
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

    auto& ctx = (*xess::getContext());
    auto parameters = api::getContext()->parameters;

    if (!param::getPointerParam(parameters, param::common::kPFunRegisterEvaluateCallbacks, &ctx.registerEvaluateCallbacks))
    {
        SL_LOG_ERROR("sl.xess: failed to obtain kPFunRegisterEvaluateCallbacks");
        return false;
    }
    ctx.registerEvaluateCallbacks(kFeatureXeSS, xessBeginEvaluation, xessEndEvaluation);

    json& config = *(json*)api::getContext()->loaderConfig;
    uint32_t deviceType{};
    int appId{};
    config.at("appId").get_to(appId);
    config.at("deviceType").get_to(deviceType);
    ctx.platform = (RenderAPI)deviceType;

    if (!param::getPointerParam(parameters, sl::param::common::kComputeAPI, &ctx.compute))
    {
        SL_LOG_ERROR("sl.xess: failed to obtain compute interface");
        return false;
    }

    // NOTE: libxess.dll is loaded LAZILY on first evaluate (see xessEndEvaluation), NOT here — startup runs
    // inside the interposed vkCreateDevice and loading a Vulkan-enumerating DLL here recursively re-enters
    // the plugin manager (resource-deadlock crash).
    SL_LOG_INFO("sl.xess plugin started");
    return true;
}

void slOnPluginShutdown()
{
    auto& ctx = (*xess::getContext());
    if (ctx.xessContext && ctx.api.destroyContext)
        ctx.api.destroyContext(ctx.xessContext);
    ctx.xessContext = nullptr;
    if (ctx.api.module)
        FreeLibrary(ctx.api.module);
    ctx.api.module = nullptr;
    ctx.registerEvaluateCallbacks(kFeatureXeSS, nullptr, nullptr);
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

    SL_EXPORT_FUNCTION(slXeSSSetOptions);
    SL_EXPORT_FUNCTION(slXeSSGetState);
    SL_EXPORT_FUNCTION(slXeSSGetOptimalSettings);

    return nullptr;
}

}
