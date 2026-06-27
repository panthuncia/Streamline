/*
* Community Shaders sl.xess plugin — public host API.
*
* Intel XeSS super-resolution upscaling, driven through the Streamline interposer like DLSS. The host
* tags inputs (slSetTagForFrame: ScalingInputColor/ScalingOutputColor/Depth/MotionVectors) + sets common
* constants (slSetConstants) and calls slEvaluateFeature(kFeatureXeSS, ...). Upscale-only (no frame gen).
*
* Mirrors sl_dlss.h.
*/

#pragma once

namespace sl
{

//! XeSS quality mode (maps to xess_quality_settings_t in the plugin).
enum class XeSSMode : uint32_t
{
    eOff,
    eUltraPerformance,
    ePerformance,
    eBalanced,
    eQuality,
    eUltraQuality,
    eUltraQualityPlus,
    eNativeAA,
    eCount,
};

// {A1B2C3D4-5E6F-4708-8192-A3B4C5D6E7F8}
SL_STRUCT_BEGIN(XeSSOptions, StructType({ 0xa1b2c3d4, 0x5e6f, 0x4708, { 0x81, 0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8 } }), kStructVersion1)
    //! Quality mode (eOff disables the feature for this viewport).
    XeSSMode mode = XeSSMode::eOff;
    //! Final output (display) target width/height.
    uint32_t outputWidth = INVALID_UINT;
    uint32_t outputHeight = INVALID_UINT;
    //! Sharpness in [0,1].
    float sharpness = 0.0f;
    //! Whether the tagged color buffers are HDR.
    Boolean colorBuffersHDR = Boolean::eFalse;
    //! IMPORTANT: New members go here or chain a new struct (see sl_struct.h).
SL_STRUCT_END()

// {B2C3D4E5-6F70-4819-92A3-B4C5D6E7F809}
SL_STRUCT_BEGIN(XeSSOptimalSettings, StructType({ 0xb2c3d4e5, 0x6f70, 0x4819, { 0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09 } }), kStructVersion1)
    uint32_t optimalRenderWidth{};
    uint32_t optimalRenderHeight{};
    float optimalSharpness{};
SL_STRUCT_END()

// {C3D4E5F6-7081-492A-A3B4-C5D6E7F8091A}
SL_STRUCT_BEGIN(XeSSState, StructType({ 0xc3d4e5f6, 0x7081, 0x492a, { 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09, 0x1a } }), kStructVersion1)
    uint64_t estimatedVRAMUsageInBytes{};
SL_STRUCT_END()

}

//! Provides optimal XeSS render resolution for the requested quality mode. NOT thread safe.
using PFun_slXeSSGetOptimalSettings = sl::Result(const sl::XeSSOptions& options, sl::XeSSOptimalSettings& settings);

//! XeSS state (VRAM estimate) for the given viewport. NOT thread safe.
using PFun_slXeSSGetState = sl::Result(const sl::ViewportHandle& viewport, sl::XeSSState& state);

//! Set per-viewport XeSS upscale options. NOT thread safe.
using PFun_slXeSSSetOptions = sl::Result(const sl::ViewportHandle& viewport, const sl::XeSSOptions& options);

//! HELPERS
inline sl::Result slXeSSGetOptimalSettings(const sl::XeSSOptions& options, sl::XeSSOptimalSettings& settings)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureXeSS, slXeSSGetOptimalSettings);
    return s_slXeSSGetOptimalSettings(options, settings);
}

inline sl::Result slXeSSGetState(const sl::ViewportHandle& viewport, sl::XeSSState& state)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureXeSS, slXeSSGetState);
    return s_slXeSSGetState(viewport, state);
}

inline sl::Result slXeSSSetOptions(const sl::ViewportHandle& viewport, const sl::XeSSOptions& options)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureXeSS, slXeSSSetOptions);
    return s_slXeSSSetOptions(viewport, options);
}
