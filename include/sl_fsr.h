/*
* Community Shaders sl.fsr plugin — public host API (UPSCALE ONLY).
*
* AMD FidelityFX FSR3 upscaling under kFeatureFSR, driven through the same Streamline interposer as
* DLSS. The host tags inputs (slSetTagForFrame) + sets common constants (slSetConstants) and calls
* slEvaluateFeature(kFeatureFSR, ...) exactly like the DLSS path. FSR3 FRAME GENERATION lives in a
* SEPARATE, independently-loadable feature/plugin (kFeatureFSR_G / sl.fsr_g, see sl_fsr_g.h) so it is
* interchangeable with DLSS-G behind identical hooking — only one FG feature loads at a time.
*
* Mirrors sl_dlss.h (upscale); the frame-gen half mirrors sl_dlss_g.h (see sl_fsr_g.h).
*/

#pragma once

namespace sl
{

//! FSR upscale quality mode (maps to FfxApiUpscaleQualityMode in the plugin).
enum class FSRMode : uint32_t
{
    eOff,
    eMaxPerformance,    //!< 50% (2.0x per axis)
    eBalanced,          //!< 59%
    eMaxQuality,        //!< 67% (1.5x per axis)
    eUltraPerformance,  //!< 33% (3.0x per axis)
    eNativeAA,          //!< 100% (FSR native anti-aliasing)
    eCount,
};

// {B1F2A9C0-3D4E-4A5B-9C6D-7E8F90A1B2C3}
SL_STRUCT_BEGIN(FSROptions, StructType({ 0xb1f2a9c0, 0x3d4e, 0x4a5b, { 0x9c, 0x6d, 0x7e, 0x8f, 0x90, 0xa1, 0xb2, 0xc3 } }), kStructVersion1)
    //! Which quality mode to use (eOff disables the feature for this viewport).
    FSRMode mode = FSRMode::eOff;
    //! Final output (display) target width.
    uint32_t outputWidth = INVALID_UINT;
    //! Final output (display) target height.
    uint32_t outputHeight = INVALID_UINT;
    //! RCAS sharpening in [0,1].
    float sharpness = 0.0f;
    //! Whether the tagged color buffers are HDR (affects FFX tonemap handling).
    Boolean colorBuffersHDR = Boolean::eFalse;
    //! IMPORTANT: New members go here or chain a new struct (see sl_struct.h).
SL_STRUCT_END()

// {C2A3B4D5-6E7F-4081-9A2B-3C4D5E6F7081}
SL_STRUCT_BEGIN(FSROptimalSettings, StructType({ 0xc2a3b4d5, 0x6e7f, 0x4081, { 0x9a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81 } }), kStructVersion1)
    uint32_t optimalRenderWidth{};
    uint32_t optimalRenderHeight{};
    float optimalSharpness{};
SL_STRUCT_END()

// {D3B4C5E6-7F80-4192-A3B4-C5D6E7F80192}
SL_STRUCT_BEGIN(FSRState, StructType({ 0xd3b4c5e6, 0x7f80, 0x4192, { 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x01, 0x92 } }), kStructVersion1)
    uint64_t estimatedVRAMUsageInBytes{};
SL_STRUCT_END()

// FSR frame-generation options/state + their APIs live in sl_fsr_g.h (feature kFeatureFSR_G).

}

//! Provides optimal FSR render resolution for the requested quality mode. NOT thread safe.
using PFun_slFSRGetOptimalSettings = sl::Result(const sl::FSROptions& options, sl::FSROptimalSettings& settings);

//! FSR state (VRAM estimate) for the given viewport. NOT thread safe.
using PFun_slFSRGetState = sl::Result(const sl::ViewportHandle& viewport, sl::FSRState& state);

//! Set per-viewport FSR upscale options (mode/output/sharpness). NOT thread safe.
using PFun_slFSRSetOptions = sl::Result(const sl::ViewportHandle& viewport, const sl::FSROptions& options);

//! HELPERS
inline sl::Result slFSRGetOptimalSettings(const sl::FSROptions& options, sl::FSROptimalSettings& settings)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureFSR, slFSRGetOptimalSettings);
    return s_slFSRGetOptimalSettings(options, settings);
}

inline sl::Result slFSRGetState(const sl::ViewportHandle& viewport, sl::FSRState& state)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureFSR, slFSRGetState);
    return s_slFSRGetState(viewport, state);
}

inline sl::Result slFSRSetOptions(const sl::ViewportHandle& viewport, const sl::FSROptions& options)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureFSR, slFSRSetOptions);
    return s_slFSRSetOptions(viewport, options);
}
