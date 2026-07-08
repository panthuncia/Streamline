/*
* Community Shaders sl.fsr_g plugin — public host API (FRAME GENERATION ONLY).
*
* AMD FidelityFX FSR3 frame generation under kFeatureFSR_G, split out of sl.fsr (kFeatureFSR, upscale)
* so it is an independently loadable Streamline feature — a structural twin of sl.dlss_g. Frame
* generation is enabled via slFSRFrameGenerationSetOptions and inserted at the plugin's own present
* hook (an FFX FrameInterpolationSwapChain), exactly as sl.dlss_g inserts DLSS-G at its present hook.
* Only one FG feature (kFeatureFSR_G or kFeatureDLSS_G) is loaded at a time, so the host's load/unload
* + present-hook behaviour is identical regardless of which frame-generation method is selected.
*
* The host tags depth/MV/hudless on a DEDICATED frame-gen viewport (no color) and calls
* slEvaluateFeature(kFeatureFSR_G, ...) to run the FrameGenerationPrepare, exactly like the upscale
* evaluate on kFeatureFSR. Mirrors sl_dlss_g.h.
*/

#pragma once

namespace sl
{

//! FSR frame-generation options (mirrors DLSSGOptions).
// {E4C5D6F7-8091-42A3-B4C5-D6E7F8091203}
SL_STRUCT_BEGIN(FSRFrameGenOptions, StructType({ 0xe4c5d6f7, 0x8091, 0x42a3, { 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09, 0x12, 0x03 } }), kStructVersion1)
    //! True to enable FSR3 frame generation (present-hook interpolation).
    Boolean enabled = Boolean::eFalse;
    //! Render (dynamic) size of the tagged depth/MV inputs.
    uint32_t renderWidth = INVALID_UINT;
    uint32_t renderHeight = INVALID_UINT;
    //! Display (back-buffer) size.
    uint32_t displayWidth = INVALID_UINT;
    uint32_t displayHeight = INVALID_UINT;
    //! Whether the presented color is HDR + its peak luminance (for FFX tonemap/transfer-function).
    Boolean colorBuffersHDR = Boolean::eFalse;
    float hdrPeakNits = 1000.0f;
    //! FSR3 frame-generation debug overlays, forwarded to ffxConfigureDescFrameGeneration every present.
    Boolean debugView = Boolean::eFalse;             //! FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW
    Boolean debugTearLines = Boolean::eFalse;        //! FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES
    Boolean debugPacingLines = Boolean::eFalse;      //! FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_PACING_LINES
    Boolean onlyPresentGenerated = Boolean::eFalse;  //! present only generated frames (FFX onlyPresentGenerated)
SL_STRUCT_END()

//! FSR frame-generation state (mirrors DLSSGState).
// {F5D6E7081-9102-43B4-C5D6-E7F8091203A4}
SL_STRUCT_BEGIN(FSRFrameGenState, StructType({ 0xf5d6e708, 0x9102, 0x43b4, { 0xc5, 0xd6, 0xe7, 0xf8, 0x09, 0x12, 0x03, 0xa4 } }), kStructVersion1)
    //! 0 == OK.
    uint32_t status{};
    //! Frames actually presented per app frame (2 == frame generation doubling is active).
    uint32_t numFramesActuallyPresented{};
    uint64_t estimatedVRAMUsageInBytes{};
SL_STRUCT_END()

}

//! Enable/disable + configure FSR frame generation for the given viewport. NOT thread safe.
using PFun_slFSRFrameGenerationSetOptions = sl::Result(const sl::ViewportHandle& viewport, const sl::FSRFrameGenOptions& options);

//! FSR frame-generation state incl. presented-frame count. NOT thread safe.
using PFun_slFSRGetFrameGenState = sl::Result(const sl::ViewportHandle& viewport, sl::FSRFrameGenState& state);

//! HELPERS
inline sl::Result slFSRFrameGenerationSetOptions(const sl::ViewportHandle& viewport, const sl::FSRFrameGenOptions& options)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureFSR_G, slFSRFrameGenerationSetOptions);
    return s_slFSRFrameGenerationSetOptions(viewport, options);
}

inline sl::Result slFSRGetFrameGenState(const sl::ViewportHandle& viewport, sl::FSRFrameGenState& state)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureFSR_G, slFSRGetFrameGenState);
    return s_slFSRGetFrameGenState(viewport, state);
}
