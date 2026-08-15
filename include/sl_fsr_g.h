/*
 * Public host API for the Community Shaders FSR 3 frame-generation Streamline plugin.
 * The host tags depth, motion vectors, and HUD-less color on a dedicated viewport.
 */

#pragma once

namespace sl
{

//! FSR frame-generation options (mirrors DLSSGOptions).
// {E4C5D6F7-8091-42A3-B4C5-D6E7F8091203}
SL_STRUCT_BEGIN(FSRFrameGenOptions, StructType({ 0xe4c5d6f7, 0x8091, 0x42a3, { 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09, 0x12, 0x03 } }), kStructVersion1)
    //! True to enable FSR3 frame generation (present-hook interpolation).
    Boolean enabled = Boolean::eFalse;
    //! Whether the presented color is HDR.
    Boolean colorBuffersHDR = Boolean::eFalse;
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

//! Discard a prepared frame. Returns eErrorInvalidState when no frame was pending. Thread safe.
using PFun_slFSRFrameGenerationDiscardPreparedFrame = sl::Result(const sl::ViewportHandle& viewport);

//! Return whether the supplied Vulkan swapchain is the active FSR frame-generation replacement. Thread safe.
using PFun_slFSRFrameGenerationOwnsSwapchain = bool(VkSwapchainKHR swapchain);

//! Complete and confirm FSR swapchain teardown, optionally releasing the interpolation context. Thread safe.
using PFun_slFSRFrameGenerationCompleteSwapchainTeardown = bool(bool releaseFeatureContext);

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

inline sl::Result slFSRFrameGenerationDiscardPreparedFrame(const sl::ViewportHandle& viewport)
{
    SL_FEATURE_FUN_IMPORT_STATIC(sl::kFeatureFSR_G, slFSRFrameGenerationDiscardPreparedFrame);
    return s_slFSRFrameGenerationDiscardPreparedFrame(viewport);
}

inline bool slFSRFrameGenerationOwnsSwapchain(VkSwapchainKHR swapchain)
{
    static PFun_slFSRFrameGenerationOwnsSwapchain* s_slFSRFrameGenerationOwnsSwapchain{};
    if (!s_slFSRFrameGenerationOwnsSwapchain &&
        slGetFeatureFunction(sl::kFeatureFSR_G, "slFSRFrameGenerationOwnsSwapchain",
            reinterpret_cast<void*&>(s_slFSRFrameGenerationOwnsSwapchain)) != sl::Result::eOk)
        return false;
    return s_slFSRFrameGenerationOwnsSwapchain && s_slFSRFrameGenerationOwnsSwapchain(swapchain);
}

inline bool slFSRFrameGenerationCompleteSwapchainTeardown(bool releaseFeatureContext)
{
    static PFun_slFSRFrameGenerationCompleteSwapchainTeardown* s_slFSRFrameGenerationCompleteSwapchainTeardown{};
    if (!s_slFSRFrameGenerationCompleteSwapchainTeardown &&
        slGetFeatureFunction(sl::kFeatureFSR_G, "slFSRFrameGenerationCompleteSwapchainTeardown",
            reinterpret_cast<void*&>(s_slFSRFrameGenerationCompleteSwapchainTeardown)) != sl::Result::eOk)
        return false;
    return s_slFSRFrameGenerationCompleteSwapchainTeardown &&
        s_slFSRFrameGenerationCompleteSwapchainTeardown(releaseFeatureContext);
}
