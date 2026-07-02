# FidelityFX prebuilt VK backend

`amd_fidelityfx_vk.dll` (+ import lib/exp) is the FFX-API Vulkan backend that
`sl.fsr` resolves at runtime for FSR upscaling and frame generation. It was
built from `external/fidelityfx-sdk` (the FidelityFX-SDK submodule, pinned to
v1.1.4 = `c6efa6b`, the last release before the 2.0 "Kits" restructure removed
`ffx-api/`) with one local patch applied:

- `../patches/ffx-api-try2-macro.patch` — wraps the `TRY2` macro in
  `ffx-api/src/ffx_provider.h` in `do { } while(0)` for macro hygiene.

To rebuild: apply the patch to the submodule, then run `BuildFfxApiDll.bat`
(kept here; originally lived inside the vendored `ffx-api/` tree — adjust its
paths to point at the submodule checkout).

`sl.fsr` itself compiles against `external/fidelityfx-sdk/ffx-api/include`
only (byte-identical to upstream v1.1.4); the patch affects DLL rebuilds, not
the plugin build.
