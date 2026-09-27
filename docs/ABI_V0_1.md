# Thiran 0.1.0 ABI and format inventory

This inventory is for the V0.1 release boundary. Values are taken from the
named source constants and are checked against
`release/THIRAN-0.1.0.json` by the TH-025 release validator.

| Persistent boundary | Value | Source authority |
|---|---:|---|
| `.tha` artifact format | 0 | `artifactFormatVersion` in `include/artifact/v0/NativeArtifacts.hpp` |
| Compiler/artifact ABI | 1 | `compilerArtifactAbiVersion` in the same header |
| Native artifact runtime ABI | 1 | `nativeRuntimeAbiVersion` in the same header |
| Research extension ABI | 1 | `extensionAbiVersion` in `include/extension/v0/Extension.hpp` |
| `.thm` model bundle format | 0 | `modelBundleFormatVersion` in `include/model/v0/ModelBundle.hpp` |
| Model runtime ABI | 1 | `modelRuntimeAbiVersion` in the same header |
| `.thc` checkpoint format | 0 | `checkpointFormatVersion` in `include/training/v0/Checkpoint.hpp` |
| Training checkpoint ABI | 1 | `trainingCheckpointAbiVersion` in the same header |

The product version and these ABI values are independent. Moving the product
version to 0.1.0 does not change an accepted persistent representation.
Persistent ABI and format compatibility across future 0.x releases is not yet
guaranteed. `.tha`, `.thm`, and `.thc` inputs and native extension libraries
are trusted local inputs, not signed or sandboxed interchange formats.
