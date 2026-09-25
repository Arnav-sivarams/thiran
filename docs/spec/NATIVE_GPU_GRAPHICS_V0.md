# Native GPU graphics V0

TH-020 implements the `GRAPHICS_V0` raster contract with native CUDA compute.
The public typed runtime surface is `graphics/v0/GraphicsGpu.hpp`; it introduces
no source-language CUDA terms and no graphics-driver API.

## Pipeline boundary

Submission first calls the shared TH-019 preparation authority. On the CPU it
validates ordinary storage-backed tensors and matrices, evaluates
`projection * view * model`, performs fixed-order six-plane homogeneous
clipping, creates the stable clipped-triangle fan, and performs the shared
perspective-divide, viewport, degeneracy, and winding normalization step.

The resulting backend-owned packet contains screen-space binary64 X/Y/depth
and reciprocal-W values plus f32 RGBA attributes. One physical GPU work item
owns one framebuffer pixel. It walks clipped triangles in submission order,
evaluates the exact TH-019 top-left coverage rule with binary64 operations,
interpolates depth, performs perspective-correct RGBA interpolation, applies
strict-less depth, and writes one final color/depth value and per-pixel
statistics. No fragment atomics or race-dependent ordering exist. Zero-pixel
frames perform no kernel launch.

The PTX is deterministic backend-owned text loaded through the existing CUDA
Driver path. It uses explicit round-to-nearest f32/f64 operations and contains
no approximate, flush-to-zero, or fast-math modifier. CUDA contexts, streams,
events, modules, allocations, pointers, grid dimensions, and PTX names remain
inside `backend/v0/NativeGpu`; they are not graphics or language semantics.

## Async and ownership

`submitRenderGpuAsync` returns a `PendingGpuRender`, not a framebuffer. The
operation uses TH-014 read reservations for the six Tensor inputs and a write
reservation for its unpublished output identity. Reservations normalize by
underlying storage identity, so aliased matrices or geometry do not acquire
duplicate reservations. The pending state retains the clipped packet, host
transfer buffers, CUDA context/module/stream/event, and RAII device allocations
through observation or dropped-handle drain.

Observation synchronizes only the operation's event, surfaces deferred Driver
errors, completes D2H transfers, sums deterministic statistics, and then
materializes fresh ordinary color `[H*W,4]` and depth `[H*W]` f32 tensors.
Repeated observation returns the same completed logical result. Independent
pending renders own independent contexts/resources; observing one does not
release another.

## Interoperability and transfers

Inputs and observed outputs are ordinary validated Thiran `Tensor` objects.
Color and depth have independent fresh storage and do not alias inputs. PPM
output therefore works without a GPU-specific image format.

V0 has an explicit host-materialization boundary:

```text
GPU raster -> D2H staging -> ordinary Tensor storage
ordinary Tensor -> later native GPU TensorRegion -> H2D again
```

There is no device-local graphics-output handle and no zero-copy claim. This
keeps TH-013 through TH-016 ownership and artifact contracts unchanged. A
persistent cross-subsystem device-resident tensor model is future work.

## Errors and fallback

Shared malformed geometry/matrix/framebuffer checks are reported as validation
failures before any CUDA launch. Disabled backend, unavailable driver/device,
and invalid device selection remain availability errors. Context, allocation,
transfer, PTX/module, launch, synchronization, and D2H failures remain native
GPU execution errors. `renderGpu` is exactly submission followed by
observation; it never calls the CPU renderer and never retries on CPU.

CPU selection remains the existing `render` entry point and does not initialize
CUDA. GPU selection is the explicit `renderGpu`/`submitRenderGpuAsync` entry
point. No source syntax, TH-016 graphics artifact ABI, external renderer,
OpenGL, Vulkan, Direct3D, CUDA graphics interop, or window system is added.

## Reproducibility scope

Qualification requires exact CPU/GPU coverage masks and tight f32 value
agreement for the reference scene matrix. Same-device repeated renders are
required to produce identical bytes/digests. Cross-device bitwise identity and
rendering-performance claims are outside V0.
