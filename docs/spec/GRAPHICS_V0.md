# Numerical graphics V0

TH-019 adds a bounded, headless native CPU graphics runtime. It is a typed
library beside the structured semantic/runtime pipeline, not a second source
language, scene graph, historical `Graph` revival, GUI, game engine, shader
system, external renderer, or GPU path.

```text
Tensor geometry + Tensor transforms
  -> vertex transform
  -> homogeneous six-plane clipping
  -> perspective divide and viewport mapping
  -> structured triangle/pixel loops
  -> explicit private framebuffer mutation
  -> immutable Tensor-backed color/depth result
```

The public V0 surface is `graphics/v0/Graphics.hpp`. It accepts positions as
`Tensor<f32,2>[N,3]`, triangle indices as `Tensor<i64,2>[T,3]`, vertex colors as
`Tensor<f32,2>[N,4]`, and model/view/projection matrices as
`Tensor<f32,2>[4,4]`. Results own fresh row-major color `[H*W,4]` and depth
`[H*W]` tensors. A rank-three image type is intentionally unnecessary.

Input tensor handles and views are read-only for the render call. The renderer
does not retain them. A private `FrameBuilder` exclusively owns its color and
depth vectors while raster loops mutate them, then seals each vector into new
storage. The returned tensors do not alias geometry or each other, and no
copy-on-write or raw user pointer is present. This construction is the runtime
counterpart of TH-006 exclusive mutation; existing StorageObjectId, descriptor,
checked-allocation, view-retention, and TH-014 access checks remain unchanged.

## Mathematical conventions

- Matrix elements are physically row-major, independently of the mathematical
  rule. Vectors are mathematical column vectors and transform composition is
  `projection * view * model * [x,y,z,1]`.
- World/view space is right-handed. `lookAt` looks from `eye` toward `target`;
  visible view-space points are along negative Z.
- Clip space is `-w <= x <= w`, `-w <= y <= w`, and `0 <= z <= w`.
  Consequently NDC X/Y are `[-1,1]` and normalized depth is `[0,1]`.
- The viewport origin is the upper left. NDC `(-1,+1)` maps to `(0,0)` and
  `(+1,-1)` maps to `(width,height)`. Pixel `(x,y)` is sampled at
  `(x+0.5,y+0.5)`.
- Counter-clockwise winding in NDC is front-facing. V0 performs no face
  culling, so both windings rasterize after deterministic orientation
  normalization.
- Vertex and clear RGBA components are finite values in `[0,1]`. Interpolated
  values are clamped to that interval after finite checking.
- Coverage uses oriented double-precision edge functions and an exact
  top-left rule in screen coordinates. With positive normalized screen area,
  an edge is inclusive when `dy < 0`, or when `dy == 0 && dx > 0`. Other
  boundary edges are exclusive.
- Depth is initialized to positive infinity as the untouched sentinel. A
  finite fragment in `[0,1]` passes only when `incoming < stored`; therefore
  nearer distinct depth is order-independent and the first submitted fragment
  wins at exactly equal depth.

Transform, projection, and clipping arithmetic is explicitly sequenced in f32
without fast-math assumptions. Raster edge/barycentric calculations use
ordinary IEEE binary64 to make the top-left decision and interpolation
numerically robust. The qualification promise is deterministic output for an
identical input on the same qualified platform/profile, not cross-architecture
bitwise reproducibility.

## Checked transforms and geometry

The runtime provides identity, translation, scale, X/Y/Z rotation, matrix
multiplication, vector transformation, right-handed look-at, perspective, and
orthographic construction. Camera/projection constructors reject non-finite
parameters, a zero/negative aspect ratio, invalid field of view, invalid
near/far order, coincident eye/target, and an up vector parallel to the view
direction. Matrix dtype and exact `[4,4]` shape are checked independently of
physical contiguity, so valid retained tensor views remain readable.

Geometry validation precedes indexed access or framebuffer allocation. It
checks dtype, rank and trailing extents, equal position/color counts, every
signed index before conversion, finite positions, and finite in-range colors.
Negative and out-of-range indices are rejected. Repeated indices, collinear
triangles, and zero/near-zero screen area are legal inputs that conservatively
produce no fragments. Non-finite source or transformed positions fail the
call. A post-clip vertex with `w <= 1e-7` conservatively rejects that emitted
triangle before division.

## Clipping and rasterization

Each indexed triangle is transformed once into homogeneous clip coordinates.
Sutherland-Hodgman clipping runs in the fixed order left, right, bottom, top,
near, far. Intersections interpolate both clip position and color with the same
homogeneous edge parameter. A surviving polygon is triangulated as the stable
fan `(v0,v1,v2)`, `(v0,v2,v3)`, and so on. A triangle may therefore produce
zero, one, or multiple post-clip triangles; a near-plane crossing is not
discarded merely because one input vertex is outside.

Raster bounds derive from the clipped viewport coordinates, are clamped to the
frame, and are converted only after dimensions are limited to the signed
32-bit traversal domain. Every flattened pixel/color offset uses checked
unsigned multiply/add and per-axis bounds checks. No signed-overflow path is
used.

Screen barycentrics linearly interpolate NDC depth. RGBA uses
perspective-correct interpolation:

```text
attribute = sum(lambda_i * attribute_i / clip_w_i)
            / sum(lambda_i / clip_w_i)
```

Clipping-created colors participate identically. `shadedPixelCount` counts
valid covered fragments before depth comparison; `depthPassCount` counts
successful strict depth tests and framebuffer writes.

## Framebuffer and output

Framebuffer dimensions are unsigned runtime values. Zero width or height is
valid and produces color shape `[0,4]`, depth shape `[0]`, and no fragments.
Dimensions above `INT32_MAX`, pixel/color count overflow, byte-count overflow,
container length overflow, and allocation failure are rejected before any
raster access. There is no V0 API for importing an externally initialized
depth buffer, so invalid external depth values cannot enter this pipeline.

Color/depth digests are FNV-1a-64 over the little-endian IEEE f32 bit patterns
in logical row-major order. They are deterministic regression identities, not
cryptographic signatures. `writePpm` emits an auditable binary P6 file with
round-to-nearest RGB8 conversion; alpha and depth remain independently
available as raw tensors. Zero-sized PPM output is rejected because PPM has no
useful empty-image representation.

## Compiler boundary and exclusions

TH-015 TensorRegion remains a straight-line tensor/dataflow optimization
boundary. Eligible source-level matrix and transform expressions may use
planning/fusion before entering this runtime as backend coverage grows.
Clipping control flow, raster traversal, ordered checks, and exclusive
framebuffer mutation are structured operations and are deliberately not
misrepresented as one giant TensorRegion DAG.

The V0 source language has no module/library mechanism mature enough to expose
this API ergonomically, so TH-019 does not invent graphics syntax or compiler
intrinsics. The native runtime API establishes semantics first. It is not
embedded into TH-016 `.tha` artifacts because that ABI does not represent the
required structured mutation. Rasterization is not differentiable and has no
AD rule. There is no Python, NumPy, PyTorch, OpenCV, SDL, GLFW, Qt, OpenGL,
Vulkan, DirectX, CUDA, texture, material, lighting, or external-rasterizer
dependency. GPU graphics remains outside TH-019.

## Native GPU implementation

TH-020 implements these same semantics through a native CUDA compute backend.
Validation, transform, homogeneous clipping, stable triangulation, viewport
setup, and orientation are shared with this CPU authority. Coverage, depth,
and perspective-correct interpolation execute in a one-work-item-per-pixel PTX
kernel. The observed result uses the same `Framebuffer` Tensor representation
and PPM writer. Transfer, async, fallback, and interoperability details are
specified in `NATIVE_GPU_GRAPHICS_V0.md`.
