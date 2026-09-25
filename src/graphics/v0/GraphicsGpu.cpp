#include "graphics/v0/GraphicsGpu.hpp"

#include "graphics/v0/GraphicsShared.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef THIRAN_ENABLE_NATIVE_GPU
#define THIRAN_ENABLE_NATIVE_GPU 0
#endif

namespace thiran::v0::graphics {
namespace {

template<class T>
void append(std::vector<std::byte>& bytes, const T& value) {
    const auto oldSize = bytes.size();
    const auto newSize = storage::checkedAdd(oldSize, sizeof(T));
    if (newSize > bytes.max_size()) throw std::length_error("TH020-PACKET-SIZE");
    bytes.resize(static_cast<std::size_t>(newSize));
    std::memcpy(bytes.data() + oldSize, &value, sizeof(T));
}

std::string graphicsPtx() {
    return R"ptx(.version 6.0
.target sm_50
.address_size 64

.visible .entry thiran_graphics_raster(
 .param .u64 params,
 .param .u64 vertices,
 .param .u64 colors,
 .param .u64 out_color,
 .param .u64 out_depth,
 .param .u64 out_stats) {
 .reg .pred %p<32>;
 .reg .b32 %r<32>;
 .reg .b64 %rd<48>;
 .reg .f32 %f<32>;
 .reg .f64 %fd<96>;

 ld.param.u64 %rd1,[params];
 ld.param.u64 %rd2,[vertices];
 ld.param.u64 %rd3,[colors];
 ld.param.u64 %rd4,[out_color];
 ld.param.u64 %rd5,[out_depth];
 ld.param.u64 %rd6,[out_stats];
 mov.u32 %r1,%ctaid.x;
 mov.u32 %r2,%ntid.x;
 mov.u32 %r3,%tid.x;
 cvt.u64.u32 %rd7,%r1;
 cvt.u64.u32 %rd8,%r2;
 mul.lo.u64 %rd9,%rd7,%rd8;
 cvt.u64.u32 %rd10,%r3;
 add.u64 %rd11,%rd9,%rd10;
 ld.global.u64 %rd12,[%rd1+0];
 ld.global.u64 %rd13,[%rd1+8];
 mul.lo.u64 %rd14,%rd12,%rd13;
 setp.ge.u64 %p1,%rd11,%rd14;
 @%p1 bra DONE;

 rem.u64 %rd15,%rd11,%rd12;
 div.u64 %rd16,%rd11,%rd12;
 cvt.rn.f64.u64 %fd1,%rd15;
 cvt.rn.f64.u64 %fd2,%rd16;
 ld.global.f64 %fd80,[%rd1+40];
 add.rn.f64 %fd1,%fd1,%fd80;
 add.rn.f64 %fd2,%fd2,%fd80;
 ld.global.f64 %fd81,[%rd1+48];
 ld.global.f64 %fd82,[%rd1+56];
 ld.global.f64 %fd83,[%rd1+64];
 ld.global.f64 %fd84,[%rd1+72];
 ld.global.f32 %f1,[%rd1+24];
 ld.global.f32 %f2,[%rd1+28];
 ld.global.f32 %f3,[%rd1+32];
 ld.global.f32 %f4,[%rd1+36];
 mov.b32 %f5,0f7F800000;
 mov.u64 %rd17,0;
 mov.u64 %rd18,0;
 mov.u64 %rd19,0;
 ld.global.u64 %rd20,[%rd1+16];

TRIANGLE_LOOP:
 setp.ge.u64 %p2,%rd19,%rd20;
 @%p2 bra STORE_PIXEL;
 mul.lo.u64 %rd21,%rd19,96;
 add.u64 %rd22,%rd2,%rd21;
 ld.global.f64 %fd10,[%rd22+0];
 ld.global.f64 %fd11,[%rd22+8];
 ld.global.f64 %fd12,[%rd22+16];
 ld.global.f64 %fd13,[%rd22+24];
 ld.global.f64 %fd14,[%rd22+32];
 ld.global.f64 %fd15,[%rd22+40];
 ld.global.f64 %fd16,[%rd22+48];
 ld.global.f64 %fd17,[%rd22+56];
 ld.global.f64 %fd18,[%rd22+64];
 ld.global.f64 %fd19,[%rd22+72];
 ld.global.f64 %fd20,[%rd22+80];
 ld.global.f64 %fd21,[%rd22+88];

 // edge(v1,v2,p)
 sub.rn.f64 %fd30,%fd18,%fd14;
 sub.rn.f64 %fd31,%fd2,%fd15;
 mul.rn.f64 %fd32,%fd30,%fd31;
 sub.rn.f64 %fd33,%fd19,%fd15;
 sub.rn.f64 %fd34,%fd1,%fd14;
 mul.rn.f64 %fd35,%fd33,%fd34;
 sub.rn.f64 %fd40,%fd32,%fd35;
 setp.gt.f64 %p3,%fd40,%fd81;
 @%p3 bra EDGE0_OK;
 setp.ne.f64 %p4,%fd40,%fd81;
 @%p4 bra NEXT_TRIANGLE;
 setp.lt.f64 %p5,%fd33,%fd81;
 @%p5 bra EDGE0_OK;
 setp.ne.f64 %p6,%fd33,%fd81;
 @%p6 bra NEXT_TRIANGLE;
 setp.gt.f64 %p7,%fd30,%fd81;
 @!%p7 bra NEXT_TRIANGLE;
EDGE0_OK:
 // edge(v2,v0,p)
 sub.rn.f64 %fd30,%fd10,%fd18;
 sub.rn.f64 %fd31,%fd2,%fd19;
 mul.rn.f64 %fd32,%fd30,%fd31;
 sub.rn.f64 %fd33,%fd11,%fd19;
 sub.rn.f64 %fd34,%fd1,%fd18;
 mul.rn.f64 %fd35,%fd33,%fd34;
 sub.rn.f64 %fd41,%fd32,%fd35;
 setp.gt.f64 %p3,%fd41,%fd81;
 @%p3 bra EDGE1_OK;
 setp.ne.f64 %p4,%fd41,%fd81;
 @%p4 bra NEXT_TRIANGLE;
 setp.lt.f64 %p5,%fd33,%fd81;
 @%p5 bra EDGE1_OK;
 setp.ne.f64 %p6,%fd33,%fd81;
 @%p6 bra NEXT_TRIANGLE;
 setp.gt.f64 %p7,%fd30,%fd81;
 @!%p7 bra NEXT_TRIANGLE;
EDGE1_OK:
 // edge(v0,v1,p)
 sub.rn.f64 %fd30,%fd14,%fd10;
 sub.rn.f64 %fd31,%fd2,%fd11;
 mul.rn.f64 %fd32,%fd30,%fd31;
 sub.rn.f64 %fd33,%fd15,%fd11;
 sub.rn.f64 %fd34,%fd1,%fd10;
 mul.rn.f64 %fd35,%fd33,%fd34;
 sub.rn.f64 %fd42,%fd32,%fd35;
 setp.gt.f64 %p3,%fd42,%fd81;
 @%p3 bra EDGE2_OK;
 setp.ne.f64 %p4,%fd42,%fd81;
 @%p4 bra NEXT_TRIANGLE;
 setp.lt.f64 %p5,%fd33,%fd81;
 @%p5 bra EDGE2_OK;
 setp.ne.f64 %p6,%fd33,%fd81;
 @%p6 bra NEXT_TRIANGLE;
 setp.gt.f64 %p7,%fd30,%fd81;
 @!%p7 bra NEXT_TRIANGLE;
EDGE2_OK:
 // positive area edge(v0,v1,v2)
 sub.rn.f64 %fd30,%fd14,%fd10;
 sub.rn.f64 %fd31,%fd19,%fd11;
 mul.rn.f64 %fd32,%fd30,%fd31;
 sub.rn.f64 %fd33,%fd15,%fd11;
 sub.rn.f64 %fd34,%fd18,%fd10;
 mul.rn.f64 %fd35,%fd33,%fd34;
 sub.rn.f64 %fd43,%fd32,%fd35;
 div.rn.f64 %fd44,%fd40,%fd43;
 div.rn.f64 %fd45,%fd41,%fd43;
 div.rn.f64 %fd46,%fd42,%fd43;
 mul.rn.f64 %fd47,%fd44,%fd12;
 mul.rn.f64 %fd48,%fd45,%fd16;
 add.rn.f64 %fd47,%fd47,%fd48;
 mul.rn.f64 %fd48,%fd46,%fd20;
 add.rn.f64 %fd47,%fd47,%fd48;
 setp.neu.f64 %p8,%fd47,%fd47;
 @%p8 bra NEXT_TRIANGLE;
 setp.lt.f64 %p9,%fd47,%fd82;
 @%p9 bra NEXT_TRIANGLE;
 setp.gt.f64 %p10,%fd47,%fd83;
 @%p10 bra NEXT_TRIANGLE;
 max.f64 %fd47,%fd47,%fd81;
 min.f64 %fd47,%fd47,%fd84;
 mul.rn.f64 %fd49,%fd44,%fd13;
 mul.rn.f64 %fd50,%fd45,%fd17;
 add.rn.f64 %fd49,%fd49,%fd50;
 mul.rn.f64 %fd50,%fd46,%fd21;
 add.rn.f64 %fd49,%fd49,%fd50;
 setp.neu.f64 %p8,%fd49,%fd49;
 @%p8 bra NEXT_TRIANGLE;
 setp.le.f64 %p9,%fd49,%fd81;
 @%p9 bra NEXT_TRIANGLE;

 mul.lo.u64 %rd23,%rd19,48;
 add.u64 %rd24,%rd3,%rd23;
 // perspective-correct RGBA, unrolled
 ld.global.f32 %f10,[%rd24+0]; cvt.f64.f32 %fd60,%f10;
 mul.rn.f64 %fd60,%fd44,%fd60; mul.rn.f64 %fd60,%fd60,%fd13;
 ld.global.f32 %f10,[%rd24+16]; cvt.f64.f32 %fd61,%f10;
 mul.rn.f64 %fd61,%fd45,%fd61; mul.rn.f64 %fd61,%fd61,%fd17;
 add.rn.f64 %fd60,%fd60,%fd61;
 ld.global.f32 %f10,[%rd24+32]; cvt.f64.f32 %fd61,%f10;
 mul.rn.f64 %fd61,%fd46,%fd61; mul.rn.f64 %fd61,%fd61,%fd21;
 add.rn.f64 %fd60,%fd60,%fd61; div.rn.f64 %fd60,%fd60,%fd49;
 setp.neu.f64 %p8,%fd60,%fd60; @%p8 bra NEXT_TRIANGLE;
 max.f64 %fd60,%fd60,%fd81; min.f64 %fd60,%fd60,%fd84; cvt.rn.f32.f64 %f20,%fd60;

 ld.global.f32 %f10,[%rd24+4]; cvt.f64.f32 %fd60,%f10;
 mul.rn.f64 %fd60,%fd44,%fd60; mul.rn.f64 %fd60,%fd60,%fd13;
 ld.global.f32 %f10,[%rd24+20]; cvt.f64.f32 %fd61,%f10;
 mul.rn.f64 %fd61,%fd45,%fd61; mul.rn.f64 %fd61,%fd61,%fd17;
 add.rn.f64 %fd60,%fd60,%fd61;
 ld.global.f32 %f10,[%rd24+36]; cvt.f64.f32 %fd61,%f10;
 mul.rn.f64 %fd61,%fd46,%fd61; mul.rn.f64 %fd61,%fd61,%fd21;
 add.rn.f64 %fd60,%fd60,%fd61; div.rn.f64 %fd60,%fd60,%fd49;
 setp.neu.f64 %p8,%fd60,%fd60; @%p8 bra NEXT_TRIANGLE;
 max.f64 %fd60,%fd60,%fd81; min.f64 %fd60,%fd60,%fd84; cvt.rn.f32.f64 %f21,%fd60;

 ld.global.f32 %f10,[%rd24+8]; cvt.f64.f32 %fd60,%f10;
 mul.rn.f64 %fd60,%fd44,%fd60; mul.rn.f64 %fd60,%fd60,%fd13;
 ld.global.f32 %f10,[%rd24+24]; cvt.f64.f32 %fd61,%f10;
 mul.rn.f64 %fd61,%fd45,%fd61; mul.rn.f64 %fd61,%fd61,%fd17;
 add.rn.f64 %fd60,%fd60,%fd61;
 ld.global.f32 %f10,[%rd24+40]; cvt.f64.f32 %fd61,%f10;
 mul.rn.f64 %fd61,%fd46,%fd61; mul.rn.f64 %fd61,%fd61,%fd21;
 add.rn.f64 %fd60,%fd60,%fd61; div.rn.f64 %fd60,%fd60,%fd49;
 setp.neu.f64 %p8,%fd60,%fd60; @%p8 bra NEXT_TRIANGLE;
 max.f64 %fd60,%fd60,%fd81; min.f64 %fd60,%fd60,%fd84; cvt.rn.f32.f64 %f22,%fd60;

 ld.global.f32 %f10,[%rd24+12]; cvt.f64.f32 %fd60,%f10;
 mul.rn.f64 %fd60,%fd44,%fd60; mul.rn.f64 %fd60,%fd60,%fd13;
 ld.global.f32 %f10,[%rd24+28]; cvt.f64.f32 %fd61,%f10;
 mul.rn.f64 %fd61,%fd45,%fd61; mul.rn.f64 %fd61,%fd61,%fd17;
 add.rn.f64 %fd60,%fd60,%fd61;
 ld.global.f32 %f10,[%rd24+44]; cvt.f64.f32 %fd61,%f10;
 mul.rn.f64 %fd61,%fd46,%fd61; mul.rn.f64 %fd61,%fd61,%fd21;
 add.rn.f64 %fd60,%fd60,%fd61; div.rn.f64 %fd60,%fd60,%fd49;
 setp.neu.f64 %p8,%fd60,%fd60; @%p8 bra NEXT_TRIANGLE;
 max.f64 %fd60,%fd60,%fd81; min.f64 %fd60,%fd60,%fd84; cvt.rn.f32.f64 %f23,%fd60;

 add.u64 %rd17,%rd17,1;
 cvt.rn.f32.f64 %f24,%fd47;
 setp.lt.f32 %p11,%f24,%f5;
 @!%p11 bra NEXT_TRIANGLE;
 mov.f32 %f5,%f24;
 mov.f32 %f1,%f20; mov.f32 %f2,%f21; mov.f32 %f3,%f22; mov.f32 %f4,%f23;
 add.u64 %rd18,%rd18,1;

NEXT_TRIANGLE:
 add.u64 %rd19,%rd19,1;
 bra TRIANGLE_LOOP;

STORE_PIXEL:
 mul.lo.u64 %rd25,%rd11,16;
 add.u64 %rd26,%rd4,%rd25;
 st.global.f32 [%rd26+0],%f1;
 st.global.f32 [%rd26+4],%f2;
 st.global.f32 [%rd26+8],%f3;
 st.global.f32 [%rd26+12],%f4;
 mul.lo.u64 %rd27,%rd11,4;
 add.u64 %rd28,%rd5,%rd27;
 st.global.f32 [%rd28],%f5;
 mul.lo.u64 %rd29,%rd11,16;
 add.u64 %rd30,%rd6,%rd29;
 st.global.u64 [%rd30+0],%rd17;
 st.global.u64 [%rd30+8],%rd18;
DONE:
 ret;
}
)ptx";
}

backend::GpuError validationError(const std::exception& error) {
    return {backend::GpuErrorCategory::ValidationFailure, "GPU-GRAPHICS-VALIDATION", error.what()};
}

} // namespace

struct PendingGpuRenderState {
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    std::uint64_t pixelCount = 0;
    RenderStatistics statistics;
    backend::PendingNativeGpuKernel kernel;
    std::optional<RenderResult> completed;
    std::optional<backend::GpuError> failure;

    GpuRenderEvidence evidence() const noexcept {
        return {kernel.evidence(), statistics.inputTriangleCount,
                statistics.postClipTriangleCount};
    }
};

std::string emitGraphicsGpuPtx() {
#if THIRAN_ENABLE_NATIVE_GPU
    return graphicsPtx();
#else
    return {};
#endif
}

PendingGpuRender::PendingGpuRender(std::shared_ptr<PendingGpuRenderState> state)
    : state_(std::move(state)) {}
bool PendingGpuRender::valid() const noexcept { return state_ && state_->kernel.valid(); }
runtime::AsyncOperationState PendingGpuRender::state() const noexcept {
    return state_ ? state_->kernel.state() : runtime::AsyncOperationState::Failed;
}
bool PendingGpuRender::observed() const noexcept {
    return state_ && state_->kernel.observed();
}
std::optional<RenderResult> PendingGpuRender::value() const {
    return state_ && state_->kernel.observed() && !state_->failure ? state_->completed : std::nullopt;
}
runtime::AsyncResource PendingGpuRender::outputResource() const {
    return state_ ? state_->kernel.outputResource() : runtime::AsyncResource{};
}
GpuRenderEvidence PendingGpuRender::evidence() const noexcept {
    return state_ ? state_->evidence() : GpuRenderEvidence{};
}

GpuRenderResult PendingGpuRender::observe() noexcept {
    GpuRenderResult result;
    if (!valid()) {
        result.error = backend::GpuError{backend::GpuErrorCategory::RuntimeFailure,
            "GPU-GRAPHICS-INVALID-HANDLE", "GPU graphics async handle is invalid"};
        return result;
    }
    if (state_->completed) {
        result.frame = state_->completed;
        result.evidence = state_->evidence();
        return result;
    }
    const auto observed = state_->kernel.observe();
    if (!observed.ok()) {
        state_->failure = observed.error.value_or(backend::GpuError{
            backend::GpuErrorCategory::RuntimeFailure, "GPU-GRAPHICS-OBSERVE",
            "GPU graphics observation failed"});
        result.error = state_->failure;
        result.evidence = state_->evidence();
        return result;
    }
    try {
        if (observed.outputs->size() != 3)
            throw std::runtime_error("GPU-GRAPHICS-OUTPUT-ARITY");
        const auto colorCount = storage::checkedMultiply(state_->pixelCount, 4);
        const auto colorBytes = storage::checkedByteCount(colorCount, storage::DType::F32);
        const auto depthBytes = storage::checkedByteCount(state_->pixelCount, storage::DType::F32);
        const auto statCount = storage::checkedMultiply(state_->pixelCount, 2);
        const auto statBytes = storage::checkedByteCount(statCount, storage::DType::I64);
        if ((*observed.outputs)[0].size() != colorBytes ||
            (*observed.outputs)[1].size() != depthBytes ||
            (*observed.outputs)[2].size() != statBytes)
            throw std::runtime_error("GPU-GRAPHICS-OUTPUT-SIZE");
        std::vector<float> colors(static_cast<std::size_t>(colorCount));
        std::vector<float> depths(static_cast<std::size_t>(state_->pixelCount));
        std::vector<std::uint64_t> stats(static_cast<std::size_t>(statCount));
        if (colorBytes) std::memcpy(colors.data(), (*observed.outputs)[0].data(), colorBytes);
        if (depthBytes) std::memcpy(depths.data(), (*observed.outputs)[1].data(), depthBytes);
        if (statBytes) std::memcpy(stats.data(), (*observed.outputs)[2].data(), statBytes);
        for (std::uint64_t pixel = 0; pixel < state_->pixelCount; ++pixel) {
            state_->statistics.shadedPixelCount = storage::checkedAdd(
                state_->statistics.shadedPixelCount, stats[static_cast<std::size_t>(pixel * 2)]);
            state_->statistics.depthPassCount = storage::checkedAdd(
                state_->statistics.depthPassCount, stats[static_cast<std::size_t>(pixel * 2 + 1)]);
        }
        auto framebuffer = makeFramebuffer(state_->width, state_->height,
            storage::Tensor::materializeF32({state_->pixelCount, 4}, colors),
            storage::Tensor::materializeF32({state_->pixelCount}, depths));
        state_->completed = RenderResult{std::move(framebuffer), state_->statistics};
        result.frame = state_->completed;
    } catch (const std::exception& error) {
        state_->failure = backend::GpuError{backend::GpuErrorCategory::RuntimeFailure,
                                            "GPU-GRAPHICS-MATERIALIZE", error.what()};
        result.error = state_->failure;
    }
    result.evidence = state_->evidence();
    return result;
}

GpuRenderSubmission submitRenderGpuAsync(
    const storage::Tensor& positions, const storage::Tensor& indices,
    const storage::Tensor& colors, const storage::Tensor& model,
    const storage::Tensor& view, const storage::Tensor& projection,
    std::uint64_t width, std::uint64_t height, std::array<float, 4> clearColor,
    int device) noexcept {
    GpuRenderSubmission result;
    try {
        auto prepared = detail::prepareRender(positions, indices, colors, model, view, projection,
                                              width, height, clearColor);
        std::vector<detail::ScreenTriangle> triangles;
        triangles.reserve(prepared.triangles.size());
        for (const auto& triangle : prepared.triangles) {
            detail::ScreenTriangle screen;
            if (detail::prepareScreenTriangle(triangle, width, height, screen))
                triangles.push_back(screen);
        }

        backend::NativeGpuKernelRequest request;
        request.ptx = graphicsPtx();
        request.entry = "thiran_graphics_raster";
        request.workItems = prepared.pixelCount;
        request.inputBuffers.resize(3);
        append(request.inputBuffers[0], width);
        append(request.inputBuffers[0], height);
        const auto triangleCount = static_cast<std::uint64_t>(triangles.size());
        append(request.inputBuffers[0], triangleCount);
        for (float component : clearColor) append(request.inputBuffers[0], component);
        for (double value : {0.5, 0.0, -1.0e-6, 1.000001, 1.0})
            append(request.inputBuffers[0], value);
        for (const auto& triangle : triangles) {
            for (const auto& vertex : triangle) {
                append(request.inputBuffers[1], vertex.x);
                append(request.inputBuffers[1], vertex.y);
                append(request.inputBuffers[1], vertex.depth);
                append(request.inputBuffers[1], vertex.reciprocalW);
            }
            for (const auto& vertex : triangle)
                for (float component : vertex.color) append(request.inputBuffers[2], component);
        }
        const auto colorCount = storage::checkedMultiply(prepared.pixelCount, 4);
        const auto statCount = storage::checkedMultiply(prepared.pixelCount, 2);
        request.outputByteCounts = {
            storage::checkedByteCount(colorCount, storage::DType::F32),
            storage::checkedByteCount(prepared.pixelCount, storage::DType::F32),
            storage::checkedByteCount(statCount, storage::DType::I64)};
        request.retainedReadResources = {
            positions.asyncResource(), indices.asyncResource(), colors.asyncResource(),
            model.asyncResource(), view.asyncResource(), projection.asyncResource()};
        auto submitted = backend::submitNativeGpuKernelAsync(std::move(request), device);
        result.evidence = {submitted.evidence, prepared.statistics.inputTriangleCount,
                           prepared.statistics.postClipTriangleCount};
        if (!submitted.ok()) {
            result.error = submitted.error;
            return result;
        }
        auto state = std::make_shared<PendingGpuRenderState>();
        state->width = width;
        state->height = height;
        state->pixelCount = prepared.pixelCount;
        state->statistics = prepared.statistics;
        state->kernel = std::move(*submitted.pending);
        result.pending = PendingGpuRender(state);
        result.evidence = state->evidence();
        return result;
    } catch (const std::exception& error) {
        result.error = validationError(error);
    } catch (...) {
        result.error = backend::GpuError{backend::GpuErrorCategory::ValidationFailure,
                                         "GPU-GRAPHICS-VALIDATION",
                                         "unknown GPU graphics validation failure"};
    }
    return result;
}

GpuRenderResult renderGpu(
    const storage::Tensor& positions, const storage::Tensor& indices,
    const storage::Tensor& colors, const storage::Tensor& model,
    const storage::Tensor& view, const storage::Tensor& projection,
    std::uint64_t width, std::uint64_t height, std::array<float, 4> clearColor,
    int device) noexcept {
    auto submitted = submitRenderGpuAsync(positions, indices, colors, model, view, projection,
                                          width, height, clearColor, device);
    if (!submitted.ok()) return {{}, submitted.error, submitted.evidence};
    return submitted.pending->observe();
}

} // namespace thiran::v0::graphics
