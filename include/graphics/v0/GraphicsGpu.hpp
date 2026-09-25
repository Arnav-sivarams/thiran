#pragma once

#include "backend/v0/NativeGpu.hpp"
#include "graphics/v0/Graphics.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace thiran::v0::graphics {

struct GpuRenderEvidence {
    backend::GpuExecutionEvidence execution;
    std::uint64_t inputTriangleCount = 0;
    std::uint64_t postClipTriangleCount = 0;
};

struct GpuRenderResult {
    std::optional<RenderResult> frame;
    std::optional<backend::GpuError> error;
    GpuRenderEvidence evidence;
    bool ok() const noexcept { return frame.has_value() && !error.has_value(); }
};

struct PendingGpuRenderState;
struct GpuRenderSubmission;

// Pending graphics never exposes partially downloaded Tensor storage. Only a
// successful observation publishes a Tensor-backed RenderResult.
class PendingGpuRender {
public:
    PendingGpuRender() = default;
    PendingGpuRender(const PendingGpuRender&) = delete;
    PendingGpuRender& operator=(const PendingGpuRender&) = delete;
    PendingGpuRender(PendingGpuRender&&) noexcept = default;
    PendingGpuRender& operator=(PendingGpuRender&&) noexcept = default;
    bool valid() const noexcept;
    runtime::AsyncOperationState state() const noexcept;
    bool observed() const noexcept;
    std::optional<RenderResult> value() const;
    runtime::AsyncResource outputResource() const;
    GpuRenderEvidence evidence() const noexcept;
    GpuRenderResult observe() noexcept;
private:
    std::shared_ptr<PendingGpuRenderState> state_;
    explicit PendingGpuRender(std::shared_ptr<PendingGpuRenderState>);
    friend struct GpuRenderSubmission;
    friend GpuRenderSubmission submitRenderGpuAsync(
        const storage::Tensor&, const storage::Tensor&, const storage::Tensor&,
        const storage::Tensor&, const storage::Tensor&, const storage::Tensor&,
        std::uint64_t, std::uint64_t, std::array<float, 4>, int) noexcept;
};

struct GpuRenderSubmission {
    std::optional<PendingGpuRender> pending;
    std::optional<backend::GpuError> error;
    GpuRenderEvidence evidence;
    bool ok() const noexcept { return pending.has_value() && !error.has_value(); }
};

GpuRenderSubmission submitRenderGpuAsync(
    const storage::Tensor& positions,
    const storage::Tensor& indices,
    const storage::Tensor& colors,
    const storage::Tensor& model,
    const storage::Tensor& view,
    const storage::Tensor& projection,
    std::uint64_t width,
    std::uint64_t height,
    std::array<float, 4> clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
    int device = 0) noexcept;

// Exactly submitRenderGpuAsync followed by observe. There is no CPU fallback.
GpuRenderResult renderGpu(
    const storage::Tensor& positions,
    const storage::Tensor& indices,
    const storage::Tensor& colors,
    const storage::Tensor& model,
    const storage::Tensor& view,
    const storage::Tensor& projection,
    std::uint64_t width,
    std::uint64_t height,
    std::array<float, 4> clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
    int device = 0) noexcept;

// Deterministic backend-owned PTX for audit/tests. Empty in CPU-only builds.
std::string emitGraphicsGpuPtx();

} // namespace thiran::v0::graphics
