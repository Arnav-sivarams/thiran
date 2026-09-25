#pragma once

#include "storage/v0/Storage.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace thiran::v0::graphics {

// Matrices are Tensor<f32,2>[4,4] values in row-major physical storage.  The
// mathematical convention is column vectors: composed transforms are P*V*M.
storage::Tensor identity();
storage::Tensor translation(float x, float y, float z);
storage::Tensor scale(float x, float y, float z);
storage::Tensor rotationX(float radians);
storage::Tensor rotationY(float radians);
storage::Tensor rotationZ(float radians);
storage::Tensor multiply(const storage::Tensor& left, const storage::Tensor& right);
std::array<float, 4> transform(const storage::Tensor& matrix,
                               const std::array<float, 4>& vector);
storage::Tensor lookAt(const std::array<float, 3>& eye,
                       const std::array<float, 3>& target,
                       const std::array<float, 3>& up);
storage::Tensor perspective(float verticalFieldOfViewRadians, float aspect,
                            float nearDistance, float farDistance);
storage::Tensor orthographic(float left, float right, float bottom, float top,
                             float nearDistance, float farDistance);

struct RenderStatistics {
    std::uint64_t inputTriangleCount = 0;
    std::uint64_t postClipTriangleCount = 0;
    std::uint64_t shadedPixelCount = 0;
    std::uint64_t depthPassCount = 0;
};

struct RenderResult;

class Framebuffer {
public:
    Framebuffer(const Framebuffer&) = default;
    Framebuffer& operator=(const Framebuffer&) = default;
    Framebuffer(Framebuffer&&) = default;
    Framebuffer& operator=(Framebuffer&&) = default;

    std::uint64_t width() const noexcept { return width_; }
    std::uint64_t height() const noexcept { return height_; }
    const storage::Tensor& color() const noexcept { return color_; }
    const storage::Tensor& depth() const noexcept { return depth_; }
    std::array<float, 4> pixelColor(std::uint64_t x, std::uint64_t y) const;
    float pixelDepth(std::uint64_t x, std::uint64_t y) const;
    std::string colorDigest() const;
    std::string depthDigest() const;

private:
    Framebuffer(std::uint64_t width, std::uint64_t height,
                storage::Tensor color, storage::Tensor depth);
    std::uint64_t width_ = 0;
    std::uint64_t height_ = 0;
    storage::Tensor color_;
    storage::Tensor depth_;
    friend struct RenderResult;
    friend RenderResult render(const storage::Tensor&, const storage::Tensor&,
                               const storage::Tensor&, const storage::Tensor&,
                               const storage::Tensor&, const storage::Tensor&,
                               std::uint64_t, std::uint64_t,
                               std::array<float, 4>);
};

struct RenderResult {
    Framebuffer framebuffer;
    RenderStatistics statistics;
};

// positions: [N,3] f32; indices: [T,3] i64; colors: [N,4] f32.
// Geometry and transforms are read-only aliases for the duration of the call.
// The returned framebuffer owns fresh color [H*W,4] and depth [H*W] storage.
RenderResult render(const storage::Tensor& positions,
                    const storage::Tensor& indices,
                    const storage::Tensor& colors,
                    const storage::Tensor& model,
                    const storage::Tensor& view,
                    const storage::Tensor& projection,
                    std::uint64_t width,
                    std::uint64_t height,
                    std::array<float, 4> clearColor = {0.0f, 0.0f, 0.0f, 0.0f});

// Deterministic binary PPM (P6), RGB only. Alpha and depth remain available
// through the framebuffer tensors.
void writePpm(const Framebuffer& framebuffer, const std::filesystem::path& path);

} // namespace thiran::v0::graphics
