#pragma once

#include "graphics/v0/Graphics.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace thiran::v0::graphics::detail {

struct ClipVertex {
    std::array<float, 4> position{};
    std::array<float, 4> color{};
};

struct ScreenVertex {
    double x = 0.0;
    double y = 0.0;
    double depth = 0.0;
    double reciprocalW = 0.0;
    std::array<float, 4> color{};
};

using ClipTriangle = std::array<ClipVertex, 3>;
using ScreenTriangle = std::array<ScreenVertex, 3>;

struct PreparedRender {
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    std::uint64_t pixelCount = 0;
    std::array<float, 4> clearColor{};
    RenderStatistics statistics;
    std::vector<ClipTriangle> triangles;
};

// Shared TH-019 authority for validation, f32 transform sequencing and fixed-order
// homogeneous clipping. Backends consume the resulting deterministic packet.
PreparedRender prepareRender(const storage::Tensor& positions,
                             const storage::Tensor& indices,
                             const storage::Tensor& colors,
                             const storage::Tensor& model,
                             const storage::Tensor& view,
                             const storage::Tensor& projection,
                             std::uint64_t width,
                             std::uint64_t height,
                             std::array<float, 4> clearColor);

// Shared viewport/orientation setup. Degenerate or unsafe post-clip triangles
// return false and are ignored identically by CPU and GPU backends.
bool prepareScreenTriangle(const ClipTriangle& triangle,
                           std::uint64_t width,
                           std::uint64_t height,
                           ScreenTriangle& output);

} // namespace thiran::v0::graphics::detail
