#include "graphics/v0/Graphics.hpp"
#include "graphics/v0/GraphicsShared.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thiran::v0::graphics {
namespace {

[[noreturn]] void fail(const char* id) { throw std::runtime_error(id); }

bool finite(float value) { return std::isfinite(value); }

float add(float left, float right) {
    volatile float value = left + right;
    return value;
}

float multiplyFloat(float left, float right) {
    volatile float value = left * right;
    return value;
}

float multiplyAdd(float total, float left, float right) {
    return add(total, multiplyFloat(left, right));
}

double subtractProducts(double a, double b, double c, double d) {
    volatile double first = a * b;
    volatile double second = c * d;
    volatile double value = first - second;
    return value;
}

std::vector<float> matrixValues(const storage::Tensor& matrix) {
    const auto& descriptor = matrix.descriptor();
    if (descriptor.dtype != storage::DType::F32) fail("TH019-MATRIX-DTYPE");
    if (descriptor.shape != std::vector<std::uint64_t>{4, 4}) fail("TH019-MATRIX-SHAPE");
    auto values = matrix.logicalF32Values();
    if (!std::all_of(values.begin(), values.end(), finite)) fail("TH019-NONFINITE-MATRIX");
    return values;
}

storage::Tensor matrix(std::vector<float> values) {
    if (values.size() != 16 || !std::all_of(values.begin(), values.end(), finite))
        fail("TH019-NONFINITE-MATRIX");
    return storage::Tensor::materializeF32({4, 4}, values);
}

void requireFinite(std::initializer_list<float> values, const char* id) {
    for (float value : values) if (!finite(value)) fail(id);
}

std::array<float, 3> subtract(const std::array<float, 3>& left,
                              const std::array<float, 3>& right) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

float dot(const std::array<float, 3>& left, const std::array<float, 3>& right) {
    float result = 0.0f;
    for (std::size_t index = 0; index < 3; ++index)
        result = multiplyAdd(result, left[index], right[index]);
    return result;
}

std::array<float, 3> cross(const std::array<float, 3>& left,
                           const std::array<float, 3>& right) {
    return {
        add(multiplyFloat(left[1], right[2]), -multiplyFloat(left[2], right[1])),
        add(multiplyFloat(left[2], right[0]), -multiplyFloat(left[0], right[2])),
        add(multiplyFloat(left[0], right[1]), -multiplyFloat(left[1], right[0]))
    };
}

std::array<float, 3> normalize(const std::array<float, 3>& value, const char* id) {
    const float lengthSquared = dot(value, value);
    if (!finite(lengthSquared) || lengthSquared <= 0.0f) fail(id);
    const float length = std::sqrt(lengthSquared);
    if (!finite(length) || length <= std::numeric_limits<float>::epsilon()) fail(id);
    std::array<float, 3> result{};
    for (std::size_t index = 0; index < 3; ++index) result[index] = value[index] / length;
    if (!std::all_of(result.begin(), result.end(), finite)) fail(id);
    return result;
}

using detail::ClipTriangle;
using detail::ClipVertex;
using detail::PreparedRender;
using detail::ScreenTriangle;
using detail::ScreenVertex;

enum class ClipPlane { Left, Right, Bottom, Top, Near, Far };

float distance(const ClipVertex& vertex, ClipPlane plane) {
    const auto& position = vertex.position;
    switch (plane) {
    case ClipPlane::Left: return add(position[0], position[3]);
    case ClipPlane::Right: return add(position[3], -position[0]);
    case ClipPlane::Bottom: return add(position[1], position[3]);
    case ClipPlane::Top: return add(position[3], -position[1]);
    case ClipPlane::Near: return position[2];
    case ClipPlane::Far: return add(position[3], -position[2]);
    }
    fail("TH019-CLIP-PLANE");
}

ClipVertex interpolate(const ClipVertex& first, const ClipVertex& second, float amount) {
    ClipVertex result;
    for (std::size_t component = 0; component < 4; ++component) {
        result.position[component] = add(first.position[component],
            multiplyFloat(amount, add(second.position[component], -first.position[component])));
        result.color[component] = add(first.color[component],
            multiplyFloat(amount, add(second.color[component], -first.color[component])));
    }
    return result;
}

std::vector<ClipVertex> clipAgainst(const std::vector<ClipVertex>& input, ClipPlane plane) {
    std::vector<ClipVertex> output;
    if (input.empty()) return output;
    output.reserve(input.size() + 1);
    ClipVertex previous = input.back();
    float previousDistance = distance(previous, plane);
    bool previousInside = previousDistance >= 0.0f;
    for (const auto& current : input) {
        const float currentDistance = distance(current, plane);
        const bool currentInside = currentDistance >= 0.0f;
        if (previousInside != currentInside) {
            const float denominator = add(previousDistance, -currentDistance);
            if (denominator == 0.0f || !finite(denominator)) fail("TH019-CLIP-NUMERIC");
            const float amount = previousDistance / denominator;
            if (!finite(amount)) fail("TH019-CLIP-NUMERIC");
            output.push_back(interpolate(previous, current, amount));
        }
        if (currentInside) output.push_back(current);
        previous = current;
        previousDistance = currentDistance;
        previousInside = currentInside;
    }
    return output;
}

std::vector<ClipVertex> clipTriangle(std::vector<ClipVertex> polygon) {
    for (ClipPlane plane : {ClipPlane::Left, ClipPlane::Right, ClipPlane::Bottom,
                            ClipPlane::Top, ClipPlane::Near, ClipPlane::Far}) {
        polygon = clipAgainst(polygon, plane);
        if (polygon.size() < 3) return {};
    }
    return polygon;
}

double edge(const ScreenVertex& first, const ScreenVertex& second, double x, double y) {
    return subtractProducts(second.x - first.x, y - first.y,
                            second.y - first.y, x - first.x);
}

bool topLeft(const ScreenVertex& first, const ScreenVertex& second) {
    const double dx = second.x - first.x;
    const double dy = second.y - first.y;
    return dy < 0.0 || (dy == 0.0 && dx > 0.0);
}

bool covered(double value, bool inclusive) {
    return value > 0.0 || (value == 0.0 && inclusive);
}

std::uint64_t checkedPixel(std::uint64_t x, std::uint64_t y, std::uint64_t width,
                           std::uint64_t height, std::uint64_t pixelCount) {
    if (x >= width || y >= height) fail("TH019-FRAMEBUFFER-OFFSET");
    const auto pixel = storage::checkedAdd(storage::checkedMultiply(y, width), x);
    if (pixel >= pixelCount) fail("TH019-FRAMEBUFFER-OFFSET");
    return pixel;
}

std::size_t checkedHostIndex(std::uint64_t value) {
    if (value > std::numeric_limits<std::size_t>::max()) fail("TH019-FRAMEBUFFER-SIZE");
    return static_cast<std::size_t>(value);
}

std::string digest(const std::vector<float>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (float value : values) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0; shift != 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(bits >> shift);
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

struct FrameBuilder {
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    std::uint64_t pixelCount = 0;
    std::vector<float> color;
    std::vector<float> depth;
};

std::uint64_t validateFramebuffer(std::uint64_t width, std::uint64_t height,
                                  const std::array<float, 4>& clearColor) {
    for (float component : clearColor)
        if (!finite(component) || component < 0.0f || component > 1.0f)
            fail("TH019-CLEAR-COLOR");
    if (width > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        height > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
        fail("TH019-FRAMEBUFFER-DIMENSION");
    const auto pixelCount = storage::checkedMultiply(width, height);
    const auto colorCount = storage::checkedMultiply(pixelCount, 4);
    (void)storage::checkedByteCount(colorCount, storage::DType::F32);
    (void)storage::checkedByteCount(pixelCount, storage::DType::F32);
    const std::vector<float> limits;
    if (colorCount > limits.max_size() || pixelCount > limits.max_size())
        fail("TH019-FRAMEBUFFER-SIZE");
    return pixelCount;
}

FrameBuilder makeFrame(std::uint64_t width, std::uint64_t height,
                       const std::array<float, 4>& clearColor) {
    FrameBuilder result;
    result.width = width;
    result.height = height;
    try {
        result.pixelCount = validateFramebuffer(width, height, clearColor);
        const auto colorCount = storage::checkedMultiply(result.pixelCount, 4);
        result.color.resize(checkedHostIndex(colorCount));
        result.depth.assign(checkedHostIndex(result.pixelCount),
                            std::numeric_limits<float>::infinity());
    } catch (const std::length_error&) {
        fail("TH019-FRAMEBUFFER-SIZE");
    } catch (const std::bad_alloc&) {
        fail("TH019-FRAMEBUFFER-ALLOCATION");
    }
    for (std::uint64_t pixel = 0; pixel < result.pixelCount; ++pixel)
        for (std::uint64_t component = 0; component < 4; ++component)
            result.color[checkedHostIndex(storage::checkedAdd(storage::checkedMultiply(pixel, 4), component))] =
                clearColor[checkedHostIndex(component)];
    return result;
}

void rasterize(const ScreenTriangle& screenTriangle, FrameBuilder& frame,
               RenderStatistics& statistics) {
    if (frame.width == 0 || frame.height == 0) return;
    const auto& vertices = screenTriangle;
    const double area = edge(vertices[0], vertices[1], vertices[2].x, vertices[2].y);

    const double minimumX = std::min({vertices[0].x, vertices[1].x, vertices[2].x});
    const double maximumX = std::max({vertices[0].x, vertices[1].x, vertices[2].x});
    const double minimumY = std::min({vertices[0].y, vertices[1].y, vertices[2].y});
    const double maximumY = std::max({vertices[0].y, vertices[1].y, vertices[2].y});
    auto firstX = static_cast<std::int64_t>(std::ceil(minimumX - 0.5));
    auto lastX = static_cast<std::int64_t>(std::floor(maximumX - 0.5));
    auto firstY = static_cast<std::int64_t>(std::ceil(minimumY - 0.5));
    auto lastY = static_cast<std::int64_t>(std::floor(maximumY - 0.5));
    firstX = std::max<std::int64_t>(firstX, 0);
    firstY = std::max<std::int64_t>(firstY, 0);
    lastX = std::min<std::int64_t>(lastX, static_cast<std::int64_t>(frame.width) - 1);
    lastY = std::min<std::int64_t>(lastY, static_cast<std::int64_t>(frame.height) - 1);
    if (firstX > lastX || firstY > lastY) return;

    const bool edge0Inclusive = topLeft(vertices[1], vertices[2]);
    const bool edge1Inclusive = topLeft(vertices[2], vertices[0]);
    const bool edge2Inclusive = topLeft(vertices[0], vertices[1]);
    for (std::int64_t y = firstY; y <= lastY; ++y) {
        for (std::int64_t x = firstX; x <= lastX; ++x) {
            const double centerX = static_cast<double>(x) + 0.5;
            const double centerY = static_cast<double>(y) + 0.5;
            const double edge0 = edge(vertices[1], vertices[2], centerX, centerY);
            const double edge1 = edge(vertices[2], vertices[0], centerX, centerY);
            const double edge2 = edge(vertices[0], vertices[1], centerX, centerY);
            if (!covered(edge0, edge0Inclusive) || !covered(edge1, edge1Inclusive) ||
                !covered(edge2, edge2Inclusive)) continue;
            const double weight0 = edge0 / area;
            const double weight1 = edge1 / area;
            const double weight2 = edge2 / area;
            double depth = weight0 * vertices[0].depth + weight1 * vertices[1].depth +
                           weight2 * vertices[2].depth;
            if (!std::isfinite(depth) || depth < -1.0e-6 || depth > 1.0 + 1.0e-6) continue;
            depth = std::clamp(depth, 0.0, 1.0);
            const double denominator = weight0 * vertices[0].reciprocalW +
                                       weight1 * vertices[1].reciprocalW +
                                       weight2 * vertices[2].reciprocalW;
            if (!std::isfinite(denominator) || denominator <= 0.0) continue;
            std::array<float, 4> color{};
            bool validColor = true;
            for (std::size_t component = 0; component < 4; ++component) {
                const double numerator =
                    weight0 * static_cast<double>(vertices[0].color[component]) * vertices[0].reciprocalW +
                    weight1 * static_cast<double>(vertices[1].color[component]) * vertices[1].reciprocalW +
                    weight2 * static_cast<double>(vertices[2].color[component]) * vertices[2].reciprocalW;
                const double value = numerator / denominator;
                if (!std::isfinite(value)) { validColor = false; break; }
                color[component] = static_cast<float>(std::clamp(value, 0.0, 1.0));
            }
            if (!validColor) continue;
            statistics.shadedPixelCount = storage::checkedAdd(statistics.shadedPixelCount, 1);
            const auto pixel = checkedPixel(static_cast<std::uint64_t>(x),
                                            static_cast<std::uint64_t>(y),
                                            frame.width, frame.height, frame.pixelCount);
            const auto hostPixel = checkedHostIndex(pixel);
            if (!(static_cast<float>(depth) < frame.depth[hostPixel])) continue;
            frame.depth[hostPixel] = static_cast<float>(depth);
            for (std::uint64_t component = 0; component < 4; ++component) {
                const auto offset = storage::checkedAdd(storage::checkedMultiply(pixel, 4), component);
                frame.color[checkedHostIndex(offset)] = color[checkedHostIndex(component)];
            }
            statistics.depthPassCount = storage::checkedAdd(statistics.depthPassCount, 1);
        }
    }
}

void validateGeometry(const storage::Tensor& positions,
                      const storage::Tensor& indices,
                      const storage::Tensor& colors) {
    const auto& position = positions.descriptor();
    const auto& index = indices.descriptor();
    const auto& color = colors.descriptor();
    if (position.dtype != storage::DType::F32 || color.dtype != storage::DType::F32 ||
        index.dtype != storage::DType::I64) fail("TH019-GEOMETRY-DTYPE");
    if (position.shape.size() != 2 || position.shape[1] != 3 ||
        index.shape.size() != 2 || index.shape[1] != 3 ||
        color.shape.size() != 2 || color.shape[1] != 4)
        fail("TH019-GEOMETRY-SHAPE");
    if (position.shape[0] != color.shape[0]) fail("TH019-GEOMETRY-COUNT");
    for (float value : positions.logicalF32Values())
        if (!finite(value)) fail("TH019-NONFINITE-VERTEX");
    for (float value : colors.logicalF32Values())
        if (!finite(value) || value < 0.0f || value > 1.0f) fail("TH019-COLOR-RANGE");
    for (std::int64_t value : indices.logicalI64Values())
        if (value < 0 || static_cast<std::uint64_t>(value) >= position.shape[0])
            fail("TH019-INDEX");
}

} // namespace

namespace detail {

bool prepareScreenTriangle(const ClipTriangle& triangle,
                           std::uint64_t width,
                           std::uint64_t height,
                           ScreenTriangle& output) {
    constexpr float minimumW = 1.0e-7f;
    for (std::size_t index = 0; index < 3; ++index) {
        const float w = triangle[index].position[3];
        if (!finite(w) || w <= minimumW) return false;
        const float x = triangle[index].position[0] / w;
        const float y = triangle[index].position[1] / w;
        const float z = triangle[index].position[2] / w;
        if (!finite(x) || !finite(y) || !finite(z)) return false;
        output[index].x = (static_cast<double>(x) * 0.5 + 0.5) *
                          static_cast<double>(width);
        output[index].y = (0.5 - static_cast<double>(y) * 0.5) *
                          static_cast<double>(height);
        output[index].depth = z;
        output[index].reciprocalW = 1.0 / static_cast<double>(w);
        output[index].color = triangle[index].color;
    }
    double area = edge(output[0], output[1], output[2].x, output[2].y);
    if (!std::isfinite(area) || std::fabs(area) <= 1.0e-12) return false;
    if (area < 0.0) std::swap(output[1], output[2]);
    return true;
}

PreparedRender prepareRender(const storage::Tensor& positions,
                             const storage::Tensor& indices,
                             const storage::Tensor& colors,
                             const storage::Tensor& model,
                             const storage::Tensor& view,
                             const storage::Tensor& projection,
                             std::uint64_t width,
                             std::uint64_t height,
                             std::array<float, 4> clearColor) {
    validateGeometry(positions, indices, colors);
    PreparedRender prepared;
    prepared.width = width;
    prepared.height = height;
    prepared.pixelCount = validateFramebuffer(width, height, clearColor);
    prepared.clearColor = clearColor;
    const auto modelView = multiply(view, model);
    const auto modelViewProjection = multiply(projection, modelView);
    const auto vertexCount = positions.descriptor().shape[0];
    std::vector<ClipVertex> vertices;
    vertices.reserve(checkedHostIndex(vertexCount));
    for (std::uint64_t vertex = 0; vertex < vertexCount; ++vertex) {
        ClipVertex transformed;
        transformed.position = transform(modelViewProjection,
            {positions.loadF32({vertex, 0}), positions.loadF32({vertex, 1}),
             positions.loadF32({vertex, 2}), 1.0f});
        for (std::uint64_t component = 0; component < 4; ++component)
            transformed.color[checkedHostIndex(component)] = colors.loadF32({vertex, component});
        vertices.push_back(transformed);
    }
    prepared.statistics.inputTriangleCount = indices.descriptor().shape[0];
    for (std::uint64_t triangleIndex = 0;
         triangleIndex < prepared.statistics.inputTriangleCount; ++triangleIndex) {
        std::vector<ClipVertex> polygon;
        polygon.reserve(3);
        for (std::uint64_t corner = 0; corner < 3; ++corner) {
            const auto index = static_cast<std::uint64_t>(indices.loadI64({triangleIndex, corner}));
            polygon.push_back(vertices[checkedHostIndex(index)]);
        }
        polygon = clipTriangle(std::move(polygon));
        if (polygon.size() < 3) continue;
        for (std::size_t corner = 1; corner + 1 < polygon.size(); ++corner) {
            prepared.statistics.postClipTriangleCount = storage::checkedAdd(
                prepared.statistics.postClipTriangleCount, 1);
            prepared.triangles.push_back({polygon[0], polygon[corner], polygon[corner + 1]});
        }
    }
    return prepared;
}

} // namespace detail

storage::Tensor identity() {
    return matrix({1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1});
}

storage::Tensor translation(float x, float y, float z) {
    requireFinite({x, y, z}, "TH019-NONFINITE-TRANSFORM");
    return matrix({1, 0, 0, x,
                   0, 1, 0, y,
                   0, 0, 1, z,
                   0, 0, 0, 1});
}

storage::Tensor scale(float x, float y, float z) {
    requireFinite({x, y, z}, "TH019-NONFINITE-TRANSFORM");
    return matrix({x, 0, 0, 0,
                   0, y, 0, 0,
                   0, 0, z, 0,
                   0, 0, 0, 1});
}

storage::Tensor rotationX(float radians) {
    requireFinite({radians}, "TH019-NONFINITE-TRANSFORM");
    const float cosine = std::cos(radians), sine = std::sin(radians);
    return matrix({1, 0, 0, 0,
                   0, cosine, -sine, 0,
                   0, sine, cosine, 0,
                   0, 0, 0, 1});
}

storage::Tensor rotationY(float radians) {
    requireFinite({radians}, "TH019-NONFINITE-TRANSFORM");
    const float cosine = std::cos(radians), sine = std::sin(radians);
    return matrix({cosine, 0, sine, 0,
                   0, 1, 0, 0,
                   -sine, 0, cosine, 0,
                   0, 0, 0, 1});
}

storage::Tensor rotationZ(float radians) {
    requireFinite({radians}, "TH019-NONFINITE-TRANSFORM");
    const float cosine = std::cos(radians), sine = std::sin(radians);
    return matrix({cosine, -sine, 0, 0,
                   sine, cosine, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1});
}

storage::Tensor multiply(const storage::Tensor& left, const storage::Tensor& right) {
    const auto a = matrixValues(left), b = matrixValues(right);
    std::vector<float> result(16, 0.0f);
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            for (std::size_t inner = 0; inner < 4; ++inner)
                result[row * 4 + column] = multiplyAdd(result[row * 4 + column],
                                                       a[row * 4 + inner],
                                                       b[inner * 4 + column]);
    return matrix(std::move(result));
}

std::array<float, 4> transform(const storage::Tensor& matrixValue,
                               const std::array<float, 4>& vector) {
    const auto values = matrixValues(matrixValue);
    for (float value : vector) if (!finite(value)) fail("TH019-NONFINITE-VERTEX");
    std::array<float, 4> result{};
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            result[row] = multiplyAdd(result[row], values[row * 4 + column], vector[column]);
    for (float value : result) if (!finite(value)) fail("TH019-NONFINITE-TRANSFORMED");
    return result;
}

storage::Tensor lookAt(const std::array<float, 3>& eye,
                       const std::array<float, 3>& target,
                       const std::array<float, 3>& up) {
    for (float value : eye) if (!finite(value)) fail("TH019-CAMERA");
    for (float value : target) if (!finite(value)) fail("TH019-CAMERA");
    for (float value : up) if (!finite(value)) fail("TH019-CAMERA");
    const auto forward = normalize(subtract(target, eye), "TH019-CAMERA-DIRECTION");
    const auto side = normalize(cross(forward, up), "TH019-CAMERA-UP");
    const auto cameraUp = cross(side, forward);
    return matrix({side[0], side[1], side[2], -dot(side, eye),
                   cameraUp[0], cameraUp[1], cameraUp[2], -dot(cameraUp, eye),
                   -forward[0], -forward[1], -forward[2], dot(forward, eye),
                   0, 0, 0, 1});
}

storage::Tensor perspective(float verticalFieldOfViewRadians, float aspect,
                            float nearDistance, float farDistance) {
    requireFinite({verticalFieldOfViewRadians, aspect, nearDistance, farDistance},
                  "TH019-PROJECTION");
    constexpr float pi = 3.14159265358979323846f;
    if (verticalFieldOfViewRadians <= 0.0f || verticalFieldOfViewRadians >= pi ||
        aspect <= 0.0f || nearDistance <= 0.0f || farDistance <= nearDistance)
        fail("TH019-PROJECTION");
    const float focal = 1.0f / std::tan(verticalFieldOfViewRadians * 0.5f);
    const float depthScale = farDistance / (nearDistance - farDistance);
    const float depthOffset = multiplyFloat(farDistance, nearDistance) /
                              (nearDistance - farDistance);
    return matrix({focal / aspect, 0, 0, 0,
                   0, focal, 0, 0,
                   0, 0, depthScale, depthOffset,
                   0, 0, -1, 0});
}

storage::Tensor orthographic(float left, float right, float bottom, float top,
                             float nearDistance, float farDistance) {
    requireFinite({left, right, bottom, top, nearDistance, farDistance},
                  "TH019-PROJECTION");
    if (right <= left || top <= bottom || nearDistance < 0.0f || farDistance <= nearDistance)
        fail("TH019-PROJECTION");
    return matrix({2.0f / (right - left), 0, 0, -(right + left) / (right - left),
                   0, 2.0f / (top - bottom), 0, -(top + bottom) / (top - bottom),
                   0, 0, 1.0f / (nearDistance - farDistance),
                   nearDistance / (nearDistance - farDistance),
                   0, 0, 0, 1});
}

Framebuffer::Framebuffer(std::uint64_t width, std::uint64_t height,
                         storage::Tensor color, storage::Tensor depth)
    : width_(width), height_(height), color_(std::move(color)), depth_(std::move(depth)) {}

Framebuffer makeFramebuffer(std::uint64_t width, std::uint64_t height,
                            storage::Tensor color, storage::Tensor depth) {
    const auto pixels = storage::checkedMultiply(width, height);
    if (color.descriptor().dtype != storage::DType::F32 ||
        color.descriptor().shape != std::vector<std::uint64_t>{pixels, 4} ||
        depth.descriptor().dtype != storage::DType::F32 ||
        depth.descriptor().shape != std::vector<std::uint64_t>{pixels})
        fail("TH019-FRAMEBUFFER-TENSOR");
    if (color.storageId() == depth.storageId()) fail("TH019-FRAMEBUFFER-ALIAS");
    return Framebuffer(width, height, std::move(color), std::move(depth));
}

std::array<float, 4> Framebuffer::pixelColor(std::uint64_t x, std::uint64_t y) const {
    const auto count = storage::checkedMultiply(width_, height_);
    const auto pixel = checkedPixel(x, y, width_, height_, count);
    return {color_.loadF32({pixel, 0}), color_.loadF32({pixel, 1}),
            color_.loadF32({pixel, 2}), color_.loadF32({pixel, 3})};
}

float Framebuffer::pixelDepth(std::uint64_t x, std::uint64_t y) const {
    const auto count = storage::checkedMultiply(width_, height_);
    return depth_.loadF32({checkedPixel(x, y, width_, height_, count)});
}

std::string Framebuffer::colorDigest() const { return digest(color_.logicalF32Values()); }
std::string Framebuffer::depthDigest() const { return digest(depth_.logicalF32Values()); }

RenderResult render(const storage::Tensor& positions,
                    const storage::Tensor& indices,
                    const storage::Tensor& colors,
                    const storage::Tensor& model,
                    const storage::Tensor& view,
                    const storage::Tensor& projection,
                    std::uint64_t width,
                    std::uint64_t height,
                    std::array<float, 4> clearColor) {
    auto prepared = detail::prepareRender(positions, indices, colors, model, view, projection,
                                          width, height, clearColor);
    auto frame = makeFrame(width, height, clearColor);
    auto statistics = prepared.statistics;
    for (const auto& triangle : prepared.triangles) {
        ScreenTriangle screen;
        if (detail::prepareScreenTriangle(triangle, width, height, screen))
            rasterize(screen, frame, statistics);
    }

    auto framebuffer = makeFramebuffer(width, height,
        storage::Tensor::materializeF32({frame.pixelCount, 4}, frame.color),
        storage::Tensor::materializeF32({frame.pixelCount}, frame.depth));
    return {std::move(framebuffer), statistics};
}

void writePpm(const Framebuffer& framebuffer, const std::filesystem::path& path) {
    if (framebuffer.width() == 0 || framebuffer.height() == 0)
        fail("TH019-PPM-DIMENSIONS");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) fail("TH019-PPM-OPEN");
    output << "P6\n" << framebuffer.width() << ' ' << framebuffer.height() << "\n255\n";
    const auto values = framebuffer.color().logicalF32Values();
    const auto pixelCount = storage::checkedMultiply(framebuffer.width(), framebuffer.height());
    for (std::uint64_t pixel = 0; pixel < pixelCount; ++pixel) {
        for (std::uint64_t component = 0; component < 3; ++component) {
            const auto offset = storage::checkedAdd(storage::checkedMultiply(pixel, 4), component);
            const float value = std::clamp(values[checkedHostIndex(offset)], 0.0f, 1.0f);
            const auto byte = static_cast<unsigned char>(std::floor(value * 255.0f + 0.5f));
            output.put(static_cast<char>(byte));
        }
    }
    if (!output) fail("TH019-PPM-WRITE");
}

} // namespace thiran::v0::graphics
