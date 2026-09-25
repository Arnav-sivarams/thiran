#include "graphics/v0/Graphics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace graphics = thiran::v0::graphics;
namespace storage = thiran::v0::storage;
namespace fs = std::filesystem;

namespace {

int checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void close(float actual, float expected, float tolerance = 1.0e-5f) {
    require(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance,
            "numeric mismatch actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

template<class Function>
void rejected(Function&& function, const std::string& expected = {}) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        require(expected.empty() || error.what() == expected,
                "wrong rejection: expected=" + expected + " actual=" + error.what());
        return;
    }
    require(false, "operation was not rejected: " + expected);
}

storage::Tensor f32(std::vector<std::uint64_t> shape, std::vector<float> values) {
    return storage::Tensor::materializeF32(std::move(shape), values);
}

storage::Tensor i64(std::vector<std::uint64_t> shape, std::vector<std::int64_t> values) {
    return storage::Tensor::materializeI64(std::move(shape), values);
}

storage::Tensor solidColors(std::uint64_t count,
                            std::array<float, 4> color = {1, 1, 1, 1}) {
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(count * 4));
    for (std::uint64_t index = 0; index < count; ++index)
        values.insert(values.end(), color.begin(), color.end());
    return f32({count, 4}, std::move(values));
}

graphics::RenderResult identityRender(const storage::Tensor& positions,
                                      const storage::Tensor& indices,
                                      const storage::Tensor& colors,
                                      std::uint64_t width = 16,
                                      std::uint64_t height = 16) {
    const auto unit = graphics::identity();
    return graphics::render(positions, indices, colors, unit, unit, unit, width, height);
}

std::uint64_t finiteDepthPixels(const graphics::Framebuffer& framebuffer) {
    const auto values = framebuffer.depth().logicalF32Values();
    return static_cast<std::uint64_t>(std::count_if(
        values.begin(), values.end(), [](float value) { return std::isfinite(value); }));
}

graphics::RenderResult clippingCase(const std::vector<float>& positionValues) {
    return identityRender(f32({3, 3}, positionValues), i64({1, 3}, {0, 1, 2}),
                          solidColors(3));
}

void transformTests() {
    const auto unit = graphics::identity();
    const auto unchanged = graphics::transform(unit, {1, -2, 3, 1});
    close(unchanged[0], 1); close(unchanged[1], -2); close(unchanged[2], 3); close(unchanged[3], 1);

    const auto moved = graphics::transform(graphics::translation(2, 3, 4), {1, 2, 3, 1});
    close(moved[0], 3); close(moved[1], 5); close(moved[2], 7); close(moved[3], 1);
    const auto scaled = graphics::transform(graphics::scale(2, 3, 4), {1, 2, 3, 1});
    close(scaled[0], 2); close(scaled[1], 6); close(scaled[2], 12);

    constexpr float halfPi = 1.57079632679489661923f;
    const auto aroundX = graphics::transform(graphics::rotationX(halfPi), {0, 1, 0, 1});
    close(aroundX[0], 0); close(aroundX[1], 0); close(aroundX[2], 1);
    const auto aroundY = graphics::transform(graphics::rotationY(halfPi), {1, 0, 0, 1});
    close(aroundY[0], 0); close(aroundY[2], -1);
    const auto aroundZ = graphics::transform(graphics::rotationZ(halfPi), {1, 0, 0, 1});
    close(aroundZ[0], 0); close(aroundZ[1], 1);

    const auto composed = graphics::multiply(graphics::translation(1, 2, 3),
                                             graphics::scale(2, 2, 2));
    const auto composedPoint = graphics::transform(composed, {1, 1, 1, 1});
    close(composedPoint[0], 3); close(composedPoint[1], 4); close(composedPoint[2], 5);

    const auto view = graphics::lookAt({0, 0, 3}, {0, 0, 0}, {0, 1, 0});
    const auto viewedOrigin = graphics::transform(view, {0, 0, 0, 1});
    close(viewedOrigin[0], 0); close(viewedOrigin[1], 0); close(viewedOrigin[2], -3);

    const auto projection = graphics::perspective(halfPi, 1, 1, 11);
    const auto nearPoint = graphics::transform(projection, {0, 0, -1, 1});
    const auto farPoint = graphics::transform(projection, {0, 0, -11, 1});
    close(nearPoint[2] / nearPoint[3], 0);
    close(farPoint[2] / farPoint[3], 1);

    const auto ortho = graphics::orthographic(-2, 2, -1, 1, 1, 11);
    const auto orthoCorner = graphics::transform(ortho, {2, 1, -11, 1});
    close(orthoCorner[0], 1); close(orthoCorner[1], 1); close(orthoCorner[2], 1);

    rejected([] { (void)graphics::perspective(1, 0, 1, 10); }, "TH019-PROJECTION");
    rejected([] { (void)graphics::perspective(1, 1, 0, 10); }, "TH019-PROJECTION");
    rejected([] { (void)graphics::perspective(1, 1, 2, 1); }, "TH019-PROJECTION");
    rejected([] { (void)graphics::perspective(std::numeric_limits<float>::quiet_NaN(), 1, 1, 10); },
             "TH019-PROJECTION");
    rejected([] { (void)graphics::lookAt({0, 0, 0}, {0, 0, 0}, {0, 1, 0}); },
             "TH019-CAMERA-DIRECTION");
    rejected([] { (void)graphics::lookAt({0, 0, 1}, {0, 0, 0}, {0, 0, 1}); },
             "TH019-CAMERA-UP");
    rejected([] { (void)graphics::orthographic(1, 1, -1, 1, 1, 10); },
             "TH019-PROJECTION");
}

void geometryAndSafetyTests() {
    const auto positions = f32({3, 3}, {-.5f, -.5f, .5f, .5f, -.5f, .5f, 0, .5f, .5f});
    const auto indices = i64({1, 3}, {0, 1, 2});
    const auto colors = solidColors(3);
    const auto rendered = identityRender(positions, indices, colors);
    require(rendered.statistics.inputTriangleCount == 1 &&
            rendered.statistics.postClipTriangleCount == 1 &&
            rendered.statistics.depthPassCount != 0,
            "valid triangle did not render");

    rejected([&] { (void)identityRender(positions, i64({1, 3}, {-1, 1, 2}), colors); }, "TH019-INDEX");
    rejected([&] { (void)identityRender(positions, i64({1, 3}, {0, 1, 3}), colors); }, "TH019-INDEX");
    rejected([&] { (void)identityRender(i64({3, 3}, {0,0,0, 0,0,0, 0,0,0}),
                         indices, colors); }, "TH019-GEOMETRY-DTYPE");
    rejected([&] { (void)identityRender(positions, f32({1, 3}, {0,1,2}), colors); },
             "TH019-GEOMETRY-DTYPE");
    rejected([&] { (void)identityRender(f32({3, 2}, {0,0, 0,0, 0,0}), indices, colors); },
             "TH019-GEOMETRY-SHAPE");
    rejected([&] { (void)identityRender(positions, i64({3}, {0,1,2}), colors); },
             "TH019-GEOMETRY-SHAPE");
    rejected([&] { (void)identityRender(positions, indices, f32({3, 3}, std::vector<float>(9))); },
             "TH019-GEOMETRY-SHAPE");
    rejected([&] { (void)identityRender(positions, indices, solidColors(2)); },
             "TH019-GEOMETRY-COUNT");
    rejected([&] { (void)identityRender(f32({3, 3}, {0,0,.5f, 1,0,.5f,
                         std::numeric_limits<float>::quiet_NaN(),1,.5f}), indices, colors); },
             "TH019-NONFINITE-VERTEX");
    rejected([&] { (void)identityRender(f32({3, 3}, {0,0,.5f, 1,0,.5f,
                         std::numeric_limits<float>::infinity(),1,.5f}), indices, colors); },
             "TH019-NONFINITE-VERTEX");
    rejected([&] { (void)identityRender(positions, indices,
                         f32({3, 4}, {2,0,0,1, 1,1,1,1, 1,1,1,1})); },
             "TH019-COLOR-RANGE");

    const auto repeated = identityRender(positions, i64({1, 3}, {0, 0, 0}), colors);
    require(repeated.statistics.depthPassCount == 0, "repeated-index triangle was not degenerate");
    const auto collinear = identityRender(
        f32({3, 3}, {-.5f, 0, .5f, 0, 0, .5f, .5f, 0, .5f}), indices, colors);
    require(collinear.statistics.depthPassCount == 0, "collinear triangle was not rejected");

    auto zeroMatrix = f32({4, 4}, std::vector<float>(16));
    const auto unit = graphics::identity();
    const auto zeroW = graphics::render(positions, indices, colors, unit, unit, zeroMatrix, 16, 16);
    require(zeroW.statistics.depthPassCount == 0, "w=0 reached perspective division");
    rejected([&] { (void)graphics::render(positions, indices, colors, unit, unit,
                         f32({3, 3}, std::vector<float>(9)), 16, 16); }, "TH019-MATRIX-SHAPE");
    rejected([&] { (void)graphics::render(positions, indices, colors, unit, unit,
                         i64({4, 4}, std::vector<std::int64_t>(16)), 16, 16); },
             "TH019-MATRIX-DTYPE");
    rejected([&] { (void)graphics::render(positions, indices, colors, unit, unit, unit,
                         16, 16, {0, 0, 0, std::numeric_limits<float>::quiet_NaN()}); },
             "TH019-CLEAR-COLOR");

    const auto empty = identityRender(positions, indices, colors, 0, 0);
    require(empty.framebuffer.color().descriptor().shape == std::vector<std::uint64_t>({0, 4}) &&
            empty.framebuffer.depth().descriptor().shape == std::vector<std::uint64_t>({0}) &&
            empty.statistics.depthPassCount == 0,
            "zero framebuffer semantics are incorrect");
    rejected([&] { (void)identityRender(positions, indices, colors,
                         std::numeric_limits<std::uint64_t>::max(), 2); },
             "TH019-FRAMEBUFFER-DIMENSION");
    rejected([&] { (void)identityRender(positions, indices, colors,
                         static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()),
                         static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())); },
             "TH007-SIZE-OVERFLOW");
    rejected([&] { (void)rendered.framebuffer.pixelDepth(16, 0); }, "TH019-FRAMEBUFFER-OFFSET");

    const auto noGeometry = identityRender(f32({0, 3}, {}), i64({0, 3}, {}),
                                           f32({0, 4}, {}), 4, 4);
    require(noGeometry.statistics.inputTriangleCount == 0 &&
            noGeometry.statistics.depthPassCount == 0 &&
            finiteDepthPixels(noGeometry.framebuffer) == 0,
            "empty geometry did not preserve a clear framebuffer");

    const float huge = std::numeric_limits<float>::max() / 4.0f;
    try {
        const auto result = identityRender(
            f32({3, 3}, {-huge, -huge, .5f, huge, -huge, .5f, 0, huge, .5f}),
            indices, colors);
        require(result.statistics.depthPassCount <= 256, "huge geometry escaped framebuffer bounds");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "TH019-CLIP-NUMERIC" ||
                std::string(error.what()) == "TH019-NONFINITE-TRANSFORMED",
                "huge geometry used an undocumented rejection");
    }
}

void clippingTests() {
    const auto inside = clippingCase({-.5f,-.5f,.5f, .5f,-.5f,.5f, 0,.5f,.5f});
    require(inside.statistics.postClipTriangleCount == 1, "inside triangle clipping changed topology");

    const std::vector<std::vector<float>> outside = {
        {-3,-.5f,.5f, -2,-.5f,.5f, -2.5f,.5f,.5f},
        {2,-.5f,.5f, 3,-.5f,.5f, 2.5f,.5f,.5f},
        {-.5f,-3,.5f, .5f,-3,.5f, 0,-2,.5f},
        {-.5f,2,.5f, .5f,2,.5f, 0,3,.5f},
        {-.5f,-.5f,-2, .5f,-.5f,-2, 0,.5f,-2},
        {-.5f,-.5f,2, .5f,-.5f,2, 0,.5f,2}
    };
    for (const auto& triangle : outside)
        require(clippingCase(triangle).statistics.postClipTriangleCount == 0,
                "fully outside frustum plane was not rejected");

    const auto oneOutside = clippingCase({-2,0,.5f, .5f,-.6f,.5f, .5f,.6f,.5f});
    require(oneOutside.statistics.postClipTriangleCount == 2 &&
            oneOutside.statistics.depthPassCount != 0,
            "one-outside triangle did not become a quad/fan");
    const auto twoOutside = clippingCase({-2,-2,.5f, -2,2,.5f, .5f,0,.5f});
    require(twoOutside.statistics.postClipTriangleCount >= 1 &&
            twoOutside.statistics.depthPassCount != 0,
            "two-outside triangle was not clipped");

    for (const auto& crossing : std::vector<std::vector<float>>{
        {-.5f,-.5f,-.5f, .5f,-.5f,.5f, 0,.5f,.5f},
        {-.5f,-.5f,1.5f, .5f,-.5f,.5f, 0,.5f,.5f},
        {-1.5f,0,.5f, .5f,-.5f,.5f, .5f,.5f,.5f},
        {1.5f,0,.5f, -.5f,-.5f,.5f, -.5f,.5f,.5f},
        {0,-1.5f,.5f, -.5f,.5f,.5f, .5f,.5f,.5f},
        {0,1.5f,.5f, -.5f,-.5f,.5f, .5f,-.5f,.5f}}) {
        const auto result = clippingCase(crossing);
        require(result.statistics.postClipTriangleCount != 0 && result.statistics.depthPassCount != 0,
                "frustum crossing lost visible polygon");
    }

    const auto offscreen = clippingCase({-100,-100,.5f, 100,-100,.5f, 0,100,.5f});
    require(offscreen.statistics.depthPassCount != 0 && finiteDepthPixels(offscreen.framebuffer) <= 256,
            "offscreen clipping did not bound raster traversal");
    const auto clippedDegenerate = clippingCase({-2,0,.5f, -2,0,.5f, -1,0,.5f});
    require(clippedDegenerate.statistics.depthPassCount == 0,
            "degenerate clipped result generated fragments");
}

void rasterAndDepthTests() {
    const auto triangle = identityRender(
        f32({3, 3}, {-.5f,-.5f,.5f, .5f,-.5f,.5f, -.5f,.5f,.5f}),
        i64({1, 3}, {0,1,2}), solidColors(3), 8, 8);
    require(triangle.statistics.depthPassCount == 6,
            "known 8x8 triangle coverage changed: " +
            std::to_string(triangle.statistics.depthPassCount));
    require(std::isfinite(triangle.framebuffer.pixelDepth(2, 3)) &&
            !std::isfinite(triangle.framebuffer.pixelDepth(7, 7)),
            "viewport/pixel-center mapping changed");

    const auto quadPositions = f32({4, 3},
        {-.5f,-.5f,.5f, .5f,-.5f,.5f, .5f,.5f,.5f, -.5f,.5f,.5f});
    const auto quad = identityRender(quadPositions, i64({2, 3}, {0,1,2, 0,2,3}),
                                     solidColors(4), 8, 8);
    require(quad.statistics.shadedPixelCount == 16 && quad.statistics.depthPassCount == 16 &&
            finiteDepthPixels(quad.framebuffer) == 16,
            "top-left shared-edge quad has a crack or double coverage");

    const auto positions = f32({6, 3},
        {-.7f,-.7f,.2f, .7f,-.7f,.2f, 0,.7f,.2f,
         -.7f,-.7f,.8f, .7f,-.7f,.8f, 0,.7f,.8f});
    const auto colors = f32({6, 4},
        {1,0,0,1, 1,0,0,1, 1,0,0,1,
         0,0,1,1, 0,0,1,1, 0,0,1,1});
    const auto nearThenFar = identityRender(positions, i64({2, 3}, {0,1,2, 3,4,5}), colors, 24, 24);
    const auto farThenNear = identityRender(positions, i64({2, 3}, {3,4,5, 0,1,2}), colors, 24, 24);
    require(nearThenFar.framebuffer.colorDigest() == farThenNear.framebuffer.colorDigest() &&
            nearThenFar.framebuffer.depthDigest() == farThenNear.framebuffer.depthDigest(),
            "distinct-depth visibility depends on submission order");
    const auto center = nearThenFar.framebuffer.pixelColor(12, 12);
    close(center[0], 1); close(center[2], 0);
    require(nearThenFar.statistics.depthPassCount < farThenNear.statistics.depthPassCount,
            "depth-pass evidence did not distinguish ordering while preserving output");

    const auto equalPositions = f32({6, 3},
        {-.7f,-.7f,.4f, .7f,-.7f,.4f, 0,.7f,.4f,
         -.7f,-.7f,.4f, .7f,-.7f,.4f, 0,.7f,.4f});
    const auto equal = identityRender(equalPositions, i64({2, 3}, {0,1,2, 3,4,5}), colors, 24, 24);
    close(equal.framebuffer.pixelColor(12, 12)[0], 1);
    require(equal.statistics.shadedPixelCount > equal.statistics.depthPassCount,
            "strict less equal-depth policy was not applied");

    const auto crossingPositions = f32({6, 3},
        {-.8f,-.8f,.1f, .8f,-.8f,.9f, 0,.8f,.5f,
         -.8f,-.8f,.9f, .8f,-.8f,.1f, 0,.8f,.5f});
    const auto crossing = identityRender(crossingPositions,
        i64({2, 3}, {0,1,2, 3,4,5}), colors, 32, 32);
    require(crossing.statistics.shadedPixelCount > crossing.statistics.depthPassCount &&
            finiteDepthPixels(crossing.framebuffer) != 0,
            "crossing-triangle depth test was not exercised");
}

void interpolationTests() {
    const auto constant = identityRender(
        f32({3, 3}, {-.7f,-.7f,.5f, .7f,-.7f,.5f, 0,.7f,.5f}),
        i64({1, 3}, {0,1,2}), solidColors(3, {.25f,.5f,.75f,1}), 32, 32);
    const auto constantSample = constant.framebuffer.pixelColor(16, 16);
    close(constantSample[0], .25f); close(constantSample[1], .5f); close(constantSample[2], .75f);

    const auto barycentric = identityRender(
        f32({3, 3}, {-1,-1,.5f, 1,-1,.5f, -1,1,.5f}),
        i64({1, 3}, {0,1,2}),
        f32({3, 4}, {1,0,0,1, 0,1,0,1, 0,0,1,1}), 4, 4);
    const auto barycentricSample = barycentric.framebuffer.pixelColor(0, 3);
    close(barycentricSample[0], .75f); close(barycentricSample[1], .125f);
    close(barycentricSample[2], .125f); close(barycentricSample[3], 1);

    constexpr float halfPi = 1.57079632679489661923f;
    const auto positions = f32({3, 3}, {-1,-1,-2, 1,-1,-2, 0,1,-8});
    const auto colors = f32({3, 4}, {1,0,0,1, 0,1,0,1, 0,0,1,1});
    const auto unit = graphics::identity();
    const auto perspective = graphics::render(positions, i64({1, 3}, {0,1,2}), colors,
        unit, unit, graphics::perspective(halfPi, 1, 1, 10), 64, 64);
    const auto sample = perspective.framebuffer.pixelColor(32, 32);
    // Hand-derived at pixel center (32.5,32.5): affine barycentrics are
    // (0.096875, 0.128125, 0.775), then divide their (1/w)-weighted values
    // by 0.209375. The affine blue value 0.775 is intentionally different.
    close(sample[0], 0.23134328f, 2.0e-4f);
    close(sample[1], 0.30597015f, 2.0e-4f);
    close(sample[2], 0.46268657f, 2.0e-4f);
    require(std::fabs(sample[2] - .775f) > .2f, "color interpolation was affine");

    const auto clippedColors = f32({3, 4}, {1,0,0,1, 0,0,1,1, 0,0,1,1});
    const auto clipped = identityRender(
        f32({3, 3}, {-2,0,.5f, 0,-.8f,.5f, 0,.8f,.5f}),
        i64({1, 3}, {0,1,2}), clippedColors, 32, 32);
    const auto clippedSample = clipped.framebuffer.pixelColor(0, 16);
    require(clippedSample[0] > .35f && clippedSample[0] < .55f &&
            clippedSample[2] > .45f,
            "clipped-vertex attribute interpolation changed");
}

graphics::RenderResult renderCube() {
    const auto positions = f32({8, 3}, {
        -1,-1,-1,  1,-1,-1,  1,1,-1,  -1,1,-1,
        -1,-1, 1,  1,-1, 1,  1,1, 1,  -1,1, 1
    });
    const auto indices = i64({12, 3}, {
        0,2,1, 0,3,2, 4,5,6, 4,6,7,
        0,1,5, 0,5,4, 3,7,6, 3,6,2,
        0,4,7, 0,7,3, 1,2,6, 1,6,5
    });
    const auto colors = f32({8, 4}, {
        1,0,0,1, 0,1,0,1, 0,0,1,1, 1,1,0,1,
        1,0,1,1, 0,1,1,1, 1,1,1,1, .25f,.5f,.75f,1
    });
    const auto model = graphics::multiply(graphics::rotationY(.55f), graphics::rotationX(.25f));
    const auto view = graphics::lookAt({3,2,4}, {0,0,0}, {0,1,0});
    const auto projection = graphics::perspective(1.04719755119659774615f, 1, .1f, 100);
    return graphics::render(positions, indices, colors, model, view, projection, 96, 96);
}

void cubeAndOutputTests() {
    const auto cube = renderCube();
    const auto repeated = renderCube();
    require(cube.framebuffer.width() == 96 && cube.framebuffer.height() == 96,
            "cube framebuffer dimensions changed");
    require(cube.statistics.inputTriangleCount == 12 &&
            cube.statistics.postClipTriangleCount == 12 &&
            cube.statistics.shadedPixelCount == 3150 && cube.statistics.depthPassCount == 2218,
            "cube did not exercise the 3D pipeline");
    require(cube.framebuffer.colorDigest() == "fnv1a64:0d0c47343db9df53" &&
            cube.framebuffer.depthDigest() == "fnv1a64:4e244c6d47059a59",
            "cube reference digest changed");
    require(cube.framebuffer.colorDigest() == repeated.framebuffer.colorDigest() &&
            cube.framebuffer.depthDigest() == repeated.framebuffer.depthDigest() &&
            cube.statistics.shadedPixelCount == repeated.statistics.shadedPixelCount &&
            cube.statistics.depthPassCount == repeated.statistics.depthPassCount,
            "cube rerender is nondeterministic");

    const auto path = fs::path("/tmp") /
        ("th019-cube-" + std::to_string(static_cast<long long>(::getpid())) + ".ppm");
    graphics::writePpm(cube.framebuffer, path);
    std::ifstream input(path, std::ios::binary);
    std::string magic;
    std::getline(input, magic);
    require(magic == "P6" && fs::file_size(path) > 96 * 96 * 3,
            "deterministic PPM output is malformed");
    fs::remove(path);

    const auto empty = identityRender(f32({0, 3}, {}), i64({0, 3}, {}), f32({0, 4}, {}), 0, 0);
    rejected([&] { graphics::writePpm(empty.framebuffer, path); }, "TH019-PPM-DIMENSIONS");

    std::cout << "TH019_CUBE dimensions=" << cube.framebuffer.width() << 'x'
              << cube.framebuffer.height()
              << " input_triangles=" << cube.statistics.inputTriangleCount
              << " post_clip_triangles=" << cube.statistics.postClipTriangleCount
              << " shaded_pixels=" << cube.statistics.shadedPixelCount
              << " depth_passes=" << cube.statistics.depthPassCount
              << " color_digest=" << cube.framebuffer.colorDigest()
              << " depth_digest=" << cube.framebuffer.depthDigest() << '\n';
}

} // namespace

int main() {
    try {
        transformTests();
        geometryAndSafetyTests();
        clippingTests();
        rasterAndDepthTests();
        interpolationTests();
        cubeAndOutputTests();
        std::cout << "V0GraphicsTests PASS " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0GraphicsTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
