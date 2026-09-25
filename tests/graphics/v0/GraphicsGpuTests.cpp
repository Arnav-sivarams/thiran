#include "graphics/v0/GraphicsGpu.hpp"

#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace backend = thiran::v0::backend;
namespace graphics = thiran::v0::graphics;
namespace storage = thiran::v0::storage;

namespace {
int checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
storage::Tensor f32(std::vector<std::uint64_t> shape, std::vector<float> values) {
    return storage::Tensor::materializeF32(std::move(shape), values);
}
storage::Tensor i64(std::vector<std::uint64_t> shape, std::vector<std::int64_t> values) {
    return storage::Tensor::materializeI64(std::move(shape), values);
}
}

int main() {
    try {
        const auto positions = f32({3,3}, {-.5f,-.5f,.5f, .5f,-.5f,.5f, 0,.5f,.5f});
        const auto indices = i64({1,3}, {0,1,2});
        const auto colors = f32({3,4}, {1,0,0,1, 1,1,1,1, 0,0,1,1});
        const auto unit = graphics::identity();
        const auto cpu = graphics::render(positions, indices, colors, unit, unit, unit, 8, 8);
        require(cpu.statistics.depthPassCount != 0, "explicit CPU selection failed");

        const auto checkValidation = [&](const storage::Tensor& p, const storage::Tensor& i,
                                         const storage::Tensor& c, const storage::Tensor& transform,
                                         std::uint64_t width, std::uint64_t height) {
            const auto submitted = graphics::submitRenderGpuAsync(
                p, i, c, transform, unit, unit, width, height);
            require(!submitted.ok() && submitted.error &&
                    submitted.error->category == backend::GpuErrorCategory::ValidationFailure &&
                    submitted.evidence.execution.kernelLaunches == 0,
                    "malformed graphics request was not rejected before GPU launch");
        };
        checkValidation(i64({3,3}, {0,0,0, 0,0,0, 0,0,0}), indices, colors, unit, 8, 8);
        checkValidation(f32({3,2}, {0,0, 0,0, 0,0}), indices, colors, unit, 8, 8);
        checkValidation(positions, i64({1,3}, {-1,1,2}), colors, unit, 8, 8);
        checkValidation(positions, i64({1,3}, {0,1,3}), colors, unit, 8, 8);
        checkValidation(f32({3,3}, {0,0,.5f, 1,0,.5f,
            std::numeric_limits<float>::quiet_NaN(),1,.5f}), indices, colors, unit, 8, 8);
        checkValidation(positions, indices, f32({3,4}, {2,0,0,1, 1,1,1,1, 1,1,1,1}),
                        unit, 8, 8);
        checkValidation(positions, indices, colors, f32({4,4}, {
            1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,
            std::numeric_limits<float>::infinity()}), 8, 8);
        checkValidation(positions, indices, colors, unit,
                        std::numeric_limits<std::uint64_t>::max(), 2);

        if (!backend::nativeGpuBackendBuilt()) {
            const auto unavailable = graphics::renderGpu(
                positions, indices, colors, unit, unit, unit, 8, 8);
            require(!unavailable.ok() && unavailable.error &&
                    unavailable.error->category == backend::GpuErrorCategory::BackendUnavailable &&
                    unavailable.error->code == "GPU-BACKEND-NOT-BUILT" &&
                    unavailable.evidence.execution.kernelLaunches == 0,
                    "CPU-only GPU request did not report deterministic unavailability");
            require(graphics::emitGraphicsGpuPtx().empty(), "CPU-only build exposed graphics PTX");
        } else {
            const auto ptx = graphics::emitGraphicsGpuPtx();
            require(ptx.find("thiran_graphics_raster") != std::string::npos &&
                    ptx.find(".f64") != std::string::npos &&
                    ptx.find(".approx") == std::string::npos &&
                    ptx.find(".ftz") == std::string::npos,
                    "graphics PTX audit failed");
        }
        std::cout << "V0GraphicsGpuTests PASS " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0GraphicsGpuTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
