#include "graphics/v0/GraphicsGpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace graphics = thiran::v0::graphics;
namespace storage = thiran::v0::storage;

namespace {
int checks = 0;
void require(bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
storage::Tensor f32(std::vector<std::uint64_t> shape, std::vector<float> values) {
    return storage::Tensor::materializeF32(std::move(shape), values);
}
storage::Tensor i64(std::vector<std::uint64_t> shape, std::vector<std::int64_t> values) {
    return storage::Tensor::materializeI64(std::move(shape), values);
}

void equivalent(const graphics::RenderResult& cpu, const graphics::RenderResult& gpu,
                float tolerance = 2.0e-6f) {
    const auto cpuDepth = cpu.framebuffer.depth().logicalF32Values();
    const auto gpuDepth = gpu.framebuffer.depth().logicalF32Values();
    const auto cpuColor = cpu.framebuffer.color().logicalF32Values();
    const auto gpuColor = gpu.framebuffer.color().logicalF32Values();
    require(cpuDepth.size() == gpuDepth.size() && cpuColor.size() == gpuColor.size(),
            "CPU/GPU output shape mismatch");
    for (std::size_t index = 0; index < cpuDepth.size(); ++index) {
        require(std::isfinite(cpuDepth[index]) == std::isfinite(gpuDepth[index]),
                "CPU/GPU coverage mask mismatch");
        if (std::isfinite(cpuDepth[index]))
            require(std::fabs(cpuDepth[index] - gpuDepth[index]) <= tolerance,
                    "CPU/GPU depth mismatch");
    }
    for (std::size_t index = 0; index < cpuColor.size(); ++index)
        require(std::fabs(cpuColor[index] - gpuColor[index]) <= tolerance,
                "CPU/GPU color mismatch");
}

storage::Tensor solid(std::uint64_t count, std::array<float,4> color = {1,1,1,1}) {
    std::vector<float> values;
    for (std::uint64_t index = 0; index < count; ++index)
        values.insert(values.end(), color.begin(), color.end());
    return f32({count,4}, std::move(values));
}

thiran::v0::backend::TensorRegion negateRegion(std::uint64_t rows) {
    namespace backend = thiran::v0::backend;
    namespace semantic = thiran::v0::semantic;
    backend::TensorRegion region;
    region.function = 1;
    region.name = "graphics_color_negate";
    const auto type = semantic::tensor(semantic::scalar(semantic::TypeKind::F32), 2);
    backend::RegionNode input;
    input.id = 1; input.op = backend::RegionOp::Input; input.type = type;
    input.shape = semantic::ShapeFact{{static_cast<std::int64_t>(rows), std::int64_t{4}}};
    input.provenance = thiran::v0::analysis::ProvenanceKind::Fresh;
    backend::RegionNode negate = input;
    negate.id = 2; negate.op = backend::RegionOp::Negate; negate.dependencies = {1};
    region.nodes = {input, negate};
    region.inputs = {1};
    region.output = 2;
    region.outputType = type;
    return region;
}
}

int main() {
    try {
        const auto probe = thiran::v0::backend::probeNativeGpu();
        if (!probe.deviceAvailable) {
            std::cerr << "physical CUDA GPU unavailable\n";
            return 77;
        }
        const auto positions = f32({3,3}, {-.5f,-.5f,.5f, .5f,-.5f,.5f, -.5f,.5f,.5f});
        const auto indices = i64({1,3}, {0,1,2});
        const auto colors = f32({3,4}, {1,0,0,1, 0,1,0,1, 0,0,1,1});
        const auto unit = graphics::identity();
        const auto cpu = graphics::render(positions, indices, colors, unit, unit, unit, 8, 8);
        auto gpu = graphics::renderGpu(positions, indices, colors, unit, unit, unit, 8, 8);
        require(gpu.ok(), gpu.error ? gpu.error->code + ": " + gpu.error->message : "GPU render failed");
        std::cout << "CPU color=" << cpu.framebuffer.colorDigest()
                  << " depth=" << cpu.framebuffer.depthDigest()
                  << " shaded=" << cpu.statistics.shadedPixelCount
                  << " passes=" << cpu.statistics.depthPassCount << '\n'
                  << "GPU color=" << gpu.frame->framebuffer.colorDigest()
                  << " depth=" << gpu.frame->framebuffer.depthDigest()
                  << " shaded=" << gpu.frame->statistics.shadedPixelCount
                  << " passes=" << gpu.frame->statistics.depthPassCount << '\n';
        require(gpu.frame->framebuffer.colorDigest() == cpu.framebuffer.colorDigest(), "color mismatch");
        require(gpu.frame->framebuffer.depthDigest() == cpu.framebuffer.depthDigest(), "depth mismatch");
        require(gpu.frame->statistics.shadedPixelCount == cpu.statistics.shadedPixelCount,
                "shaded count mismatch");
        require(gpu.evidence.execution.kernelLaunches == 1, "missing physical launch");
        equivalent(cpu, *gpu.frame);

        const auto quadPositions = f32({4,3},
            {-.5f,-.5f,.5f, .5f,-.5f,.5f, .5f,.5f,.5f, -.5f,.5f,.5f});
        const auto quadIndices = i64({2,3}, {0,1,2, 0,2,3});
        const auto quadCpu = graphics::render(quadPositions, quadIndices, solid(4),
                                              unit, unit, unit, 13, 9);
        const auto quadGpu = graphics::renderGpu(quadPositions, quadIndices, solid(4),
                                                 unit, unit, unit, 13, 9);
        require(quadGpu.ok(), quadGpu.error ? quadGpu.error->message : "GPU quad failed");
        equivalent(quadCpu, *quadGpu.frame);

        const auto overlapPositions = f32({6,3},
            {-.7f,-.7f,.2f, .7f,-.7f,.2f, 0,.7f,.2f,
             -.7f,-.7f,.8f, .7f,-.7f,.8f, 0,.7f,.8f});
        const auto overlapColors = f32({6,4},
            {1,0,0,1, 1,0,0,1, 1,0,0,1,
             0,0,1,1, 0,0,1,1, 0,0,1,1});
        const auto nearFar = i64({2,3}, {0,1,2, 3,4,5});
        const auto farNear = i64({2,3}, {3,4,5, 0,1,2});
        const auto cpuNearFar = graphics::render(overlapPositions, nearFar, overlapColors,
                                                 unit, unit, unit, 31, 27);
        const auto cpuFarNear = graphics::render(overlapPositions, farNear, overlapColors,
                                                 unit, unit, unit, 31, 27);
        const auto gpuNearFar = graphics::renderGpu(overlapPositions, nearFar, overlapColors,
                                                    unit, unit, unit, 31, 27);
        const auto gpuFarNear = graphics::renderGpu(overlapPositions, farNear, overlapColors,
                                                    unit, unit, unit, 31, 27);
        require(gpuNearFar.ok() && gpuFarNear.ok(), "GPU overlap render failed");
        equivalent(cpuNearFar, *gpuNearFar.frame);
        equivalent(cpuFarNear, *gpuFarNear.frame);
        require(gpuNearFar.frame->framebuffer.colorDigest() ==
                    gpuFarNear.frame->framebuffer.colorDigest() &&
                gpuNearFar.frame->framebuffer.depthDigest() ==
                    gpuFarNear.frame->framebuffer.depthDigest() &&
                cpuNearFar.framebuffer.colorDigest() == cpuFarNear.framebuffer.colorDigest(),
                "distinct-depth order independence failed");

        const auto equalPositions = f32({6,3},
            {-.7f,-.7f,.4f, .7f,-.7f,.4f, 0,.7f,.4f,
             -.7f,-.7f,.4f, .7f,-.7f,.4f, 0,.7f,.4f});
        const auto equalCpu = graphics::render(equalPositions, nearFar, overlapColors,
                                               unit, unit, unit, 24, 24);
        const auto equalGpu = graphics::renderGpu(equalPositions, nearFar, overlapColors,
                                                  unit, unit, unit, 24, 24);
        require(equalGpu.ok(), "GPU equal-depth render failed");
        equivalent(equalCpu, *equalGpu.frame);
        require(equalGpu.frame->framebuffer.pixelColor(12,12)[0] == 1.0f,
                "equal-depth first-wins failed");

        const auto clippedPositions = f32({3,3},
            {-.5f,-.5f,-.5f, .5f,-.5f,.5f, 0,.5f,.5f});
        const auto oneTriangle = i64({1,3}, {0,1,2});
        const auto clippedCpu = graphics::render(clippedPositions, oneTriangle, solid(3),
                                                 unit, unit, unit, 37, 29);
        const auto clippedGpu = graphics::renderGpu(clippedPositions, oneTriangle, solid(3),
                                                    unit, unit, unit, 37, 29);
        require(clippedGpu.ok(), "GPU clipped render failed");
        equivalent(clippedCpu, *clippedGpu.frame);

        const auto perspectivePositions = f32({3,3}, {-1,-1,-2, 1,-1,-2, 0,1,-8});
        const auto varyingColors = f32({3,4}, {1,0,0,1, 0,1,0,1, 0,0,1,1});
        const auto perspectiveProjection = graphics::perspective(1.57079632679489661923f, 1, 1, 10);
        const auto perspectiveCpu = graphics::render(perspectivePositions, oneTriangle, varyingColors,
            unit, unit, perspectiveProjection, 64, 64);
        const auto perspectiveGpu = graphics::renderGpu(perspectivePositions, oneTriangle, varyingColors,
            unit, unit, perspectiveProjection, 64, 64);
        require(perspectiveGpu.ok(), "GPU perspective render failed");
        equivalent(perspectiveCpu, *perspectiveGpu.frame);
        require(std::fabs(perspectiveGpu.frame->framebuffer.pixelColor(32,32)[2] - .46268657f) < 2e-4f,
                "GPU interpolation was not perspective-correct");

        const auto cubePositions = f32({8, 3}, {
            -1,-1,-1,  1,-1,-1,  1,1,-1,  -1,1,-1,
            -1,-1, 1,  1,-1, 1,  1,1, 1,  -1,1, 1});
        const auto cubeIndices = i64({12, 3}, {
            0,2,1, 0,3,2, 4,5,6, 4,6,7, 0,1,5, 0,5,4,
            3,7,6, 3,6,2, 0,4,7, 0,7,3, 1,2,6, 1,6,5});
        const auto cubeColors = f32({8, 4}, {
            1,0,0,1, 0,1,0,1, 0,0,1,1, 1,1,0,1,
            1,0,1,1, 0,1,1,1, 1,1,1,1, .25f,.5f,.75f,1});
        const auto model = graphics::multiply(graphics::rotationY(.55f), graphics::rotationX(.25f));
        const auto view = graphics::lookAt({3,2,4}, {0,0,0}, {0,1,0});
        const auto projection = graphics::perspective(1.04719755119659774615f, 1, .1f, 100);
        const auto cpuCube = graphics::render(cubePositions, cubeIndices, cubeColors,
                                              model, view, projection, 96, 96);
        const auto gpuCube = graphics::renderGpu(cubePositions, cubeIndices, cubeColors,
                                                 model, view, projection, 96, 96);
        require(gpuCube.ok(), gpuCube.error ? gpuCube.error->message : "GPU cube failed");
        equivalent(cpuCube, *gpuCube.frame);
        require(gpuCube.frame->framebuffer.colorDigest() == cpuCube.framebuffer.colorDigest() &&
                gpuCube.frame->framebuffer.depthDigest() == cpuCube.framebuffer.depthDigest(),
                "cube digests differ");
        std::cout << "CPU cube color=" << cpuCube.framebuffer.colorDigest()
                  << " depth=" << cpuCube.framebuffer.depthDigest()
                  << " shaded=" << cpuCube.statistics.shadedPixelCount
                  << " passes=" << cpuCube.statistics.depthPassCount << '\n'
                  << "GPU cube color=" << gpuCube.frame->framebuffer.colorDigest()
                  << " depth=" << gpuCube.frame->framebuffer.depthDigest()
                  << " shaded=" << gpuCube.frame->statistics.shadedPixelCount
                  << " passes=" << gpuCube.frame->statistics.depthPassCount << '\n';

        const auto repeatedCube = graphics::renderGpu(cubePositions, cubeIndices, cubeColors,
                                                       model, view, projection, 96, 96);
        require(repeatedCube.ok() &&
                repeatedCube.frame->framebuffer.colorDigest() == gpuCube.frame->framebuffer.colorDigest() &&
                repeatedCube.frame->framebuffer.depthDigest() == gpuCube.frame->framebuffer.depthDigest(),
                "repeated GPU render was not byte deterministic");

        auto pending = graphics::submitRenderGpuAsync(
            positions, indices, colors, unit, unit, unit, 8, 8);
        require(pending.ok() && !pending.pending->value().has_value() &&
                pending.pending->outputResource().activeWrites() == 1 &&
                positions.asyncResource().activeReads() == 1 &&
                pending.evidence.execution.activeReadReservations == 4,
                "pending GPU frame became observable or lost reservations");
        const auto observed = pending.pending->observe();
        require(observed.ok() && positions.asyncResource().activeReads() == 0 &&
                pending.pending->outputResource().activeWrites() == 0 &&
                pending.pending->observe().ok(),
                "GPU observation/repeated observation contract failed");

        auto pendingA = graphics::submitRenderGpuAsync(
            positions, indices, colors, unit, unit, unit, 8, 8);
        auto pendingB = graphics::submitRenderGpuAsync(
            positions, indices, colors, unit, unit, unit, 8, 8);
        require(pendingA.ok() && pendingB.ok() && positions.asyncResource().activeReads() == 2,
                "independent pending renders were not both live");
        require(pendingA.pending->observe().ok() && positions.asyncResource().activeReads() == 1 &&
                pendingB.pending->outputResource().activeWrites() == 1 &&
                pendingB.pending->evidence().execution.synchronizations == 0,
                "observing A released B resources");
        require(pendingB.pending->observe().ok() && positions.asyncResource().activeReads() == 0,
                "pending render B did not observe independently");

        const auto beforeDrop = thiran::v0::runtime::asyncRuntimeEvidence();
        {
            auto dropped = graphics::submitRenderGpuAsync(
                positions, indices, colors, unit, unit, unit, 8, 8);
            require(dropped.ok(), "dropped-render submission failed");
        }
        const auto afterDrop = thiran::v0::runtime::asyncRuntimeEvidence();
        require(afterDrop.droppedDrains == beforeDrop.droppedDrains + 1 &&
                positions.asyncResource().activeReads() == 0,
                "dropped GPU render did not safely drain");

        const auto zero = graphics::renderGpu(positions, indices, colors,
                                              unit, unit, unit, 0, 0);
        require(zero.ok() && zero.evidence.execution.kernelLaunches == 0 &&
                zero.frame->framebuffer.color().descriptor().shape ==
                    std::vector<std::uint64_t>({0,4}) &&
                zero.frame->framebuffer.depth().descriptor().shape ==
                    std::vector<std::uint64_t>({0}),
                "zero-size GPU frame semantics failed");

        const auto invalidDevice = graphics::renderGpu(
            positions, indices, colors, unit, unit, unit, 8, 8, {0,0,0,0},
            probe.deviceCount);
        require(!invalidDevice.ok() && invalidDevice.error &&
                invalidDevice.error->category == thiran::v0::backend::GpuErrorCategory::InvalidDevice &&
                invalidDevice.evidence.execution.kernelLaunches == 0,
                "invalid GPU device was not rejected");

        const auto colorTensor = gpuCube.frame->framebuffer.color();
        require(colorTensor.descriptor().dtype == storage::DType::F32 &&
                colorTensor.descriptor().shape == std::vector<std::uint64_t>({96*96,4}) &&
                gpuCube.frame->framebuffer.depth().descriptor().shape ==
                    std::vector<std::uint64_t>({96*96}) &&
                colorTensor.storageId() != gpuCube.frame->framebuffer.depth().storageId() &&
                colorTensor.storageId() != cubePositions.storageId(),
                "observed GPU frame is not independent ordinary Tensor storage");
        const auto numerical = thiran::v0::backend::executeNativeGpu(
            negateRegion(96*96), {colorTensor});
        require(numerical.ok(), numerical.error ? numerical.error->message :
                "rendered Tensor was rejected by native numerical GPU execution");
        const auto& negated = std::get<storage::Tensor>(*numerical.value);
        require(negated.descriptor().shape == colorTensor.descriptor().shape &&
                std::fabs(negated.loadF32({0,0}) + colorTensor.loadF32({0,0})) <= 1e-6f &&
                numerical.evidence.hostToDeviceCopies >= 1 &&
                numerical.evidence.deviceToHostCopies >= 1,
                "graphics Tensor numerical interoperability failed");

        thiran::v0::backend::NativeGpuKernelRequest malformed;
        malformed.ptx = ".version 6.0\n.target sm_50\n.address_size 64\nBROKEN";
        malformed.entry = "broken";
        malformed.workItems = 1;
        malformed.outputByteCounts = {4};
        malformed.retainedReadResources = {positions.asyncResource()};
        const auto malformedResult = thiran::v0::backend::submitNativeGpuKernelAsync(
            std::move(malformed));
        require(!malformedResult.ok() && malformedResult.error &&
                malformedResult.error->category ==
                    thiran::v0::backend::GpuErrorCategory::BackendUnsupported,
                "malformed graphics PTX was not rejected safely");

        std::cout << "TH020_GPU device=" << probe.name
                  << " driver=" << probe.driverVersion
                  << " capability=" << probe.computeMajor << '.' << probe.computeMinor
                  << " color=" << gpu.frame->framebuffer.colorDigest()
                  << " depth=" << gpu.frame->framebuffer.depthDigest()
                  << " cube_color=" << gpuCube.frame->framebuffer.colorDigest()
                  << " cube_depth=" << gpuCube.frame->framebuffer.depthDigest()
                  << " launches=" << gpuCube.evidence.execution.kernelLaunches
                  << " h2d=" << gpuCube.evidence.execution.hostToDeviceCopies
                  << " d2h=" << gpuCube.evidence.execution.deviceToHostCopies
                  << " observations=" << gpuCube.evidence.execution.observations << '\n';
        std::cout << "V0GraphicsGpuIntegrationTests PASS " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0GraphicsGpuIntegrationTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
