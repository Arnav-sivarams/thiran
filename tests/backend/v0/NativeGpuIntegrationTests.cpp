#include "backend/v0/NativeGpu.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace thiran::v0;
namespace analysis = thiran::v0::analysis;
namespace backend = thiran::v0::backend;
namespace semantic = thiran::v0::semantic;
namespace storage = thiran::v0::storage;

namespace {
int checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
semantic::Module compile(const std::string& source) {
    auto parsed = parse(source, "<th013-gpu-integration>");
    if (!parsed.module) throw std::runtime_error(parsed.diagnostics.front().format());
    auto analyzed = semantic::analyze(*parsed.module);
    if (!analyzed.module) throw std::runtime_error(analyzed.diagnostics.front().format());
    if (!semantic::verify(*analyzed.module).ok) throw std::runtime_error("semantic verifier failed");
    return *analyzed.module;
}
backend::TensorRegion region(const semantic::Module& module, const std::string& name = "main",
                             bool standalone = true) {
    auto facts = analysis::analyze(module);
    if (!facts.ok()) throw std::runtime_error("ownership analysis failed");
    auto lowered = backend::extractStrictGpu(module, facts, name, standalone);
    if (!lowered.ok()) throw std::runtime_error(lowered.diagnostic);
    return *lowered.region;
}
void same(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "f32 result size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i)
        require(std::fabs(actual[i] - expected[i]) <= 1e-6f, "f32 numerical mismatch");
}
}

int main() {
    try {
        const auto probe = backend::probeNativeGpu(0);
        if (!probe.deviceAvailable) {
            std::cout << "V0NativeGpuIntegrationTests SKIP/UNAVAILABLE "
                      << (probe.error ? probe.error->code + ": " + probe.error->message : "unknown") << '\n';
            return 77;
        }

        const auto floatModule = compile(
            "fn f(A:Tensor<f32,1>,B:Tensor<f32,1>) -> Tensor<f32,1> {\n"
            "let C=A+B\nlet D=C.*B\nreturn D-A\n}");
        const auto floatRegion = region(floatModule, "f", false);
        const auto floatA = storage::Tensor::materializeF32({5}, {1.0f, 2.0f, 0.0f, 3.5f, 7.0f});
        const auto floatB = storage::Tensor::materializeF32({5}, {2.0f, 4.0f, 3.0f, 0.5f, 1.0f});
        const auto floatRun = backend::executeNativeGpu(floatRegion, {floatA, floatB});
        require(floatRun.ok(), floatRun.error ? floatRun.error->message : "f32 GPU run failed");
        const auto& floatTensor = std::get<storage::Tensor>(*floatRun.value);
        semantic::RuntimeValue refA{semantic::RuntimeTensor{semantic::TypeKind::F32, {5}, {},
            {1.0f, 2.0f, 0.0f, 3.5f, 7.0f}}};
        semantic::RuntimeValue refB{semantic::RuntimeTensor{semantic::TypeKind::F32, {5}, {},
            {2.0f, 4.0f, 3.0f, 0.5f, 1.0f}}};
        const auto floatReference = semantic::evaluateCall(floatModule, "f", {refA, refB});
        require(floatReference.ok && floatReference.value, "f32 semantic reference failed");
        const auto& floatExpected = std::get<semantic::RuntimeTensor>(floatReference.value->data).f32Values;
        same(floatTensor.logicalF32Values(), floatExpected);
        require(floatRun.evidence.kernelLaunches == 3 && floatRun.evidence.synchronizations == 3,
                "multi-kernel launch/synchronization evidence mismatch");
        require(floatRun.evidence.hostToDeviceCopies == 2 && floatRun.evidence.deviceToHostCopies == 1,
                "literal transfer evidence mismatch");

        const auto dynamicModule = compile(
            "fn f(x: Tensor<i64,1>, y: Tensor<i64,1>) -> Tensor<i64,1> {\n"
            "let z=x+y\nreturn z.*y\n}");
        const auto dynamicRegion = region(dynamicModule, "f", false);
        const auto host = storage::Tensor::materializeI64({257}, std::vector<std::int64_t>(257, 3));
        const auto dynamicRun = backend::executeNativeGpu(dynamicRegion, {host, host});
        require(dynamicRun.ok(), dynamicRun.error ? dynamicRun.error->message : "dynamic GPU run failed");
        const auto dynamicValues = std::get<storage::Tensor>(*dynamicRun.value).logicalI64Values();
        require(dynamicValues.size() == 257 && dynamicValues.front() == 18 && dynamicValues.back() == 18,
                "odd/non-power-of-two i64 result mismatch");
        semantic::RuntimeValue refHost{semantic::RuntimeTensor{
            semantic::TypeKind::I64, {257}, std::vector<std::int64_t>(257, 3), {}}};
        const auto dynamicReference = semantic::evaluateCall(dynamicModule, "f", {refHost, refHost});
        require(dynamicReference.ok && dynamicReference.value &&
                std::get<semantic::RuntimeTensor>(dynamicReference.value->data).values == dynamicValues,
                "i64 GPU result differs from semantic reference");
        require(dynamicRun.evidence.hostToDeviceCopies == 1,
                "immutable aliases caused duplicate host-to-device storage");
        require(host.logicalI64Values().front() == 3 &&
                std::get<storage::Tensor>(*dynamicRun.value).storageId() != host.storageId(),
                "GPU execution introduced hidden copy-on-write or mutated input");

        const auto again = backend::executeNativeGpu(dynamicRegion, {host, host});
        require(again.ok() && std::get<storage::Tensor>(*again.value).logicalI64Values() == dynamicValues,
                "repeat GPU execution was not deterministic");

        const auto zero = storage::Tensor::empty(storage::DType::I64, {0});
        const auto zeroRun = backend::executeNativeGpu(dynamicRegion, {zero, zero});
        require(zeroRun.ok() && std::get<storage::Tensor>(*zeroRun.value).logicalI64Values().empty() &&
                zeroRun.evidence.kernelLaunches == 0,
                "zero-size GPU path launched a kernel or changed value");

        const auto pickModule = compile("fn pick(x: Tensor<i64,1>, i:i64)->i64{return x[i]}");
        const auto pickRegion = region(pickModule, "pick", false);
        const auto pick = backend::executeNativeGpu(pickRegion, {host, std::int64_t{256}});
        require(pick.ok() && std::get<std::int64_t>(*pick.value) == 3 && pick.evidence.kernelLaunches == 1,
                "GPU Index kernel result mismatch");
        const auto badPick = backend::executeNativeGpu(pickRegion, {host, std::int64_t{257}});
        require(!badPick.ok() && badPick.error && badPick.error->code == "TH-SPEC-BOUNDS" &&
                badPick.evidence.kernelLaunches == 0,
                "out-of-bounds GPU Index did not fail before launch");

        const auto overflowHost = storage::Tensor::materializeI64(
            {1}, {std::numeric_limits<std::int64_t>::max()});
        const auto one = storage::Tensor::materializeI64({1}, {1});
        const auto overflowRun = backend::executeNativeGpu(dynamicRegion, {overflowHost, one});
        require(!overflowRun.ok() && overflowRun.error && overflowRun.error->code == "TH-SPEC-I64-OVERFLOW",
                "GPU checked i64 overflow did not propagate");

        auto matrix = storage::Tensor::materializeI64({2, 2}, {1, 2, 3, 4});
        auto transposed = matrix.transpose();
        const auto matrixModule = compile(
            "fn f(x:Tensor<i64,2>,y:Tensor<i64,2>)->Tensor<i64,2>{return x+y}");
        const auto matrixRegion = region(matrixModule, "f", false);
        const auto layoutRun = backend::executeNativeGpu(matrixRegion, {transposed, transposed});
        require(!layoutRun.ok() && layoutRun.error && layoutRun.error->code == "GPU-UNSUPPORTED-LAYOUT" &&
                layoutRun.evidence.kernelLaunches == 0,
                "non-contiguous/view input was not rejected explicitly");

        const auto invalidDevice = backend::executeNativeGpu(floatRegion, {floatA, floatB}, probe.deviceCount);
        require(!invalidDevice.ok() && invalidDevice.error &&
                invalidDevice.error->category == backend::GpuErrorCategory::InvalidDevice,
                "invalid GPU device selection did not fail explicitly");

        std::cout << "V0NativeGpuIntegrationTests PASS " << checks << " checks\n"
                  << "device=" << probe.name << '\n'
                  << "driver_version=" << probe.driverVersion << '\n'
                  << "compute_capability=" << probe.computeMajor << '.' << probe.computeMinor << '\n'
                  << "workload=f32 add,multiply,subtract; i64 add,multiply,index\n"
                  << "kernels=" << floatRun.evidence.kernelLaunches
                  << " synchronizations=" << floatRun.evidence.synchronizations << '\n'
                  << "fallback=NONE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0NativeGpuIntegrationTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
