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
namespace runtime = thiran::v0::runtime;
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
template<class F>
void contract(F&& action, const std::string& code) {
    try { action(); }
    catch (const runtime::AsyncContractError& error) {
        require(error.code() == code, "expected " + code + ", got " + error.code());
        return;
    }
    throw std::runtime_error("expected async lifetime conflict " + code);
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
            "let C=A+B\nlet D=C.*B\nlet E=D-A\nreturn E+B\n}");
        const auto floatRegion = region(floatModule, "f", false);
        auto floatA = storage::Tensor::materializeF32({5}, {1.0f, 2.0f, 0.0f, 3.5f, 7.0f});
        const auto floatB = storage::Tensor::materializeF32({5}, {2.0f, 4.0f, 3.0f, 0.5f, 1.0f});
        auto floatSubmission = backend::submitNativeGpuAsync(floatRegion, {floatA, floatB});
        require(floatSubmission.ok(), floatSubmission.error ? floatSubmission.error->message : "f32 async submit failed");
        require(floatSubmission.evidence.submittedOperations == 1 &&
                floatSubmission.evidence.pendingOperations == 1 &&
                floatSubmission.evidence.activeReadReservations == 2 &&
                floatSubmission.evidence.activeWriteReservations == 1 &&
                floatSubmission.evidence.eventRecords == 1 &&
                floatSubmission.evidence.kernelLaunches == 1 &&
                floatSubmission.evidence.fusionGroups == 1 &&
                floatSubmission.evidence.fusedKernelGroups == 1 &&
                floatSubmission.evidence.logicalIntermediates == 3 &&
                floatSubmission.evidence.materializedIntermediates == 0 &&
                floatSubmission.evidence.plannerOwnedAllocations == 1 &&
                floatSubmission.evidence.retainedPlannerAllocations == 1 &&
                floatSubmission.evidence.synchronizations == 0,
                "f32 submission/completion evidence was not separated");
        require(!floatSubmission.pending->value().has_value(),
                "pending output masqueraded as a completed value");
        contract([&] { floatSubmission.pending->outputResource().requireReadable(); },
                 "ASYNC-WRITE-PENDING");
        contract([&] { storage::MutableTensorRef mutate(floatA); }, "ASYNC-MUTATION-PENDING");
        contract([&] { auto moved = std::move(floatA); (void)moved; }, "ASYNC-MOVE-PENDING");
        auto floatAlias = floatA;
        require(floatAlias.logicalF32Values().front() == 1.0f,
                "immutable input alias was invalid while async reader was pending");
        auto floatCopy = floatA.deepCopy();
        require(floatCopy.storageId() != floatA.storageId(),
                "explicit copy shared pending input storage");
        const auto floatRun = floatSubmission.pending->observe();
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
        require(floatRun.evidence.kernelLaunches == 1 && floatRun.evidence.synchronizations == 1 &&
                floatRun.evidence.observations == 1 && floatRun.evidence.pendingOperations == 0 &&
                floatRun.evidence.activeReadReservations == 0 &&
                floatRun.evidence.activeWriteReservations == 0 &&
                floatRun.evidence.releasedReservations == 3 &&
                floatRun.evidence.retainedPlannerAllocations == 0,
                "fused observation/reservation/planner evidence mismatch");
        require(floatRun.evidence.hostToDeviceCopies == 2 && floatRun.evidence.deviceToHostCopies == 1,
                "literal transfer evidence mismatch");
        require(floatSubmission.pending->value().has_value() &&
                floatSubmission.pending->observe().ok() &&
                floatSubmission.pending->evidence().synchronizations == 1,
                "completed value/double observation contract failed");
        const auto floatUnfused = backend::executeNativeGpu(
            floatRegion, {floatA, floatB}, 0, {true, false});
        require(floatUnfused.ok(), floatUnfused.error ? floatUnfused.error->message :
                "optimization-disabled f32 GPU run failed");
        same(std::get<storage::Tensor>(*floatUnfused.value).logicalF32Values(), floatExpected);
        require(floatUnfused.evidence.kernelLaunches == 4 &&
                floatUnfused.evidence.fusionGroups == 4 &&
                floatUnfused.evidence.plannerOwnedAllocations == 3 &&
                floatUnfused.evidence.physicalTemporarySlots == 2 &&
                floatUnfused.evidence.reusedSlotAssignments == 1 &&
                floatUnfused.evidence.retainedPlannerAllocations == 0,
                "GPU optimization-enabled/disabled structural evidence mismatch");

        const auto dynamicModule = compile(
            "fn f(x: Tensor<i64,1>, y: Tensor<i64,1>) -> Tensor<i64,1> {\n"
            "let z=x+y\nreturn z.*y\n}");
        const auto dynamicRegion = region(dynamicModule, "f", false);
        const auto host = storage::Tensor::materializeI64({257}, std::vector<std::int64_t>(257, 3));
        auto dynamicSubmission = backend::submitNativeGpuAsync(dynamicRegion, {host, host});
        require(dynamicSubmission.ok() && dynamicSubmission.evidence.activeReadReservations == 1,
                "same aliased input did not collapse to one read reservation");
        const auto dynamicRun = dynamicSubmission.pending->observe();
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
        require(dynamicRun.evidence.inputStorageUploads == 1 &&
                dynamicRun.evidence.hostToDeviceCopies == 2,
                "immutable aliases caused duplicate host-to-device storage");
        require(host.logicalI64Values().front() == 3 &&
                std::get<storage::Tensor>(*dynamicRun.value).storageId() != host.storageId(),
                "GPU execution introduced hidden copy-on-write or mutated input");

        for (std::size_t caseIndex = 0; caseIndex < 100; ++caseIndex) {
            std::vector<std::int64_t> left, right;
            for (std::size_t element = 0; element < 17; ++element) {
                left.push_back(static_cast<std::int64_t>((caseIndex * 37 + element * 19) % 257) - 128);
                right.push_back(static_cast<std::int64_t>((caseIndex * 29 + element * 31) % 129) - 64);
            }
            const auto nativeLeft = storage::Tensor::materializeI64({17}, left);
            const auto nativeRight = storage::Tensor::materializeI64({17}, right);
            semantic::RuntimeValue referenceLeft{semantic::RuntimeTensor{
                semantic::TypeKind::I64, {17}, left, {}}};
            semantic::RuntimeValue referenceRight{semantic::RuntimeTensor{
                semantic::TypeKind::I64, {17}, right, {}}};
            const auto oracle = semantic::evaluateCall(
                dynamicModule, "f", {referenceLeft, referenceRight});
            const auto fused = backend::executeNativeGpu(
                dynamicRegion, {nativeLeft, nativeRight}, 0, {true, true});
            const auto unfused = backend::executeNativeGpu(
                dynamicRegion, {nativeLeft, nativeRight}, 0, {false, false});
            require(oracle.ok && oracle.value && fused.ok() && unfused.ok() &&
                    std::get<storage::Tensor>(*fused.value).logicalI64Values() ==
                        std::get<semantic::RuntimeTensor>(oracle.value->data).values &&
                    std::get<storage::Tensor>(*unfused.value).logicalI64Values() ==
                        std::get<semantic::RuntimeTensor>(oracle.value->data).values,
                    "generated reference/fused/unfused physical-GPU differential mismatch");
        }

        const auto copyModule = compile(
            "fn copied(x:Tensor<i64,1>,y:Tensor<i64,1>)->Tensor<i64,1>{let z=copy(x)\nreturn z+y}");
        const auto copyRegion = region(copyModule, "copied", false);
        auto copyInput = storage::Tensor::materializeI64({257}, std::vector<std::int64_t>(257, 4));
        const auto copyRun = backend::executeNativeGpu(copyRegion, {host, copyInput});
        require(copyRun.ok() && copyRun.evidence.deviceToDeviceCopies == 1 &&
                copyRun.evidence.kernelLaunches == 1 &&
                copyRun.evidence.plannerOwnedAllocations == 2 &&
                std::get<storage::Tensor>(*copyRun.value).logicalI64Values().front() == 7,
                "explicit Copy did not execute through an independent planned device root");

        const auto aliasModule = compile(
            "fn identity(x:Tensor<i64,1>)->Tensor<i64,1>{let y=x\nreturn y}");
        const auto aliasRegion = region(aliasModule, "identity", false);
        auto aliasOutput = backend::submitNativeGpuAsync(aliasRegion, {host});
        require(aliasOutput.ok() && aliasOutput.evidence.activeReadReservations == 1 &&
                aliasOutput.evidence.activeWriteReservations == 1 && !aliasOutput.pending->value(),
                "alias-valued GPU output escaped its pending write obligation");
        const auto aliasObserved = aliasOutput.pending->observe();
        require(aliasObserved.ok() &&
                std::get<storage::Tensor>(*aliasObserved.value).logicalI64Values() == host.logicalI64Values() &&
                std::get<storage::Tensor>(*aliasObserved.value).storageId() != host.storageId(),
                "alias-valued GPU output was not safely observed as an ordinary completed value");

        const auto again = backend::executeNativeGpu(dynamicRegion, {host, host});
        require(again.ok() && std::get<storage::Tensor>(*again.value).logicalI64Values() == dynamicValues,
                "repeat GPU execution was not deterministic");

        for (std::size_t index = 0; index < 150; ++index) {
            auto stress = backend::submitNativeGpuAsync(dynamicRegion, {host, host});
            require(stress.ok() && stress.pending && !stress.pending->observed(),
                    "physical GPU async stress submission failed");
            auto observed = stress.pending->observe();
            require(observed.ok() &&
                    std::get<storage::Tensor>(*observed.value).logicalI64Values() == dynamicValues &&
                    observed.evidence.activeReadReservations == 0 &&
                    observed.evidence.activeWriteReservations == 0 &&
                    observed.evidence.retainedPlannerAllocations == 0,
                    "physical GPU async stress changed output or leaked owned resources");
        }
        require(host.asyncResource().activeReads() == 0 && host.asyncResource().activeWrites() == 0,
                "physical GPU async stress left input reservations active");

        auto independentHost = host.deepCopy();
        auto pendingA = backend::submitNativeGpuAsync(dynamicRegion, {host, host});
        auto pendingB = backend::submitNativeGpuAsync(dynamicRegion, {independentHost, independentHost});
        require(pendingA.ok() && pendingB.ok() && host.asyncResource().activeReads() == 1 &&
                independentHost.asyncResource().activeReads() == 1 &&
                pendingA.evidence.retainedPlannerAllocations == 1 &&
                pendingB.evidence.retainedPlannerAllocations == 1,
                "independent async operations did not remain separately pending");
        const auto observedA = pendingA.pending->observe();
        require(observedA.ok() && host.asyncResource().activeReads() == 0 &&
                independentHost.asyncResource().activeReads() == 1,
                "observing operation A released operation B reservations");
        require(pendingB.pending->observe().ok() && independentHost.asyncResource().activeReads() == 0,
                "independent operation B did not observe cleanly");

        auto sameReadA = backend::submitNativeGpuAsync(dynamicRegion, {host, host});
        auto sameReadB = backend::submitNativeGpuAsync(dynamicRegion, {host, host});
        require(sameReadA.ok() && sameReadB.ok() && host.asyncResource().activeReads() == 2,
                "two physical GPU reads of one immutable storage did not coexist");
        require(sameReadA.pending->observe().ok() && host.asyncResource().activeReads() == 1,
                "first same-resource observation released both reads");
        require(sameReadB.pending->observe().ok() && host.asyncResource().activeReads() == 0,
                "second same-resource observation failed");

        const auto zero = storage::Tensor::empty(storage::DType::I64, {0});
        auto zeroSubmission = backend::submitNativeGpuAsync(dynamicRegion, {zero, zero});
        require(zeroSubmission.ok() && !zeroSubmission.pending->value(),
                "zero-size async submission did not produce a pending value");
        const auto zeroRun = zeroSubmission.pending->observe();
        require(zeroRun.ok() && std::get<storage::Tensor>(*zeroRun.value).logicalI64Values().empty() &&
                zeroRun.evidence.kernelLaunches == 0,
                "zero-size GPU path launched a kernel or changed value");

        const auto pickModule = compile("fn pick(x: Tensor<i64,1>, i:i64)->i64{return x[i]}");
        const auto pickRegion = region(pickModule, "pick", false);
        const auto pick = backend::executeNativeGpu(pickRegion, {host, std::int64_t{256}});
        require(pick.ok() && std::get<std::int64_t>(*pick.value) == 3 && pick.evidence.kernelLaunches == 1,
                "GPU Index kernel result mismatch");
        const auto badPick = backend::submitNativeGpuAsync(pickRegion, {host, std::int64_t{257}});
        require(!badPick.ok() && badPick.error && badPick.error->code == "TH-SPEC-BOUNDS" &&
                badPick.evidence.kernelLaunches == 0 && host.asyncResource().activeReads() == 0,
                "out-of-bounds GPU Index leaked a reservation or launched its kernel");

        const auto overflowHost = storage::Tensor::materializeI64(
            {1}, {std::numeric_limits<std::int64_t>::max()});
        const auto one = storage::Tensor::materializeI64({1}, {1});
        const auto overflowRun = backend::executeNativeGpu(dynamicRegion, {overflowHost, one});
        require(!overflowRun.ok() && overflowRun.error && overflowRun.error->code == "TH-SPEC-I64-OVERFLOW" &&
                overflowRun.evidence.retainedPlannerAllocations == 0,
                "GPU checked i64 overflow did not propagate");

        (void)runtime::takeUnobservedAsyncErrors();
        {
            auto droppedOverflow = backend::submitNativeGpuAsync(dynamicRegion, {overflowHost, one});
            require(droppedOverflow.ok(), "deferred-error drop fixture did not submit");
        }
        const auto droppedErrors = runtime::takeUnobservedAsyncErrors();
        require(droppedErrors.size() == 1 && droppedErrors[0].code == "TH-SPEC-I64-OVERFLOW" &&
                overflowHost.asyncResource().activeReads() == 0 && one.asyncResource().activeReads() == 0,
                "dropped GPU failure disappeared or leaked input reservations");

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

        const auto synchronous = backend::executeNativeGpu(floatRegion, {floatA, floatB});
        require(synchronous.ok() && synchronous.evidence.submittedOperations == 1 &&
                synchronous.evidence.observations == 1 && synchronous.evidence.synchronizations == 1,
                "TH-013 synchronous wrapper is not submit-then-observe");

        std::cout << "V0NativeGpuIntegrationTests PASS " << checks << " checks\n"
                  << "device=" << probe.name << '\n'
                  << "driver_version=" << probe.driverVersion << '\n'
                  << "compute_capability=" << probe.computeMajor << '.' << probe.computeMinor << '\n'
                  << "workload=f32 add,multiply,subtract; i64 add,multiply,index\n"
                  << "async_submissions=" << floatRun.evidence.submittedOperations
                  << " stream_submissions=" << floatRun.evidence.streamSubmissions
                  << " event_records=" << floatRun.evidence.eventRecords
                  << " kernels=" << floatRun.evidence.kernelLaunches
                  << " fusion_groups=" << floatRun.evidence.fusionGroups
                  << " logical_intermediates=" << floatRun.evidence.logicalIntermediates
                  << " physical_temp_slots=" << floatRun.evidence.physicalTemporarySlots
                  << " planner_allocations=" << floatRun.evidence.plannerOwnedAllocations
                  << " observations=" << floatRun.evidence.observations
                  << " synchronizations=" << floatRun.evidence.synchronizations
                  << " released_reservations=" << floatRun.evidence.releasedReservations << '\n'
                  << "fallback=NONE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0NativeGpuIntegrationTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
