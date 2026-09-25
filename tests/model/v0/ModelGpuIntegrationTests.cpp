#include "model/v0/ReferenceModel.hpp"
#include "tooling/v0/BuildConfig.hpp"
#include "tooling/v0/Process.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
namespace artifact = thiran::v0::artifact;
namespace backend = thiran::v0::backend;
namespace model = thiran::v0::model;
namespace persistence = thiran::v0::persistence;
namespace semantic = thiran::v0::semantic;
namespace storage = thiran::v0::storage;
namespace tooling = thiran::v0::tooling;
namespace training = thiran::v0::training;

namespace {

int checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void same(float actual, float expected) {
    require(std::fabs(actual - expected) <= 1e-6f,
            "f32 mismatch actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}
fs::path temporary() {
    std::string pattern = "/tmp/th017-model-gpu-XXXXXX";
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
    return pattern;
}
semantic::RuntimeValue tensor(std::vector<std::int64_t> shape, std::vector<float> values) {
    return semantic::RuntimeValue{semantic::RuntimeTensor{
        semantic::TypeKind::F32, std::move(shape), {}, std::move(values)}};
}
const training::ParameterValue& parameter(const training::TrainingPlan& plan,
                                          const training::TrainingState& state,
                                          std::size_t sourceIndex) {
    const auto descriptor = std::find_if(plan.parameters.begin(), plan.parameters.end(),
        [&](const auto& candidate) { return candidate.sourceParameterIndex == sourceIndex; });
    if (descriptor == plan.parameters.end())
        throw std::runtime_error("reference parameter descriptor missing");
    const auto value = std::find_if(state.parameters.begin(), state.parameters.end(),
        [&](const auto& candidate) { return candidate.id == descriptor->id; });
    if (value == state.parameters.end())
        throw std::runtime_error("reference parameter missing");
    return *value;
}
artifact::ArtifactValue artifactValue(const training::ParameterValue& value) {
    const auto& tensorValue = std::get<semantic::RuntimeTensor>(value.value.data);
    std::vector<std::uint64_t> shape;
    for (auto extent : tensorValue.shape) shape.push_back(static_cast<std::uint64_t>(extent));
    return storage::Tensor::materializeF32(shape, tensorValue.f32Values);
}
float resultValue(const artifact::ArtifactValue& value) {
    return std::get<storage::Tensor>(value).logicalF32Values().at(0);
}
artifact::NativeToolchain toolchain() {
    const fs::path build = TH017_BUILD;
    return {TH017_CXX, tooling::configuredHostCompilerArguments(), {TH017_INCLUDE},
            {build / "libthiran_v0_storage.a", build / "libthiran_v0_async.a",
             build / "libthiran_v0_analysis.a", build / "libthiran_v0_semantic.a",
             build / "libthiran_v0_frontend.a"}};
}

} // namespace

int main() {
    const auto probe = backend::probeNativeGpu(0);
    if (!probe.deviceAvailable) {
        std::cout << "V0ModelGpuIntegrationTests SKIP/UNAVAILABLE "
                  << (probe.error ? probe.error->code + ": " + probe.error->message : "unknown") << '\n';
        return 77;
    }
    const auto root = temporary();
    try {
        auto reference = model::createReferenceAffinePlan();
        require(reference.ok(), "reference plan failed: " + reference.error);
        auto initialized = model::initializeReferenceAffineState(*reference.plan);
        require(initialized.ok(), "reference initialization failed");
        auto schema = training::createTrainingCheckpointSchema(
            std::string(model::referenceAffineModelIdentity), *reference.plan, *initialized.state);
        require(schema.ok(), "reference checkpoint schema failed");
        float initialLoss = 0.0f, finalLoss = 0.0f;
        auto trained = std::optional<training::TrainingState>{std::move(*initialized.state)};
        for (std::size_t step = 0; step < 200; ++step) {
            auto next = training::trainingStep(*reference.plan, *trained, model::referenceAffineBatch());
            if (!next.ok()) throw std::runtime_error("reference training step failed");
            if (step == 0) initialLoss = *next.loss;
            finalLoss = *next.loss;
            trained = std::move(*next.nextState);
        }
        const auto checkpoint = root / "trained.thc";
        auto saved = training::saveTrainingCheckpoint(checkpoint, *schema.schema,
                                                       *reference.plan, *trained);
        require(saved.ok() && saved.metadata->step == 200, "trained state save failed");
        const auto trainedDigest = saved.metadata->parameterDigest;
        trained.reset();
        require(!trained, "original physical-qualification TrainingState remained alive");
        auto restored = training::loadTrainingCheckpoint(checkpoint, *schema.schema, *reference.plan);
        require(restored.ok() && restored.metadata->parameterDigest == trainedDigest,
                "physical-qualification checkpoint reload failed");
        auto snapshot = model::createDeploymentSnapshot(
            *reference.plan, restored, model::referenceAffineSnapshotRequest());
        require(snapshot.ok() && snapshot.snapshot->parameterDigest == trainedDigest,
                "physical deployment snapshot provenance failed");

        auto cpuRegion = model::referenceAffineInferenceRegion(*reference.module, backend::NativeTarget::Cpu);
        auto gpuRegion = model::referenceAffineInferenceRegion(*reference.module, backend::NativeTarget::Gpu);
        require(cpuRegion.ok() && gpuRegion.ok() && cpuRegion.region->dump() == gpuRegion.region->dump(),
                "CPU/GPU lowering differs for reference inference semantics");
        auto region = std::move(*cpuRegion.region);
        const auto weight = artifactValue(parameter(*reference.plan, *restored.state, 2));
        const auto bias = artifactValue(parameter(*reference.plan, *restored.state, 3));
        const auto input = model::referenceAffinePublicInput(2.0f);
        const auto entry = artifact::specializeEntry(region, {input, weight, bias});
        auto cpuBuilt = artifact::buildCpuAot(region, toolchain(),
            {root / "affine-cpu.tha", {true, true}, entry});
        auto gpuBuilt = artifact::buildGpuAot(region,
            {root / "affine-gpu.tha", {true, true}, entry});
        require(cpuBuilt.success && gpuBuilt.success,
                gpuBuilt.error ? gpuBuilt.error->message : "dual artifact build failed");
        auto cpuArtifact = artifact::loadArtifact(root / "affine-cpu.tha");
        auto gpuArtifact = artifact::loadArtifact(root / "affine-gpu.tha");
        require(cpuArtifact.ok() && gpuArtifact.ok(), "dual artifact reload failed");
        const std::string ptx(reinterpret_cast<const char*>(gpuArtifact.artifact->payload.data()),
                              gpuArtifact.artifact->payload.size());
        require(ptx.find(".visible .entry thiran_group_") != std::string::npos &&
                ptx.find("thiran_index_f32") != std::string::npos,
                "persistent GPU payload lacks inference kernels");
        auto planned = backend::buildPhysicalPlan(region, backend::PhysicalDevice::Gpu, {true,true});
        require(planned.ok() && std::any_of(planned.plan->fusionGroups.begin(), planned.plan->fusionGroups.end(),
            [](const auto& group) { return group.fused && group.nodes.size() == 2; }),
            "reference deployment did not retain eligible TH-015 fusion");

        auto bundled = model::createModelBundle(*snapshot.snapshot,
            {{std::string(model::referenceAffineModelIdentity), *cpuArtifact.artifact},
             {std::string(model::referenceAffineModelIdentity), *gpuArtifact.artifact}});
        require(bundled.ok(), bundled.error ? bundled.error->message : "dual bundle creation failed");
        auto written = model::writeModelBundle(*bundled.bundle, root / "affine.thm");
        require(written.ok(), "dual bundle write failed");
        auto loaded = model::loadModelBundle(root / "affine.thm",
                                             std::string(model::referenceAffineModelIdentity));
        require(loaded.ok() && loaded.bundle->parameterDigest == trainedDigest,
                "dual bundle reload/digest failed");

        std::ostringstream predictions;
        predictions << std::setprecision(9) << "input reference cpu gpu\n";
        std::optional<backend::GpuExecutionEvidence> lastEvidence;
        for (float x : {0.0f, 1.0f, 2.0f, 5.0f}) {
            auto oracle = model::evaluateReferenceAffine(
                *reference.module, *reference.plan, *restored.state, x);
            auto cpu = model::executeModel(*loaded.bundle, model::ModelBackend::Cpu,
                                           {model::referenceAffinePublicInput(x)});
            auto gpu = model::executeModel(*loaded.bundle, model::ModelBackend::Gpu,
                                           {model::referenceAffinePublicInput(x)});
            require(oracle.ok && cpu.ok() && gpu.ok(),
                    gpu.error ? gpu.error->message : "CPU/GPU/reference execution failed");
            const auto expected = std::get<semantic::RuntimeTensor>(oracle.value->data).f32Values.at(0);
            const auto cpuValue = resultValue(*cpu.value);
            const auto gpuValue = resultValue(*gpu.value);
            same(cpuValue, expected); same(gpuValue, expected); same(cpuValue, gpuValue);
            lastEvidence = gpu.gpuEvidence;
            predictions << x << ' ' << expected << ' ' << cpuValue << ' ' << gpuValue << '\n';
        }
        require(lastEvidence && lastEvidence->kernelLaunches == 2 &&
                lastEvidence->fusedKernelGroups == 1 && lastEvidence->observations == 1 &&
                lastEvidence->activeReadReservations == 0 &&
                lastEvidence->activeWriteReservations == 0 &&
                lastEvidence->retainedPlannerAllocations == 0,
                "GPU inference planning/observation evidence mismatch");

        auto badDevice = model::executeModel(*loaded.bundle, model::ModelBackend::Gpu,
            {model::referenceAffinePublicInput(2.0f)}, probe.deviceCount);
        require(!badDevice.ok() && badDevice.error->code == "GPU-INVALID-DEVICE",
                "invalid model GPU device was not explicit");

        auto publicTensor = storage::Tensor::materializeF32({1}, {5.0f});
        auto pendingBundle = std::optional<model::ModelBundle>{*loaded.bundle};
        auto submitted = model::submitModelGpu(*pendingBundle, {publicTensor});
        require(submitted.ok() && !submitted.pending->observed() && !submitted.pending->value() &&
                publicTensor.asyncResource().activeReads() == 1 &&
                submitted.pending->evidence().activeReadReservations == 3 &&
                submitted.pending->evidence().activeWriteReservations == 1,
                "model GPU submission did not retain input/frozen parameter reservations");
        pendingBundle.reset();
        auto observed = submitted.pending->observe();
        require(observed.ok() && publicTensor.asyncResource().activeReads() == 0 &&
                observed.evidence.activeReadReservations == 0 &&
                observed.evidence.activeWriteReservations == 0 &&
                observed.evidence.retainedPlannerAllocations == 0,
                "model GPU observation after bundle destruction leaked or lost resources");
        auto pendingOracle = model::evaluateReferenceAffine(
            *reference.module, *reference.plan, *restored.state, 5.0f);
        require(pendingOracle.ok, "pending-operation reference evaluation failed");
        same(resultValue(*observed.value),
             std::get<semantic::RuntimeTensor>(pendingOracle.value->data).f32Values.at(0));

        auto identityMismatch = *loaded.bundle;
        identityMismatch.artifacts[1].modelIdentity = "wrong.model";
        require(model::validateModelBundle(identityMismatch)->code == "MODEL-ARTIFACT-IDENTITY",
                "CPU/GPU model identity mismatch accepted");
        const auto x2 = storage::Tensor::materializeF32({2}, {1,2});
        const auto mismatchedEntry = artifact::specializeEntry(region, {x2, weight, bias});
        auto mismatchedGpu = artifact::buildGpuAot(region,
            {root / "mismatched-gpu.tha", {true,true}, mismatchedEntry});
        require(mismatchedGpu.success, "mismatched-signature GPU fixture build failed");
        auto mismatchedLoaded = artifact::loadArtifact(root / "mismatched-gpu.tha");
        auto signatureMismatch = model::createModelBundle(*snapshot.snapshot,
            {{std::string(model::referenceAffineModelIdentity), *cpuArtifact.artifact},
             {std::string(model::referenceAffineModelIdentity), *mismatchedLoaded.artifact}});
        require(!signatureMismatch.ok() && signatureMismatch.error->code == "MODEL-ARTIFACT-SIGNATURE",
                "CPU/GPU artifact signature mismatch accepted");

        const auto fresh = root / "fresh deployment";
        fs::create_directory(fresh);
        fs::copy_file(root / "affine.thm", fresh / "model.thm");
        fs::copy_file(TH017_RUNTIME, fresh / "thiran-model");
        fs::permissions(fresh / "thiran-model", fs::perms::owner_read | fs::perms::owner_write |
                        fs::perms::owner_exec, fs::perm_options::add);
        require(!fs::exists(fresh / "trained.thc") && !fs::exists(fresh / "model.th"),
                "fresh physical deployment contains training/source state");
        const auto prior = fs::current_path();
        const std::string oldPath = std::getenv("PATH") ? std::getenv("PATH") : "";
        const std::string oldCxx = std::getenv("CXX") ? std::getenv("CXX") : "";
        const bool hadCxx = std::getenv("CXX") != nullptr;
        fs::current_path(fresh);
        ::setenv("PATH", "/nonexistent", 1);
        ::setenv("CXX", "/nonexistent", 1);
        auto inspect = tooling::runProcess({(fresh / "thiran-model").string(), {"inspect", "model.thm"}});
        auto freshCpu = tooling::runProcess({(fresh / "thiran-model").string(),
            {"run", "model.thm", "--backend", "cpu", "--input", "5"}});
        auto freshGpu = tooling::runProcess({(fresh / "thiran-model").string(),
            {"run", "model.thm", "--backend", "gpu", "--input", "5"}});
        ::setenv("PATH", oldPath.c_str(), 1);
        if (hadCxx) ::setenv("CXX", oldCxx.c_str(), 1); else ::unsetenv("CXX");
        fs::current_path(prior);
        require(inspect.exitStatus == 0 && inspect.standardOutput.find("backend.gpu=available") != std::string::npos,
                "fresh dual-bundle inspection failed");
        require(freshCpu.exitStatus == 0 && freshCpu.standardError.find("compiler_invocations=0") != std::string::npos &&
                freshCpu.standardError.find("fallback=NONE") != std::string::npos,
                "fresh CPU model deployment failed");
        require(freshGpu.exitStatus == 0 && freshGpu.standardError.find("thiran_ptx_generation=0") != std::string::npos &&
                freshGpu.standardError.find("driver_jit=1") != std::string::npos &&
                freshGpu.standardError.find("fallback=NONE") != std::string::npos,
                "fresh GPU model deployment failed: " + freshGpu.standardError);

        const auto weightValue = std::get<semantic::RuntimeTensor>(
            parameter(*reference.plan, *restored.state, 2).value.data).f32Values.at(0);
        const auto biasValue = std::get<semantic::RuntimeTensor>(
            parameter(*reference.plan, *restored.state, 3).value.data).f32Values.at(0);
        std::cout << std::setprecision(9)
                  << "V0ModelGpuIntegrationTests PASS " << checks << " checks\n"
                  << "device=" << probe.name << '\n'
                  << "driver_version=" << probe.driverVersion << '\n'
                  << "compute_capability=" << probe.computeMajor << '.' << probe.computeMinor << '\n'
                  << "initial_loss=" << initialLoss << '\n'
                  << "final_loss=" << finalLoss << '\n'
                  << "weight=" << weightValue << '\n'
                  << "bias=" << biasValue << '\n'
                  << "checkpoint_step=" << restored.state->step << '\n'
                  << "parameter_digest=" << trainedDigest << '\n'
                  << predictions.str()
                  << "gpu_kernels=" << lastEvidence->kernelLaunches
                  << " fused_groups=" << lastEvidence->fusedKernelGroups
                  << " observations=" << lastEvidence->observations
                  << " released_reservations=" << lastEvidence->releasedReservations << '\n'
                  << "aot_thiran_ptx_generation_at_execution=0 driver_jit=1 fallback=NONE\n";
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "V0ModelGpuIntegrationTests FAIL: " << failure.what() << '\n';
        std::error_code ignored;
        fs::remove_all(root, ignored);
        return 1;
    }
}
