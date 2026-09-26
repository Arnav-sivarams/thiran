#include "artifact/v0/NativeArtifacts.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"

#include <cmath>
#include <bit>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
namespace artifact = thiran::v0::artifact;
namespace backend = thiran::v0::backend;
namespace runtime = thiran::v0::runtime;
namespace semantic = thiran::v0::semantic;
namespace storage = thiran::v0::storage;
namespace tooling = thiran::v0::tooling;
using namespace thiran::v0;

namespace {
int checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
fs::path temporary() {
    std::string pattern = "/tmp/th016-gpu-artifacts-XXXXXX";
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
    return pattern;
}
semantic::Module compile(const std::string& text) {
    auto parsed = parse(text, "<th016-gpu-artifact-test>");
    if (!parsed.module) throw std::runtime_error(parsed.diagnostics.front().format());
    auto analyzed = semantic::analyze(*parsed.module);
    if (!analyzed.module) throw std::runtime_error(analyzed.diagnostics.front().format());
    if (!semantic::verify(*analyzed.module).ok) throw std::runtime_error("semantic verifier failed");
    return *analyzed.module;
}
backend::TensorRegion region(const semantic::Module& module, const std::string& name,
                             bool standalone) {
    const auto facts = analysis::analyze(module);
    auto lowered = backend::extractStrictGpu(module, facts, name, standalone);
    if (!lowered.ok()) throw std::runtime_error(lowered.diagnostic);
    return *lowered.region;
}
void same(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "f32 size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index)
        require(std::fabs(actual[index] - expected[index]) <= 1e-6f, "f32 value mismatch");
}
}

int main() {
    const auto probe = backend::probeNativeGpu(0);
    if (!probe.deviceAvailable) {
        std::cout << "NativeArtifactGpuIntegrationTests SKIP/UNAVAILABLE "
                  << (probe.error ? probe.error->code + ": " + probe.error->message : "unknown") << '\n';
        return 77;
    }
    const auto root = temporary();
    try {
        const auto floatModule = compile(
            "fn fused(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{\n"
            "let a=x+y\nlet b=a.*y\nlet c=b-x\nreturn c+y\n}");
        const auto floatRegion = region(floatModule, "fused", false);
        const auto x = storage::Tensor::materializeF32({5}, {1, 2, 0, 3.5f, 7});
        const auto y = storage::Tensor::materializeF32({5}, {2, 4, 3, 0.5f, 1});
        const auto aotPath = root / "fused-gpu.tha";
        auto built = artifact::buildGpuAot(floatRegion, {aotPath, {true, true}, {}});
        require(built.success, built.error ? built.error->message : "GPU AOT build failed");
        auto loaded = artifact::loadArtifact(aotPath);
        require(loaded.ok(), loaded.error ? loaded.error->message : "GPU AOT load failed");
        const std::string ptx(reinterpret_cast<const char*>(loaded.artifact->payload.data()),
                              loaded.artifact->payload.size());
        require(loaded.artifact->manifest.backend == artifact::NativeBackend::Gpu &&
                loaded.artifact->manifest.payloadKind == artifact::PayloadKind::Ptx &&
                loaded.artifact->manifest.runtimeRequirement.find("driver-ptx-jit") != std::string::npos &&
                ptx.find(".visible .entry thiran_group_") != std::string::npos &&
                ptx.find("_f32") != std::string::npos,
                "GPU artifact lacks persistent fused PTX/driver-JIT declaration");
        const auto inspection = artifact::inspectArtifact(*loaded.artifact);
        require(inspection.find("backend=gpu") != std::string::npos &&
                inspection.find("payload_kind=ptx") != std::string::npos &&
                inspection.find("planning.fusion=true") != std::string::npos,
                "GPU artifact inspection omitted metadata");

        auto aotRun = artifact::executeArtifact(*loaded.artifact, {x, y});
        require(aotRun.ok(), aotRun.error ? aotRun.error->message : "GPU AOT execution failed");
        const auto aotValues = std::get<storage::Tensor>(*aotRun.value).logicalF32Values();
        semantic::RuntimeValue referenceX{semantic::RuntimeTensor{semantic::TypeKind::F32, {5}, {},
            {1, 2, 0, 3.5f, 7}}};
        semantic::RuntimeValue referenceY{semantic::RuntimeTensor{semantic::TypeKind::F32, {5}, {},
            {2, 4, 3, 0.5f, 1}}};
        const auto reference = semantic::evaluateCall(floatModule, "fused", {referenceX, referenceY});
        require(reference.ok && reference.value, "GPU AOT reference failed");
        same(aotValues, std::get<semantic::RuntimeTensor>(reference.value->data).f32Values);
        require(aotRun.gpuEvidence && aotRun.gpuEvidence->kernelLaunches == 1 &&
                aotRun.gpuEvidence->fusionGroups == 1 &&
                aotRun.gpuEvidence->fusedKernelGroups == 1 &&
                aotRun.gpuEvidence->plannerOwnedAllocations == 1 &&
                aotRun.gpuEvidence->retainedPlannerAllocations == 0 &&
                aotRun.gpuEvidence->observations == 1,
                "GPU AOT did not consume fused plan through async observation");

        auto direct = backend::executeNativeGpu(floatRegion, {x, y});
        require(direct.ok(), direct.error ? direct.error->message : "direct GPU execution failed");
        same(std::get<storage::Tensor>(*direct.value).logicalF32Values(), aotValues);

        const auto specialModule = compile(
            "fn identity(x:Tensor<f32,1>)->Tensor<f32,1>{return x}");
        const auto specialRegion = region(specialModule, "identity", false);
        const std::vector<float> specials{
            0.0f, -0.0f, std::numeric_limits<float>::min(),
            std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
            std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
        const auto specialInput = storage::Tensor::materializeF32({7}, specials);
        const auto specialEntry = artifact::specializeEntry(specialRegion, {specialInput});
        auto specialBuild = artifact::buildGpuAot(
            specialRegion, {root / "f32-specials-gpu.tha", {}, specialEntry});
        auto specialRun = specialBuild.success ? artifact::loadAndExecuteArtifact(
            root / "f32-specials-gpu.tha", {specialInput}) : artifact::ArtifactExecutionResult{};
        require(specialBuild.success && specialRun.ok(), "GPU f32-special identity failed");
        const auto specialOutput = std::get<storage::Tensor>(*specialRun.value).logicalF32Values();
        require(specialOutput.size() == specials.size(), "GPU f32-special output size changed");
        for (std::size_t index = 0; index < specials.size(); ++index)
            require(std::bit_cast<std::uint32_t>(specialOutput[index]) ==
                    std::bit_cast<std::uint32_t>(specials[index]),
                    "GPU f32-special identity changed bits");

        auto plan = backend::buildPhysicalPlan(floatRegion, backend::PhysicalDevice::Gpu);
        require(plan.ok(), "GPU artifact plan reconstruction failed");
        auto pending = backend::submitNativeGpuPayloadAsync(floatRegion, *plan.plan, ptx, {x, y});
        require(pending.ok() && !pending.pending->observed() &&
                !pending.pending->value() && pending.pending->evidence().activeReadReservations == 2 &&
                pending.pending->evidence().activeWriteReservations == 1,
                "persistent PTX submission weakened TH-014 pending semantics");
        auto observed = pending.pending->observe();
        require(observed.ok() && observed.evidence.observations == 1 &&
                observed.evidence.activeReadReservations == 0 &&
                observed.evidence.activeWriteReservations == 0 &&
                observed.evidence.retainedPlannerAllocations == 0,
                "persistent PTX observation leaked reservations/resources");
        same(std::get<storage::Tensor>(*observed.value).logicalF32Values(), aotValues);

        const auto mainModule = compile(
            "fn main()->Tensor<i64,1>{\nlet x=[1,2,3]\nlet y=[4,5,6]\n"
            "let a=x+y\nlet b=a.*y\nreturn b-x\n}");
        const auto mainRegion = region(mainModule, "main", true);
        const auto standalonePath = root / "standalone-gpu.tha";
        auto standaloneBuild = artifact::buildGpuAot(
            mainRegion, {standalonePath, {true, true}, {}});
        require(standaloneBuild.success, "standalone GPU artifact build failed");
        const auto fresh = root / "fresh-gpu";
        fs::create_directory(fresh);
        fs::copy_file(standalonePath, fresh / "program.tha");
        fs::copy_file(TH016_RUNTIME, fresh / "thiran-artifact");
        fs::permissions(fresh / "thiran-artifact", fs::perms::owner_read | fs::perms::owner_write |
                        fs::perms::owner_exec, fs::perm_options::add);
        const auto prior = fs::current_path();
        fs::current_path(fresh);
        ::setenv("PATH", "/nonexistent", 1);
        ::setenv("CXX", "/nonexistent", 1);
        auto freshRun = tooling::runProcess({(fresh / "thiran-artifact").string(),
                                             {"run", "program.tha"}});
        ::unsetenv("CXX");
        ::setenv("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", 1);
        fs::current_path(prior);
        require(freshRun.exitStatus == 0 && freshRun.standardOutput ==
                "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[3],\"values\":[19,33,51]}\n" &&
                freshRun.standardError.find("thiran_ptx_generation=0") != std::string::npos &&
                freshRun.standardError.find("driver_jit=1") != std::string::npos &&
                freshRun.standardError.find("kernels=1") != std::string::npos &&
                freshRun.standardError.find("fallback=NONE") != std::string::npos,
                "fresh-process persistent GPU AOT execution failed: " + freshRun.standardError);

        const auto integerModule = compile(
            "fn add(x:Tensor<i64,1>,y:Tensor<i64,1>)->Tensor<i64,1>{let z=x+y\nreturn z.*y}");
        const auto integerRegion = region(integerModule, "add", false);
        const auto maximum = storage::Tensor::materializeI64(
            {1}, {std::numeric_limits<std::int64_t>::max()});
        const auto one = storage::Tensor::materializeI64({1}, {1});
        auto overflowBuild = artifact::buildGpuAot(
            integerRegion, {root / "overflow-gpu.tha", {}, {}});
        auto overflow = overflowBuild.success ?
            artifact::loadAndExecuteArtifact(root / "overflow-gpu.tha", {maximum, one}) :
            artifact::ArtifactExecutionResult{};
        require(overflowBuild.success && !overflow.ok() && overflow.error &&
                overflow.error->code == "TH-SPEC-I64-OVERFLOW" && overflow.gpuEvidence &&
                overflow.gpuEvidence->retainedPlannerAllocations == 0,
                "GPU AOT checked i64 overflow/lifetime semantics mismatch");
        auto invalidDevice = artifact::executeArtifact(*loaded.artifact, {x, y}, probe.deviceCount);
        require(!invalidDevice.ok() && invalidDevice.error &&
                invalidDevice.error->code == "GPU-INVALID-DEVICE",
                "GPU artifact invalid device was not explicit");

        artifact::JitCompiler jit;
        auto jitFirst = jit.compileGpu(floatRegion, {x, y});
        require(jitFirst.ok() && !jitFirst.cacheHit && jit.statistics().compilations == 1 &&
                jit.statistics().cacheHits == 0 && jit.statistics().cacheMisses == 1,
                "first Thiran-side GPU JIT compilation evidence mismatch");
        auto jitRun = jitFirst.executable->execute({x, y});
        require(jitRun.ok() && jitRun.gpuEvidence && jitRun.gpuEvidence->kernelLaunches == 1,
                jitRun.error ? jitRun.error->message : "GPU JIT execution failed");
        same(std::get<storage::Tensor>(*jitRun.value).logicalF32Values(), aotValues);
        auto jitHit = jit.compileGpu(floatRegion, {x, y});
        require(jitHit.ok() && jitHit.cacheHit &&
                jitHit.executable.get() == jitFirst.executable.get() &&
                jit.statistics().compilations == 1 && jit.statistics().cacheHits == 1,
                "identical GPU JIT specialization did not hit cache");
        const auto x6 = storage::Tensor::materializeF32({6}, {1, 2, 3, 4, 5, 6});
        const auto y6 = storage::Tensor::materializeF32({6}, {6, 5, 4, 3, 2, 1});
        auto shapeMiss = jit.compileGpu(floatRegion, {x6, y6});
        require(shapeMiss.ok() && !shapeMiss.cacheHit &&
                shapeMiss.executable->cacheKey() != jitFirst.executable->cacheKey() &&
                jit.statistics().compilations == 2, "GPU JIT changed shape did not miss cache");
        auto optionMiss = jit.compileGpu(floatRegion, {x, y}, {true, false});
        require(optionMiss.ok() && !optionMiss.cacheHit &&
                optionMiss.executable->cacheKey() != jitFirst.executable->cacheKey() &&
                jit.statistics().compilations == 3, "GPU JIT planning/fusion option did not miss cache");
        auto unfused = optionMiss.executable->execute({x, y});
        require(unfused.ok() && unfused.gpuEvidence && unfused.gpuEvidence->kernelLaunches == 4,
                "GPU JIT unfused specialization did not execute its planned payload");
        same(std::get<storage::Tensor>(*unfused.value).logicalF32Values(), aotValues);
        auto incompatible = jitFirst.executable->execute({x6, y6});
        require(!incompatible.ok() && incompatible.error->code == "ARTIFACT-INPUT-ABI",
                "GPU JIT executable accepted incompatible specialization");
        auto jitOverflow = jit.compileGpu(integerRegion, {maximum, one});
        auto jitOverflowRun = jitOverflow.executable->execute({maximum, one});
        require(!jitOverflowRun.ok() && jitOverflowRun.error &&
                jitOverflowRun.error->code == "TH-SPEC-I64-OVERFLOW" &&
                jitOverflowRun.gpuEvidence->retainedPlannerAllocations == 0,
                "GPU JIT checked overflow/resource cleanup mismatch");
        auto jitInvalidDevice = jitFirst.executable->execute({x, y}, probe.deviceCount);
        require(!jitInvalidDevice.ok() && jitInvalidDevice.error->code == "GPU-INVALID-DEVICE",
                "GPU JIT invalid device was not explicit");

        auto malformed = *loaded.artifact;
        const std::string invalidPtx = ".version broken\n";
        malformed.payload.resize(invalidPtx.size());
        std::memcpy(malformed.payload.data(), invalidPtx.data(), invalidPtx.size());
        malformed.manifest.payloadSize = malformed.payload.size();
        malformed.manifest.payloadDigest = artifact::digestBytes(
            malformed.payload.data(), malformed.payload.size());
        require(!artifact::writeArtifact(malformed, root / "malformed-ptx.tha"),
                "could not write malformed PTX fixture");
        require(!artifact::loadArtifact(root / "malformed-ptx.tha").ok(),
                "structurally malformed PTX artifact was accepted");

        std::cout << "NativeArtifactGpuIntegrationTests PASS " << checks << " checks\n"
                  << "device=" << probe.name << '\n'
                  << "driver_version=" << probe.driverVersion << '\n'
                  << "compute_capability=" << probe.computeMajor << '.' << probe.computeMinor << '\n'
                  << "gpu_artifact_bytes=" << fs::file_size(aotPath) << '\n'
                  << "ptx_payload_bytes=" << loaded.artifact->manifest.payloadSize << '\n'
                  << "aot_kernels=" << aotRun.gpuEvidence->kernelLaunches
                  << " jit_compilations=" << jit.statistics().compilations
                  << " cache_hits=" << jit.statistics().cacheHits
                  << " cache_misses=" << jit.statistics().cacheMisses << '\n'
                  << "aot_thiran_ptx_generation_at_execution=0 driver_jit=1 fallback=NONE\n";
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "NativeArtifactGpuIntegrationTests FAIL: " << failure.what() << '\n';
        std::error_code ignored;
        fs::remove_all(root, ignored);
        return 1;
    }
}
