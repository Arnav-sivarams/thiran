#include "artifact/v0/NativeArtifacts.hpp"
#include "backend/v0/NativeCpu.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"
#include "tooling/v0/BuildConfig.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
namespace artifact = thiran::v0::artifact;
namespace backend = thiran::v0::backend;
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
    std::string pattern = "/tmp/th016-artifacts-XXXXXX";
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
    return pattern;
}
semantic::Module compile(const std::string& text) {
    auto parsed = parse(text, "<th016-artifact-test>");
    if (!parsed.module) throw std::runtime_error(parsed.diagnostics.front().format());
    auto analyzed = semantic::analyze(*parsed.module);
    if (!analyzed.module) throw std::runtime_error(analyzed.diagnostics.front().format());
    if (!semantic::verify(*analyzed.module).ok) throw std::runtime_error("semantic verifier failed");
    return *analyzed.module;
}
backend::TensorRegion region(const semantic::Module& module, const std::string& name = "main",
                             bool standalone = true, bool gpu = false) {
    const auto facts = analysis::analyze(module);
    if (!facts.ok()) throw std::runtime_error("ownership qualification failed");
    auto lowered = gpu ? backend::extractStrictGpu(module, facts, name, standalone) :
                         backend::extractStrictNative(module, facts, name, standalone);
    if (!lowered.ok()) throw std::runtime_error(lowered.diagnostic);
    return *lowered.region;
}
artifact::NativeToolchain toolchain() {
    const fs::path build = TH016_BUILD;
    return {TH016_CXX, tooling::configuredHostCompilerArguments(), {TH016_INCLUDE},
            {build / "libthiran_v0_storage.a", build / "libthiran_v0_async.a",
             build / "libthiran_v0_analysis.a", build / "libthiran_v0_semantic.a",
             build / "libthiran_v0_frontend.a"}};
}
void writeBytes(const fs::path& path, const std::vector<std::byte>& bytes, std::size_t count) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(count));
    if (!out) throw std::runtime_error("test byte write failed");
}
std::vector<std::byte> readBytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    const auto size = in.tellg();
    std::vector<std::byte> result(static_cast<std::size_t>(size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(result.data()), size);
    return result;
}
bool rejected(const artifact::NativeArtifact& original, const fs::path& path,
              const std::function<void(artifact::NativeArtifact&)>& mutate,
              const std::string& code) {
    auto changed = original;
    mutate(changed);
    if (auto failure = artifact::writeArtifact(changed, path))
        throw std::runtime_error(failure->message);
    auto loaded = artifact::loadArtifact(path);
    return !loaded.ok() && loaded.error && loaded.error->code == code;
}
}

int main() {
    const auto root = temporary();
    try {
        const auto integerModule = compile(
            "fn main()->Tensor<i64,2>{\nlet A=[1,2;3,4]\nlet B=[5,6;7,8]\n"
            "let C=A+B\nlet D=C.*B\nreturn D-A\n}");
        const auto integerRegion = region(integerModule);
        const auto artifactPath = root / "integer.tha";
        auto integerBuild = artifact::buildCpuAot(
            integerRegion, toolchain(), {artifactPath, {true, true}, {}});
        require(integerBuild.success && integerBuild.compilerProcess &&
                integerBuild.compilerProcess->exitStatus == 0, "CPU AOT creation failed");
        auto loaded = artifact::loadArtifact(artifactPath);
        require(loaded.ok(), loaded.error ? loaded.error->message : "CPU artifact load failed");
        require(loaded.artifact->manifest.formatVersion == 0 &&
                loaded.artifact->manifest.compilerAbiVersion == 1 &&
                loaded.artifact->manifest.runtimeAbiVersion == 1 &&
                loaded.artifact->manifest.backend == artifact::NativeBackend::Cpu &&
                loaded.artifact->manifest.payloadKind == artifact::PayloadKind::ElfSharedObject,
                "CPU manifest versions/backend/payload kind mismatch");
        const auto inspection = artifact::inspectArtifact(*loaded.artifact);
        require(inspection.find("format_version=0") != std::string::npos &&
                inspection.find("payload_kind=elf-shared-object") != std::string::npos &&
                inspection.find("planning.fusion=true") != std::string::npos &&
                inspection.find("plan_digest=fnv1a64:") != std::string::npos,
                "CPU artifact inspection omitted auditable metadata");
        auto integerRun = artifact::executeArtifact(*loaded.artifact);
        require(integerRun.ok() && artifact::formatArtifactValue(*integerRun.value) ==
                "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[29,46,67,92]}",
                integerRun.error ? integerRun.error->message : "CPU AOT result mismatch");
        const auto reference = semantic::evaluateCall(integerModule, "main", {});
        require(reference.ok && reference.value && reference.format() ==
                artifact::formatArtifactValue(*integerRun.value), "CPU artifact/reference mismatch");

        const auto relocated = root / "fresh location";
        fs::create_directory(relocated);
        fs::copy_file(artifactPath, relocated / "program.tha");
        fs::copy_file(TH016_RUNTIME, relocated / "thiran-artifact");
        fs::permissions(relocated / "thiran-artifact", fs::perms::owner_exec | fs::perms::owner_read |
                        fs::perms::owner_write, fs::perm_options::add);
        const auto prior = fs::current_path();
        fs::current_path(relocated);
        const char* oldPath = std::getenv("PATH");
        const std::string savedPath = oldPath ? oldPath : "";
        ::setenv("PATH", "/nonexistent", 1);
        ::setenv("CXX", "/nonexistent", 1);
        auto fresh = tooling::runProcess({(relocated / "thiran-artifact").string(),
                                          {"run", "program.tha"}});
        ::setenv("PATH", savedPath.c_str(), 1);
        ::unsetenv("CXX");
        fs::current_path(prior);
        require(fresh.launched && fresh.exitStatus == 0 && fresh.standardOutput ==
                artifact::formatArtifactValue(*integerRun.value) + "\n" &&
                fresh.standardError.find("compiler_invocations=0 fallback=NONE") != std::string::npos,
                "relocated compiler-absent CPU execution failed: " + fresh.standardError);

        auto inspectProcess = tooling::runProcess({(relocated / "thiran-artifact").string(),
                                                   {"inspect", (relocated / "program.tha").string()}});
        require(inspectProcess.exitStatus == 0 && inspectProcess.standardOutput == inspection,
                "fresh-process artifact inspection was not deterministic");

        const auto payloadPath = root / "payload.so";
        writeBytes(payloadPath, loaded.artifact->payload, loaded.artifact->payload.size());
        auto dependencies = tooling::runProcess({"ldd", {payloadPath.string()}});
        auto symbols = tooling::runProcess({"readelf", {"-d", payloadPath.string()}});
        auto strings = tooling::runProcess({"strings", {payloadPath.string()}});
        require(dependencies.exitStatus == 0 && symbols.exitStatus == 0 && strings.exitStatus == 0,
                "CPU payload dependency inspection failed");
        const auto audit = dependencies.standardOutput + symbols.standardOutput + strings.standardOutput;
        require(audit.find("libpython") == std::string::npos && audit.find("torch") == std::string::npos &&
                audit.find("Triton") == std::string::npos,
                "CPU payload contains a framework execution dependency");
#if !defined(__SANITIZE_ADDRESS__)
        // Sanitizer instrumentation intentionally records compilation-unit paths for diagnostics.
        // Ordinary qualification artifacts must remain path independent.
        require(audit.find(TH016_SOURCE) == std::string::npos &&
                audit.find(TH016_BUILD) == std::string::npos,
                "CPU payload contains a repository/build execution dependency");
#endif

        const auto floatRegion = region(compile(
            "fn addf(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{return x+y}"),
            "addf", false);
        const auto floatX = storage::Tensor::materializeF32({1}, {1.5f});
        const auto floatY = storage::Tensor::materializeF32({1}, {2.25f});
        const auto floatEntry = artifact::specializeEntry(floatRegion, {floatX, floatY});
        auto floatBuild = artifact::buildCpuAot(
            floatRegion, toolchain(), {root / "float.tha", {}, floatEntry});
        require(floatBuild.success, floatBuild.error ? floatBuild.error->message : "f32 AOT build failed");
        auto floatRun = artifact::loadAndExecuteArtifact(root / "float.tha", {floatX, floatY});
        require(floatRun.ok() &&
                std::get<storage::Tensor>(*floatRun.value).logicalF32Values() ==
                    std::vector<float>({3.75f}),
                "CPU AOT f32 semantics mismatch");

        const auto overflowRegion = region(compile(
            "fn main()->Tensor<i64,1>{let A=[9223372036854775807]\nlet B=[1]\nreturn A+B}"));
        auto overflowBuild = artifact::buildCpuAot(
            overflowRegion, toolchain(), {root / "overflow.tha", {}, {}});
        require(overflowBuild.success, "overflow artifact build failed");
        auto overflowRun = artifact::loadAndExecuteArtifact(root / "overflow.tha");
        require(!overflowRun.ok() && overflowRun.error &&
                overflowRun.error->code == "TH-SPEC-I64-OVERFLOW",
                "CPU AOT checked i64 overflow was not preserved");

        auto unfusedBuild = artifact::buildCpuAot(
            integerRegion, toolchain(), {root / "integer-unfused.tha", {false, false}, {}});
        require(unfusedBuild.success, "optimization-disabled artifact build failed");
        auto unfusedRun = artifact::loadAndExecuteArtifact(root / "integer-unfused.tha");
        require(unfusedRun.ok() && artifact::formatArtifactValue(*unfusedRun.value) ==
                artifact::formatArtifactValue(*integerRun.value) &&
                unfusedBuild.manifest->planDigest != integerBuild.manifest->planDigest,
                "planned/fused and conservative CPU AOT semantics/identity mismatch");

        const auto addRegion = region(compile(
            "fn add(x:Tensor<i64,1>,y:Tensor<i64,1>)->Tensor<i64,1>{return x+y}"),
            "add", false);
        const auto x2 = storage::Tensor::materializeI64({2}, {1, 2});
        const auto y2 = storage::Tensor::materializeI64({2}, {3, 4});
        auto addEntry = artifact::specializeEntry(addRegion, {x2, y2});
        auto addBuild = artifact::buildCpuAot(
            addRegion, toolchain(), {root / "add.tha", {}, addEntry});
        require(addBuild.success, addBuild.error ? addBuild.error->message : "typed CPU AOT build failed");
        auto addArtifact = artifact::loadArtifact(root / "add.tha");
        auto addRun = artifact::executeArtifact(*addArtifact.artifact, {x2, y2});
        require(addRun.ok() && std::get<storage::Tensor>(*addRun.value).logicalI64Values() ==
                std::vector<std::int64_t>({4, 6}), "typed CPU AOT result mismatch");
        auto wrongCount = artifact::executeArtifact(*addArtifact.artifact, {x2});
        auto wrongKind = artifact::executeArtifact(*addArtifact.artifact, {std::int64_t{1}, y2});
        const auto x3 = storage::Tensor::materializeI64({3}, {1, 2, 3});
        auto wrongShape = artifact::executeArtifact(*addArtifact.artifact, {x3, x3});
        const auto xf32 = storage::Tensor::materializeF32({2}, {1, 2});
        auto wrongDtype = artifact::executeArtifact(*addArtifact.artifact, {xf32, xf32});
        const auto matrix = storage::Tensor::materializeI64({1, 2}, {1, 2});
        auto wrongRank = artifact::executeArtifact(*addArtifact.artifact, {matrix, matrix});
        require(!wrongCount.ok() && wrongCount.error->code == "ARTIFACT-ARGUMENT-COUNT" &&
                !wrongKind.ok() && wrongKind.error->code == "ARTIFACT-INPUT-ABI" &&
                !wrongShape.ok() && wrongShape.error->code == "ARTIFACT-INPUT-ABI" &&
                !wrongDtype.ok() && wrongDtype.error->code == "ARTIFACT-INPUT-ABI" &&
                !wrongRank.ok() && wrongRank.error->code == "ARTIFACT-INPUT-ABI",
                "typed CPU AOT ABI mismatch cases were not rejected");

        const auto pickRegion = region(compile(
            "fn pick(x:Tensor<i64,1>,i:i64)->i64{return x[i]}"), "pick", false);
        auto pickEntry = artifact::specializeEntry(pickRegion, {x2, std::int64_t{1}});
        auto pickBuild = artifact::buildCpuAot(
            pickRegion, toolchain(), {root / "pick.tha", {}, pickEntry});
        require(pickBuild.success, "checked-index artifact build failed");
        auto pickGood = artifact::loadAndExecuteArtifact(
            root / "pick.tha", {x2, std::int64_t{1}});
        auto pickBad = artifact::loadAndExecuteArtifact(
            root / "pick.tha", {x2, std::int64_t{2}});
        require(pickGood.ok() && std::get<std::int64_t>(*pickGood.value) == 2 &&
                !pickBad.ok() && pickBad.error->code == "TH-SPEC-BOUNDS",
                "CPU AOT checked bounds semantics mismatch");
        const auto zero = storage::Tensor::empty(storage::DType::I64, {0});
        auto zeroEntry = artifact::specializeEntry(addRegion, {zero, zero});
        auto zeroBuild = artifact::buildCpuAot(
            addRegion, toolchain(), {root / "zero.tha", {}, zeroEntry});
        auto zeroRun = zeroBuild.success ? artifact::loadAndExecuteArtifact(root / "zero.tha", {zero, zero}) :
                                           artifact::ArtifactExecutionResult{};
        require(zeroBuild.success && zeroRun.ok() &&
                std::get<storage::Tensor>(*zeroRun.value).logicalI64Values().empty(),
                "zero-sized tensor artifact failed");

        const auto original = *loaded.artifact;
        require(rejected(original, root / "bad-version.tha",
                         [](auto& value) { value.manifest.formatVersion = 99; },
                         "ARTIFACT-FORMAT-VERSION"), "unknown format version accepted");
        require(rejected(original, root / "bad-compiler-abi.tha",
                         [](auto& value) { value.manifest.compilerAbiVersion = 99; },
                         "ARTIFACT-COMPILER-ABI"), "unknown compiler ABI accepted");
        require(rejected(original, root / "bad-runtime-abi.tha",
                         [](auto& value) { value.manifest.runtimeAbiVersion = 99; },
                         "ARTIFACT-RUNTIME-ABI"), "unknown runtime ABI accepted");
        require(rejected(original, root / "bad-backend.tha",
                         [](auto& value) { value.manifest.backend = static_cast<artifact::NativeBackend>(99); },
                         "ARTIFACT-BACKEND"), "unknown backend accepted");
        require(rejected(original, root / "bad-target.tha",
                         [](auto& value) { value.manifest.target = "other-target"; },
                         "ARTIFACT-TARGET"), "wrong target accepted");
        require(rejected(original, root / "bad-runtime-requirement.tha",
                         [](auto& value) { value.manifest.runtimeRequirement = "unknown-runtime"; },
                         "ARTIFACT-RUNTIME-REQUIREMENT"),
                "unsupported runtime requirement accepted");
        require(rejected(original, root / "bad-entry.tha",
                         [](auto& value) { value.manifest.entry.name = "other"; },
                         "ARTIFACT-ENTRY"), "wrong entry name accepted");
        require(rejected(original, root / "bad-signature.tha",
                         [](auto& value) { value.manifest.entry.result.rank = 9; },
                         "ARTIFACT-SIGNATURE"), "malformed entry signature accepted");
        require(rejected(original, root / "missing-payload.tha",
                         [](auto& value) { value.payload.clear(); value.manifest.payloadSize = 0;
                                          value.manifest.payloadDigest = artifact::digestBytes(nullptr, 0); },
                         "ARTIFACT-MISSING-PAYLOAD"), "missing payload accepted");
        require(rejected(original, root / "corrupt-payload.tha",
                         [](auto& value) { value.payload.back() ^= std::byte{1}; },
                         "ARTIFACT-PAYLOAD-INTEGRITY"), "corrupt payload accepted");
        require(rejected(original, root / "bad-plan.tha",
                         [](auto& value) { value.manifest.planning.enableFusion = false; },
                         "ARTIFACT-PLAN-INTEGRITY"), "plan/configuration mismatch accepted");
        auto truncatedBytes = readBytes(artifactPath);
        writeBytes(root / "truncated.tha", truncatedBytes, truncatedBytes.size() / 2);
        require(!artifact::loadArtifact(root / "truncated.tha").ok(), "truncated artifact accepted");
        auto corruptLength = truncatedBytes;
        for (std::size_t index = 24; index < 32; ++index) corruptLength[index] = std::byte{0xff};
        writeBytes(root / "corrupt-length.tha", corruptLength, corruptLength.size());
        auto lengthFailure = artifact::loadArtifact(root / "corrupt-length.tha");
        require(!lengthFailure.ok() && lengthFailure.error->code == "ARTIFACT-LENGTH",
                "corrupted length metadata was accepted");
        truncatedBytes[0] = std::byte{'X'};
        writeBytes(root / "bad-magic.tha", truncatedBytes, truncatedBytes.size());
        require(!artifact::loadArtifact(root / "bad-magic.tha").ok(), "bad artifact magic accepted");

        auto deterministicBuild = artifact::buildCpuAot(
            integerRegion, toolchain(), {root / "integer-again.tha", {true, true}, {}});
        require(deterministicBuild.success &&
                deterministicBuild.manifest->entry == integerBuild.manifest->entry &&
                deterministicBuild.manifest->planDigest == integerBuild.manifest->planDigest &&
                deterministicBuild.manifest->regionDigest == integerBuild.manifest->regionDigest,
                "artifact metadata generation was not deterministic");

        artifact::JitCompiler jit(toolchain());
        auto jitFirst = jit.compileCpu(addRegion, {x2, y2});
        require(jitFirst.ok() && !jitFirst.cacheHit && jit.statistics().compilations == 1 &&
                jit.statistics().cacheHits == 0 && jit.statistics().cacheMisses == 1,
                "first external-toolchain CPU JIT compilation evidence mismatch");
        auto jitRun = jitFirst.executable->execute({x2, y2});
        require(jitRun.ok() && std::get<storage::Tensor>(*jitRun.value).logicalI64Values() ==
                std::vector<std::int64_t>({4, 6}), "CPU JIT native execution mismatch");
        auto jitHit = jit.compileCpu(addRegion, {x2, y2});
        require(jitHit.ok() && jitHit.cacheHit &&
                jitHit.executable.get() == jitFirst.executable.get() &&
                jit.statistics().compilations == 1 && jit.statistics().cacheHits == 1,
                "identical CPU JIT specialization did not hit cache");
        auto jitShapeMiss = jit.compileCpu(addRegion, {x3, x3});
        require(jitShapeMiss.ok() && !jitShapeMiss.cacheHit &&
                jitShapeMiss.executable->cacheKey() != jitFirst.executable->cacheKey() &&
                jit.statistics().compilations == 2, "changed shape did not miss CPU JIT cache");
        auto jitOptionMiss = jit.compileCpu(addRegion, {x2, y2}, {true, false});
        require(jitOptionMiss.ok() && !jitOptionMiss.cacheHit &&
                jitOptionMiss.executable->cacheKey() != jitFirst.executable->cacheKey() &&
                jit.statistics().compilations == 3, "planning/fusion option did not miss CPU JIT cache");
        const auto addF32Region = region(compile(
            "fn add(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{return x+y}"),
            "add", false);
        auto jitDtypeMiss = jit.compileCpu(addF32Region, {xf32, xf32});
        require(jitDtypeMiss.ok() && !jitDtypeMiss.cacheHit &&
                jitDtypeMiss.executable->cacheKey() != jitFirst.executable->cacheKey() &&
                jit.statistics().compilations == 4, "changed dtype did not miss CPU JIT cache");
        auto jitFloatRun = jitDtypeMiss.executable->execute({xf32, xf32});
        require(jitFloatRun.ok() && std::get<storage::Tensor>(*jitFloatRun.value).logicalF32Values() ==
                std::vector<float>({2, 4}), "CPU JIT f32 semantics mismatch");

        const auto overflowDynamic = region(compile(
            "fn add(x:Tensor<i64,1>,y:Tensor<i64,1>)->Tensor<i64,1>{return x+y}"),
            "add", false);
        const auto maximum = storage::Tensor::materializeI64(
            {1}, {std::numeric_limits<std::int64_t>::max()});
        const auto one = storage::Tensor::materializeI64({1}, {1});
        auto overflowJit = jit.compileCpu(overflowDynamic, {maximum, one});
        auto overflowJitRun = overflowJit.executable->execute({maximum, one});
        require(!overflowJitRun.ok() && overflowJitRun.error &&
                overflowJitRun.error->code == "TH-SPEC-I64-OVERFLOW",
                "CPU JIT checked overflow was not preserved");
        auto wrongJitInput = jitFirst.executable->execute({x3, x3});
        require(!wrongJitInput.ok() && wrongJitInput.error->code == "ARTIFACT-INPUT-ABI",
                "JIT executable accepted incompatible specialization");
        auto invalidRegion = addRegion;
        invalidRegion.output = 999;
        auto invalidJit = jit.compileCpu(invalidRegion, {x2, y2});
        require(!invalidJit.ok() && invalidJit.error->code == "JIT-INVALID-INPUT",
                "malformed JIT input was accepted");
        artifact::NativeToolchain missingCompiler = toolchain();
        missingCompiler.compilerExecutable = root / "missing-cxx";
        artifact::JitCompiler failingJit(missingCompiler);
        auto compilerFailure = failingJit.compileCpu(addRegion, {x2, y2});
        require(!compilerFailure.ok() && compilerFailure.error->code == "JIT-NATIVE-COMPILER",
                "native compiler JIT failure was not surfaced");
        std::weak_ptr<artifact::JitExecutable> releasedExecutable;
        {
            artifact::JitCompiler lifetimeJit(toolchain());
            auto compiled = lifetimeJit.compileCpu(addRegion, {x2, y2});
            require(compiled.ok(), "JIT lifetime fixture did not compile");
            releasedExecutable = compiled.executable;
        }
        require(releasedExecutable.expired(),
                "JIT cache/executable lifetime retained a native handle after owner destruction");

        const auto cpuKey = artifact::JitCompiler::cacheIdentity(
            addRegion, artifact::NativeBackend::Cpu, addEntry, {});
        require(cpuKey == artifact::JitCompiler::cacheIdentity(
                    addRegion, artifact::NativeBackend::Cpu, addEntry, {}) &&
                cpuKey != artifact::JitCompiler::cacheIdentity(
                    addRegion, artifact::NativeBackend::Gpu, addEntry, {}),
                "JIT cache identity is nondeterministic or backend-blind");

        std::cout << "NativeArtifactTests PASS " << checks << " checks\n"
                  << "cpu_artifact_bytes=" << fs::file_size(artifactPath) << '\n'
                  << "cpu_payload_bytes=" << loaded.artifact->manifest.payloadSize << '\n'
                  << "jit_compilations=" << jit.statistics().compilations
                  << " cache_hits=" << jit.statistics().cacheHits
                  << " cache_misses=" << jit.statistics().cacheMisses << '\n'
                  << "fallback=NONE\n";
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "NativeArtifactTests FAIL: " << failure.what() << '\n';
        std::error_code ignored;
        fs::remove_all(root, ignored);
        return 1;
    }
}
