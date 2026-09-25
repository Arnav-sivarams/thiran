#include "backend/v0/NativeGpu.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Verifier.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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
    auto parsed = parse(source, "<th013-test>");
    require(parsed.module.has_value(), (parsed.diagnostics.empty() ? "parse failed" : parsed.diagnostics.front().format()) +
            "\nSOURCE:\n" + source);
    auto analyzed = semantic::analyze(*parsed.module);
    require(analyzed.module.has_value(), analyzed.diagnostics.empty() ? "analysis failed" : analyzed.diagnostics.front().format());
    require(semantic::verify(*analyzed.module).ok, "semantic verifier failed");
    return *analyzed.module;
}

backend::NativeResult extract(const semantic::Module& module, const std::string& name = "main",
                              bool standalone = true) {
    auto facts = analysis::analyze(module);
    require(facts.ok(), facts.diagnostics.empty() ? "ownership/effect analysis failed" :
            "ownership/effect analysis failed: " + facts.diagnostics.front().category + " " +
            facts.diagnostics.front().message);
    return backend::extractStrictGpu(module, facts, name, standalone);
}

}

int main() {
    try {
        const auto integer = compile(
            "fn main() -> Tensor<i64,1> {\n"
            "let A=[1,2,3,4,5]\nlet B=[6,7,8,9,10]\n"
            "let C=A+B\nlet D=C.*B\nreturn D-A\n}");
        const auto floating = compile(
            "fn f(A:Tensor<f32,1>,B:Tensor<f32,1>) -> Tensor<f32,1> {\n"
            "let C=A+B\nreturn -(C.*B)\n}");
        const auto index = compile(
            "fn pick(x: Tensor<i64,2>, i: i64) -> i64 { return x[i,1] }");
        const auto copied = compile(
            "fn copied(x:Tensor<i64,1>,y:Tensor<i64,1>)->Tensor<i64,1>{let z=copy(x)\nreturn z+y}");

        const auto integerRegion = extract(integer);
        const auto floatingRegion = extract(floating, "f", false);
        const auto indexRegion = extract(index, "pick", false);
        const auto copyRegion = extract(copied, "copied", false);
        require(integerRegion.ok(), "i64 GPU extraction failed: " + integerRegion.diagnostic);
        require(floatingRegion.ok(), "f32 GPU extraction failed: " + floatingRegion.diagnostic);
        require(indexRegion.ok(), "i64 Index GPU extraction failed: " + indexRegion.diagnostic);
        require(copyRegion.ok() && copyRegion.coverage.find("Copy native-gpu") != std::string::npos,
                "explicit Copy GPU extraction failed: " + copyRegion.diagnostic);
        const auto copiedPlan = backend::buildPhysicalPlan(*copyRegion.region,
            backend::PhysicalDevice::Gpu);
        require(copiedPlan.ok(), "explicit Copy GPU planning failed");
        const auto copyNode = std::find_if(copyRegion.region->nodes.begin(), copyRegion.region->nodes.end(),
            [](const backend::RegionNode& node) { return node.op == backend::RegionOp::Copy; });
        require(copyNode != copyRegion.region->nodes.end() && copiedPlan.plan->value(copyNode->id) &&
                copiedPlan.plan->value(copyNode->id)->root == copyNode->id,
                "explicit Copy did not retain an independent physical root");
        require(integerRegion.coverage.find("Add native-gpu") != std::string::npos &&
                integerRegion.coverage.find("ElementMultiply native-gpu") != std::string::npos &&
                integerRegion.coverage.find("Subtract native-gpu") != std::string::npos &&
                integerRegion.coverage.find("fallback: NONE") != std::string::npos,
                "coherent GPU coverage missing");
        require(floatingRegion.coverage.find("Negate native-gpu") != std::string::npos,
                "f32 Negate coverage missing");
        require(integerRegion.region->dump() == extract(integer).region->dump(),
                "GPU TensorRegion extraction is nondeterministic");

        const auto matmul = extract(compile(
            "fn main() -> Tensor<i64,2> { let A=[1,2;3,4]\nreturn A*A\n}"));
        require(!matmul.ok() && matmul.diagnostic.find("MatMul lowering deferred") != std::string::npos &&
                matmul.coverage.find("fallback: NONE") != std::string::npos,
                "MatMul did not reject explicitly");
        const auto reduction = extract(compile(
            "fn main() -> Tensor<i64,1> { let A=[1,2;3,4]\nreturn sum(A,0)\n}"));
        require(!reduction.ok() && reduction.diagnostic.find("Sum lowering deferred") != std::string::npos,
                "Sum did not reject explicitly");
        const auto view = extract(compile(
            "fn main() -> Tensor<i64,2> { let A=[1,2;3,4]\nlet B=A.T\nreturn B+B\n}"));
        require(!view.ok() && view.diagnostic.find("Transpose lowering deferred") != std::string::npos,
                "Transpose did not reject explicitly");
        const auto structured = extract(compile(
            "fn main() -> i64 { if true { return 1 } else { return 2 }\n}"));
        require(!structured.ok() && structured.diagnostic.find("BACKEND-UNSUPPORTED") != std::string::npos &&
                structured.coverage.find("fallback: NONE") != std::string::npos,
                "structured control did not remain outside TensorRegion: " + structured.diagnostic);
        const auto scan = extract(compile(
            "fn step(x:i64,state:i64)->(i64,i64){return(x,state)}\n"
            "fn run(xs:Tensor<i64,1>,state:i64)->(Tensor<i64,1>,i64){return scan(step,xs,state)}"),
            "run", false);
        require(!scan.ok() && scan.diagnostic.find("Scan lowering deferred") != std::string::npos &&
                scan.coverage.find("Scan unsupported-native-gpu") != std::string::npos &&
                scan.coverage.find("fallback: NONE") != std::string::npos,
                "Structured::Scan was not explicitly GPU-unsupported");

        backend::TensorRegion malformed = *integerRegion.region;
        malformed.nodes.back().op = backend::RegionOp::Unsupported;
        require(!backend::verifyRegion(malformed).ok(), "malformed lower-level GPU region was admitted");

        const auto f32Tensor = storage::Tensor::materializeF32({3}, {1.25f, -2.5f, 0.0f});
        require(f32Tensor.descriptor().dtype == storage::DType::F32 &&
                f32Tensor.logicalF32Values() == std::vector<float>({1.25f, -2.5f, 0.0f}),
                "f32 host storage round trip failed");
        const auto f32Alias = f32Tensor;
        require(f32Alias.storageId() == f32Tensor.storageId(), "immutable host alias did not share storage");
        const auto f32Copy = f32Tensor.deepCopy();
        require(f32Copy.storageId() != f32Tensor.storageId() &&
                f32Copy.logicalF32Values() == f32Tensor.logicalF32Values(),
                "explicit f32 deep copy did not create independent storage");
        require(storage::Tensor::empty(storage::DType::F32, {0}).logicalF32Values().empty(),
                "zero-size f32 tensor failed");
        bool overflow = false;
        try { (void)storage::checkedByteCount(std::numeric_limits<std::uint64_t>::max(), storage::DType::F64); }
        catch (const std::runtime_error& error) { overflow = std::string(error.what()) == "TH007-SIZE-OVERFLOW"; }
        require(overflow, "byte-count overflow was not rejected");

        const auto gpuPlan = backend::buildPhysicalPlan(*integerRegion.region,
            backend::PhysicalDevice::Gpu);
        const auto gpuConservative = backend::buildPhysicalPlan(*integerRegion.region,
            backend::PhysicalDevice::Gpu, {true, false});
        require(gpuPlan.ok() && gpuConservative.ok(), "GPU physical planning failed");
        require(gpuPlan.plan->fusionGroups.size() == 1 &&
                gpuPlan.plan->fusionGroups.front().nodes.size() == 3 &&
                gpuConservative.plan->fusionGroups.size() == 3,
                "GPU fusion enabled/disabled group structure is wrong");
        require(gpuPlan.plan->dump() == backend::buildPhysicalPlan(*integerRegion.region,
                    backend::PhysicalDevice::Gpu).plan->dump(),
                "GPU physical plan dump is nondeterministic");

        const auto probe = backend::probeNativeGpu();
        require(probe.backendBuilt == backend::nativeGpuBackendBuilt(), "GPU build discovery mismatch");
        require((backend::nativeGpuAsyncEffects() &
                 static_cast<analysis::EffectSet>(analysis::EffectKind::Async)) != 0 &&
                (backend::nativeGpuAsyncEffects() &
                 static_cast<analysis::EffectSet>(analysis::EffectKind::Transfer)) != 0,
                "native async API is disconnected from Async/Transfer effects");
        if (backend::nativeGpuBackendBuilt()) {
            const auto ptx = backend::emitNativeGpuPtx();
            require(ptx == backend::emitNativeGpuPtx() &&
                    ptx.find(".target sm_50") != std::string::npos &&
                    ptx.find("thiran_add_i64") != std::string::npos &&
                    ptx.find("thiran_mul_f32") != std::string::npos &&
                    ptx.find("thiran_index_i64") != std::string::npos &&
                    ptx.find(".approx") == std::string::npos && ptx.find(".ftz") == std::string::npos,
                    "deterministic/default-numeric PTX module audit failed");
            const auto plannedPtx = backend::emitNativeGpuPtx(*integerRegion.region);
            const auto unfusedPtx = backend::emitNativeGpuPtx(*integerRegion.region, {true, false});
            require(plannedPtx == backend::emitNativeGpuPtx(*integerRegion.region) &&
                    plannedPtx.find("thiran_group_1_i64") != std::string::npos &&
                    plannedPtx.find("add.s64") < plannedPtx.find("mul.lo.s64", plannedPtx.find("thiran_group_1_i64")) &&
                    plannedPtx.find(".approx") == std::string::npos &&
                    plannedPtx.find(".ftz") == std::string::npos,
                    "fused checked/default-numeric PTX audit failed");
            require(unfusedPtx.find("thiran_group_3_i64") != std::string::npos,
                    "optimization-disabled PTX did not retain separate kernel groups");
            const auto arity = backend::executeNativeGpu(*indexRegion.region, {});
            require(!arity.ok() && arity.error && arity.error->code == "GPU-INPUT-ARITY" &&
                    arity.evidence.kernelLaunches == 0,
                    "GPU input arity was not rejected before device initialization");
            const auto matrix = storage::Tensor::materializeI64({2, 2}, {1, 2, 3, 4});
            const auto layout = backend::executeNativeGpu(*indexRegion.region,
                                                          {matrix.transpose(), std::int64_t{0}});
            require(!layout.ok() && layout.error && layout.error->code == "GPU-UNSUPPORTED-LAYOUT" &&
                    layout.evidence.kernelLaunches == 0,
                    "GPU view/layout was not rejected before device initialization");
            const auto inputType = backend::executeNativeGpu(*indexRegion.region,
                                                             {matrix, 0.0f});
            require(!inputType.ok() && inputType.error && inputType.error->code == "GPU-INPUT-TYPE",
                    "GPU scalar input dtype mismatch was not explicit");
            const auto invalidRegion = backend::executeNativeGpu(malformed);
            require(!invalidRegion.ok() && invalidRegion.error &&
                    invalidRegion.error->code == "GPU-INVALID-REGION" &&
                    invalidRegion.evidence.kernelLaunches == 0,
                    "malformed GPU region reached device execution");
            const auto invalidAsync = backend::submitNativeGpuAsync(malformed);
            require(!invalidAsync.ok() && invalidAsync.error &&
                    invalidAsync.error->code == "GPU-INVALID-REGION" &&
                    invalidAsync.evidence.submittedOperations == 0,
                    "malformed async GPU region acquired lifetime obligations");
        }
        if (!backend::nativeGpuBackendBuilt()) {
            require(backend::emitNativeGpuPtx().empty(), "disabled GPU build emitted a device module");
            require(backend::emitNativeGpuPtx(*integerRegion.region).empty(),
                    "disabled GPU build emitted a planned device module");
            require(probe.error && probe.error->code == "GPU-BACKEND-NOT-BUILT",
                    "disabled GPU backend did not report explicit unavailability");
            const auto execution = backend::executeNativeGpu(*integerRegion.region);
            require(!execution.ok() && execution.error && execution.error->code == "GPU-BACKEND-NOT-BUILT" &&
                    execution.evidence.kernelLaunches == 0,
                    "disabled GPU request did not fail without fallback");
            const auto asyncExecution = backend::submitNativeGpuAsync(*integerRegion.region);
            require(!asyncExecution.ok() && asyncExecution.error &&
                    asyncExecution.error->code == "GPU-BACKEND-NOT-BUILT" &&
                    asyncExecution.evidence.submittedOperations == 0,
                    "disabled async GPU request did not fail without reservations/fallback");
        } else if (!probe.deviceAvailable) {
            const auto execution = backend::executeNativeGpu(*integerRegion.region);
            require(!execution.ok() && execution.error &&
                    execution.error->category == backend::GpuErrorCategory::BackendUnavailable &&
                    execution.evidence.kernelLaunches == 0 && !execution.value,
                    "unavailable GPU request executed or fell back");
        }

        std::cout << "V0NativeGpuTests PASS " << checks << " checks\n"
                  << "backend_built=" << backend::nativeGpuBackendBuilt() << '\n'
                  << "probe=" << (probe.error ? probe.error->code : "available") << '\n'
                  << "fallback=NONE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0NativeGpuTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
