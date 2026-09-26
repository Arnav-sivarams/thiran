#include "analysis/v0/Ownership.hpp"
#include "backend/v0/NativeCpu.hpp"
#include "backend/v0/NativeGpu.hpp"
#include "extension/v0/Extension.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace analysis = thiran::v0::analysis;
namespace backend = thiran::v0::backend;
namespace extension = thiran::v0::extension;
namespace semantic = thiran::v0::semantic;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

backend::TensorRegion region(const std::string& source, backend::NativeTarget target,
                             extension::ExtensionRegistry* registry = nullptr) {
    auto parsed = thiran::v0::parse(source, "<th023-structure>");
    require(parsed.module.has_value() && parsed.diagnostics.empty(), "parse failed");
    auto analyzed = registry ? semantic::analyze(*parsed.module, *registry) :
                               semantic::analyze(*parsed.module);
    require(analyzed.module.has_value() && analyzed.diagnostics.empty(), "analysis failed");
    auto ownership = analysis::analyze(*analyzed.module);
    require(ownership.ok(), "ownership analysis failed");
    auto extracted = backend::extractStrictRegion(*analyzed.module, ownership, "bench", false, target);
    require(extracted.ok(), "strict TensorRegion extraction failed: " + extracted.diagnostic);
    return std::move(*extracted.region);
}

std::size_t materializedIntermediates(const backend::PhysicalPlan& plan) {
    return static_cast<std::size_t>(std::count_if(plan.values.begin(), plan.values.end(),
        [](const auto& value) {
            return value.value == value.root && value.materialized &&
                   value.classification == backend::PhysicalValueClass::Temporary;
        }));
}

} // namespace

int main() {
    try {
        const std::string linear =
            "fn bench(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{"
            "let a=x+y;let b=-a;let c=b.*y;let d=c-x;let e=d.*y;return e}";
        const auto linearRegion = region(linear, backend::NativeTarget::Cpu);
        auto fused = backend::buildPhysicalPlan(linearRegion, backend::PhysicalDevice::Host,
                                                 {true, true});
        auto unfused = backend::buildPhysicalPlan(linearRegion, backend::PhysicalDevice::Host,
                                                   {false, false});
        require(fused.ok() && unfused.ok(), "linear physical planning failed");
        require(fused.plan->fusionGroups.size() == 1 && fused.plan->fusionGroups[0].fused &&
                fused.plan->fusionGroups[0].nodes.size() == 5,
                "linear workload no longer forms one five-op fusion group");
        require(unfused.plan->fusionGroups.size() == 5 &&
                std::none_of(unfused.plan->fusionGroups.begin(), unfused.plan->fusionGroups.end(),
                             [](const auto& group) { return group.fused; }),
                "disabled fusion no longer emits five groups");
        require(materializedIntermediates(*fused.plan) == 1 &&
                materializedIntermediates(*unfused.plan) == 5,
                "linear materialization structure changed");

        const std::string branched =
            "fn bench(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{"
            "let a=x+y;let b=-a;let c=a.*y;let d=b+c;let e=d-x;return e}";
        const auto branchedRegion = region(branched, backend::NativeTarget::Cpu);
        auto branchedPlan = backend::buildPhysicalPlan(branchedRegion, backend::PhysicalDevice::Host,
                                                        {true, true});
        require(branchedPlan.ok() && branchedPlan.plan->fusionGroups.size() == 3 &&
                materializedIntermediates(*branchedPlan.plan) == 3,
                "multiple-consumer materialization barrier changed");

        extension::ExtensionRegistry registry;
        auto loaded = registry.load(TH023_EXTENSION);
        require(loaded.ok(), "TH-021 fixture failed to load");
        const std::string extensionSource =
            "fn bench(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{"
            "let a=-x;let b=research_square_linear(a);let c=b+y;return c}";
        const auto extensionRegion = region(extensionSource, backend::NativeTarget::Gpu, &registry);
        auto extensionPlan = backend::buildPhysicalPlan(extensionRegion, backend::PhysicalDevice::Gpu,
                                                         {true, true});
        require(extensionPlan.ok() && extensionPlan.plan->fusionGroups.size() == 1 &&
                extensionPlan.plan->fusionGroups[0].fused,
                "generic extension no longer fuses into one GPU group");
        const auto cpp = backend::emitCpp20(extensionRegion, false);
        const auto ptx = backend::emitNativeGpuPtx(extensionRegion);
        require(cpp.find("evaluateRecipe") == std::string::npos &&
                cpp.find("ExtensionRegistry") == std::string::npos &&
                cpp.find("dlopen") == std::string::npos &&
                ptx.find("research_square_linear") == std::string::npos,
                "extension hot path regained runtime registry/callback dispatch");

        std::cout << "TH-023 structural qualification PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TH-023 structural qualification FAIL: " << error.what() << '\n';
        return 1;
    }
}
