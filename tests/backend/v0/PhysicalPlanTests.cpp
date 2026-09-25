#include "autodiff/v0/Autodiff.hpp"
#include "backend/v0/PhysicalPlan.hpp"
#include "frontend/v0/Parser.hpp"
#include "runtime/v0/Async.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "storage/v0/Storage.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace thiran::v0;
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

semantic::Type type(semantic::TypeKind dtype = semantic::TypeKind::I64, std::uint32_t rank = 1) {
    return semantic::tensor(semantic::scalar(dtype), rank);
}

backend::RegionNode tensorNode(semantic::ValueId id, backend::RegionOp op,
                               std::vector<semantic::ValueId> dependencies = {},
                               semantic::ShapeFact shape = {{4}},
                               semantic::Type nodeType = type()) {
    backend::RegionNode node;
    node.id = id;
    node.op = op;
    node.type = std::move(nodeType);
    node.shape = std::move(shape);
    node.dependencies = std::move(dependencies);
    node.provenance = op == backend::RegionOp::Alias ? analysis::ProvenanceKind::AliasOf :
        op == backend::RegionOp::Copy ? analysis::ProvenanceKind::IndependentCopyOf :
        analysis::ProvenanceKind::Fresh;
    return node;
}

backend::TensorRegion linear(bool longer = true, semantic::TypeKind dtype = semantic::TypeKind::I64) {
    backend::TensorRegion region;
    region.function = 1;
    region.name = "linear";
    const auto tensorType = type(dtype);
    region.nodes = {
        tensorNode(1, backend::RegionOp::Input, {}, {{4}}, tensorType),
        tensorNode(2, backend::RegionOp::Input, {}, {{4}}, tensorType),
        tensorNode(3, backend::RegionOp::Add, {1, 2}, {{4}}, tensorType),
        tensorNode(4, backend::RegionOp::ElementMultiply, {3, 2}, {{4}}, tensorType),
        tensorNode(5, backend::RegionOp::Subtract, {4, 1}, {{4}}, tensorType)
    };
    if (longer) region.nodes.push_back(tensorNode(6, backend::RegionOp::Add, {5, 2}, {{4}}, tensorType));
    region.inputs = {1, 2};
    region.output = longer ? 6 : 5;
    region.outputType = tensorType;
    return region;
}

backend::TensorRegion branch() {
    auto region = linear(false);
    region.name = "branch";
    region.nodes = {
        tensorNode(1, backend::RegionOp::Input),
        tensorNode(2, backend::RegionOp::Input),
        tensorNode(3, backend::RegionOp::Add, {1, 2}),
        tensorNode(4, backend::RegionOp::Subtract, {3, 1}),
        tensorNode(5, backend::RegionOp::ElementMultiply, {3, 2}),
        tensorNode(6, backend::RegionOp::Add, {4, 5})
    };
    region.output = 6;
    return region;
}

backend::PhysicalPlan plan(const backend::TensorRegion& region,
                           backend::PhysicalPlanOptions options = {},
                           const backend::PhysicalPlanningObligations& obligations = {},
                           backend::PhysicalDevice device = backend::PhysicalDevice::Host) {
    auto result = backend::buildPhysicalPlan(region, device, options, obligations);
    require(result.ok(), result.errors.empty() ? "planning failed" : result.errors.front());
    return *result.plan;
}

const backend::PhysicalValuePlan& value(const backend::PhysicalPlan& plan, semantic::ValueId id) {
    const auto* result = plan.value(id);
    require(result != nullptr, "missing planned value v" + std::to_string(id));
    return *result;
}

bool has(const backend::PhysicalPlanVerification& verification, const std::string& code) {
    for (const auto& error : verification.errors) if (error.find(code) != std::string::npos) return true;
    return false;
}

void memoryPlanning() {
    const backend::PhysicalPlanOptions unfused{true, false};
    const auto region = linear();
    const auto planned = plan(region, unfused);
    require(value(planned, 3).slot && value(planned, 5).slot &&
            value(planned, 3).slot == value(planned, 5).slot,
            "linear non-overlapping temporaries did not reuse a slot");
    require(value(planned, 3).lifetime.lastObligation < value(planned, 5).lifetime.definition,
            "linear reuse lacks non-overlap evidence");
    require(value(planned, 4).slot != value(planned, 3).slot,
            "overlapping linear temporaries shared storage");
    require(value(planned, 6).classification == backend::PhysicalValueClass::Output &&
            !value(planned, 6).reusable && value(planned, 6).slot != value(planned, 5).slot,
            "output storage was prematurely reusable");
    require(!value(planned, 1).reusable &&
            value(planned, 1).classification == backend::PhysicalValueClass::External,
            "external input was reusable");

    const auto branching = plan(branch(), unfused);
    require(value(branching, 4).slot != value(branching, 5).slot,
            "branching interfering values shared storage");
    require(value(branching, 3).lifetime.lastObligation == 4,
            "multiple-consumer lifetime did not reach final consumer");

    auto aliasRegion = linear(false);
    aliasRegion.nodes.insert(aliasRegion.nodes.begin() + 3,
                             tensorNode(7, backend::RegionOp::Alias, {3}));
    aliasRegion.nodes[4].dependencies = {7, 2};
    const auto aliases = plan(aliasRegion, unfused);
    require(value(aliases, 7).root == value(aliases, 3).root &&
            value(aliases, 7).slot == value(aliases, 3).slot,
            "immutable alias did not use its root storage facts");

    backend::PhysicalPlanningObligations viewFacts;
    viewFacts.viewRoots[7] = 3;
    const auto views = plan(aliasRegion, unfused, viewFacts);
    require(value(views, 7).classification == backend::PhysicalValueClass::View &&
            !value(views, 3).reusable && value(views, 3).lifetime.lastObligation == aliasRegion.nodes.size(),
            "view/root obligation did not protect root storage");

    auto copyRegion = linear(false);
    copyRegion.nodes.insert(copyRegion.nodes.begin() + 3,
                            tensorNode(7, backend::RegionOp::Copy, {3}));
    copyRegion.nodes[4].dependencies = {7, 2};
    const auto copies = plan(copyRegion, unfused);
    require(value(copies, 7).root == 7 && value(copies, 7).root != value(copies, 3).root,
            "explicit copy was not physically independent");

    auto sizes = linear();
    sizes.nodes[4].shape = {{8}};
    sizes.nodes[4].type = type();
    // Keep the region valid by making its operands and successor the other size.
    sizes.nodes[3].shape = {{8}};
    sizes.nodes[0].shape = {{8}};
    sizes.nodes[1].shape = {{8}};
    sizes.nodes[2].shape = {{8}};
    sizes.nodes[5].shape = {{8}};
    const auto equalSized = plan(sizes, unfused);
    require(value(equalSized, 3).slot == value(equalSized, 5).slot,
            "equal-size conservative reuse unexpectedly failed\n" + equalSized.dump());
    auto incompatible = linear();
    incompatible.nodes.push_back(tensorNode(7, backend::RegionOp::Input, {}, {{2}}));
    incompatible.inputs.push_back(7);
    const auto incompatiblePlan = plan(incompatible, unfused);
    require(value(incompatiblePlan, 7).slot != value(incompatiblePlan, 1).slot,
            "incompatible shape shared an external slot");

    backend::TensorRegion overflow;
    overflow.function = 2;
    overflow.name = "overflow";
    overflow.outputType = type(semantic::TypeKind::I64, 2);
    overflow.nodes = {tensorNode(1, backend::RegionOp::Input, {},
        {{std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max()}},
        overflow.outputType)};
    overflow.inputs = {1};
    overflow.output = 1;
    require(!backend::buildPhysicalPlan(overflow, backend::PhysicalDevice::Host).ok(),
            "byte-count overflow was accepted");

    auto zeroRegion = linear(false);
    for (auto& node : zeroRegion.nodes) node.shape = {{0}};
    const auto zero = plan(zeroRegion);
    require(!value(zero, zeroRegion.output).slot &&
            value(zero, zeroRegion.output).classification == backend::PhysicalValueClass::ZeroSize,
            "zero-size value received a physical allocation");
    require(planned.dump() == plan(region, unfused).dump(), "memory planning dump is nondeterministic");

    backend::PhysicalPlanningObligations protectedFacts;
    protectedFacts.savedForBackward.insert(3);
    auto saved = plan(region, unfused, protectedFacts);
    require(value(saved, 3).classification == backend::PhysicalValueClass::SavedForBackward &&
            !value(saved, 3).reusable && value(saved, 3).lifetime.lastObligation == region.nodes.size(),
            "AD saved value was not protected");
    protectedFacts = {};
    protectedFacts.parameters.insert(3);
    auto parameter = plan(region, unfused, protectedFacts);
    require(value(parameter, 3).classification == backend::PhysicalValueClass::Parameter &&
            !value(parameter, 3).reusable, "parameter storage was reusable");
    protectedFacts = {};
    protectedFacts.trainingState.insert(3);
    auto state = plan(region, unfused, protectedFacts);
    require(value(state, 3).classification == backend::PhysicalValueClass::TrainingState &&
            !value(state, 3).reusable, "training state was reusable");

    auto malformed = planned;
    malformed.values[2].slot = 999;
    require(has(backend::verifyPhysicalPlan(region, malformed), "MPV08"), "malformed slot ID was accepted");
    malformed = planned;
    malformed.values[3].slot = malformed.values[2].slot;
    if (malformed.values[3].slot) malformed.slots[*malformed.values[3].slot - 1].values.push_back(4);
    require(has(backend::verifyPhysicalPlan(region, malformed), "MPV20"),
            "overlapping assignment was accepted");
    malformed = planned;
    const auto slotId = *value(malformed, 3).slot;
    malformed.slots[slotId - 1].capacityBytes = 1;
    require(has(backend::verifyPhysicalPlan(region, malformed), "MPV19"),
            "insufficient capacity was accepted");
    malformed = planned;
    malformed.values.back().slot.reset();
    require(has(backend::verifyPhysicalPlan(region, malformed), "MPV15"),
            "missing physical assignment was accepted");
    malformed = planned;
    const auto outputSlot = *malformed.values.back().slot;
    malformed.values.back().reusable = true;
    malformed.slots[outputSlot - 1].reusable = true;
    require(has(backend::verifyPhysicalPlan(region, malformed), "MPV16") ||
            has(backend::verifyPhysicalPlan(region, malformed), "MPV17"),
            "reusable output assignment was accepted");
    malformed = planned;
    malformed.slots.front().device = backend::PhysicalDevice::Gpu;
    require(has(backend::verifyPhysicalPlan(region, malformed), "MPV04"),
            "device-mismatched slot was accepted");
    malformed = planned;
    malformed.values[2].requirement.shape = {{8}};
    require(has(backend::verifyPhysicalPlan(region, malformed), "MPV11"),
            "incompatible layout/shape requirement was accepted");
    malformed = planned;
    malformed.slots[*malformed.values[2].slot - 1].layout = backend::PhysicalLayout::Strided;
    require(has(backend::verifyPhysicalPlan(region, malformed), "MPV19"),
            "incompatible physical layout was accepted");
}

runtime::AsyncSubmission reserve(const runtime::AsyncResource& resource,
                                  runtime::AsyncReservationKind kind) {
    runtime::AsyncBackendCallbacks callbacks;
    callbacks.submit = [] { return std::optional<runtime::AsyncError>{}; };
    callbacks.poll = [] { return runtime::AsyncPollResult{runtime::AsyncPollState::Completed, {}}; };
    callbacks.wait = [] { return std::optional<runtime::AsyncError>{}; };
    callbacks.abort = [] { return std::optional<runtime::AsyncError>{}; };
    return runtime::submitAsyncOperation({{resource, kind}}, {}, std::move(callbacks));
}

void asyncPlanning() {
    const auto region = linear();
    const backend::PhysicalPlanOptions unfused{true, false};
    auto tensor = storage::Tensor::materializeI64({1}, {1});
    auto read = reserve(tensor.asyncResource(), runtime::AsyncReservationKind::Read);
    require(read.ok(), "async read fixture failed");
    backend::PhysicalPlanningObligations obligations;
    backend::addActiveAsyncObligation(obligations, 3, tensor.asyncResource());
    auto pendingRead = plan(region, unfused, obligations);
    require(value(pendingRead, 3).classification == backend::PhysicalValueClass::AsyncReserved &&
            !value(pendingRead, 3).reusable, "pending read did not block reuse");
    require(read.operation->state() == runtime::AsyncOperationState::Completed &&
            tensor.asyncResource().activeReads() == 1,
            "completed-but-unobserved read lost its reservation");
    obligations = {};
    backend::addActiveAsyncObligation(obligations, 3, tensor.asyncResource());
    require(!value(plan(region, unfused, obligations), 3).reusable,
            "completed-but-unobserved reservation allowed reuse");
    require(read.operation->observe().success && tensor.asyncResource().activeReads() == 0,
            "read observation did not release reservation");
    obligations = {};
    backend::addActiveAsyncObligation(obligations, 3, tensor.asyncResource());
    require(value(plan(region, unfused, obligations), 3).reusable,
            "reuse did not become legal after observation");

    auto write = reserve(tensor.asyncResource(), runtime::AsyncReservationKind::Write);
    require(write.ok(), "async write fixture failed");
    obligations = {};
    backend::addActiveAsyncObligation(obligations, 3, tensor.asyncResource());
    auto pendingWrite = plan(region, unfused, obligations);
    require(value(pendingWrite, 3).classification == backend::PhysicalValueClass::AsyncReserved &&
            !value(pendingWrite, 3).reusable, "pending write did not block reuse");
    require(write.operation->observe().success, "async write observation failed");
}

void fusionPlanning() {
    const auto region = linear();
    const auto fused = plan(region);
    require(fused.fusionGroups.size() == 1 && fused.fusionGroups[0].nodes.size() == 4 &&
            fused.fusionGroups[0].fused, "eligible elementwise chain did not fuse");
    require(!value(fused, 3).materialized && !value(fused, 4).materialized &&
            !value(fused, 5).materialized && value(fused, 6).materialized,
            "fused materialization boundaries are wrong");
    auto twoOperations = linear(false);
    twoOperations.nodes.pop_back();
    twoOperations.output = 4;
    require(plan(twoOperations).fusionGroups.size() == 1 &&
            plan(twoOperations).fusionGroups.front().nodes.size() == 2,
            "two eligible elementwise operations did not fuse");
    require(plan(linear(false)).fusionGroups[0].nodes.size() == 3,
            "short eligible chain did not fuse");
    const auto branching = plan(branch());
    require(branching.fusionGroups.size() >= 3,
            "branching values were illegally fused into one group");

    backend::PhysicalPlanningObligations observed;
    observed.externallyObserved.insert(4);
    const auto splitObserved = plan(region, {}, observed);
    require(splitObserved.fusionGroups.size() > 1 && value(splitObserved, 4).materialized,
            "externally observed intermediate did not materialize");
    observed = {};
    observed.fusionBarriers.insert(4);
    const auto effectSplit = plan(region, {}, observed);
    require(effectSplit.fusionGroups.size() > 1,
            "effect/mutation/async observation barrier did not split fusion");

    auto copyRegion = region;
    copyRegion.nodes.insert(copyRegion.nodes.begin() + 3,
                            tensorNode(7, backend::RegionOp::Copy, {3}));
    copyRegion.nodes[4].dependencies = {7, 2};
    const auto copySplit = plan(copyRegion);
    require(copySplit.fusionGroups.size() > 1, "unsupported Copy boundary did not split fusion");

    auto incompatibleShape = region;
    incompatibleShape.nodes[3].shape = {{5}};
    require(!backend::buildPhysicalPlan(incompatibleShape, backend::PhysicalDevice::Host).ok(),
            "incompatible shape entered fusion");
    auto incompatibleDtype = region;
    incompatibleDtype.nodes[3].type = type(semantic::TypeKind::F32);
    require(!backend::buildPhysicalPlan(incompatibleDtype, backend::PhysicalDevice::Host).ok(),
            "incompatible dtype entered fusion");

    const auto floating = plan(linear(true, semantic::TypeKind::F32));
    require(floating.fusionGroups.size() == 1 &&
            floating.fusionGroups.front().type == type(semantic::TypeKind::F32),
            "f32 chain did not retain dtype/evaluation order in its group");
    require(fused.dump() == plan(region).dump(), "fusion formation is nondeterministic");
    auto malformed = fused;
    malformed.fusionGroups.front().output = 3;
    require(has(backend::verifyPhysicalPlan(region, malformed), "FPV01") ||
            has(backend::verifyPhysicalPlan(region, malformed), "FPV03"),
            "malformed fusion group was accepted");

    const auto disabled = plan(region, {false, false});
    require(disabled.fusionGroups.size() == 4 && disabled.slots.size() > fused.slots.size(),
            "optimization-disabled conservative plan was not distinct");
}

void actualAdSaves() {
    auto parsed = parse(
        "fn f(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{"
        "let z=x.*y\nreturn z.*x}", "<th015-ad-save>");
    require(parsed.module.has_value(), "AD-save fixture parse failed");
    auto analyzed = semantic::analyze(*parsed.module);
    require(analyzed.module.has_value(), "AD-save fixture analysis failed");
    auto facts = analysis::analyze(*analyzed.module);
    require(facts.ok(), "AD-save fixture ownership failed");
    auto differentiated = autodiff::differentiate(*analyzed.module, facts, {1, {0}});
    require(differentiated.ok(), "AD-save fixture differentiation failed");
    auto extracted = backend::extractStrictRegion(*analyzed.module, facts, "f", false,
                                                  backend::NativeTarget::Gpu);
    require(extracted.ok(), "AD-save fixture TensorRegion extraction failed");
    backend::PhysicalPlanningObligations obligations;
    for (const auto& saved : differentiated.saves) obligations.savedForBackward.insert(saved.primal);
    const auto savedPlan = plan(*extracted.region, {}, obligations, backend::PhysicalDevice::Gpu);
    bool protectedIntermediate = false;
    for (const auto& saved : differentiated.saves) {
        const auto* mapping = savedPlan.value(saved.primal);
        if (!mapping) continue;
        const bool input = std::find(extracted.region->inputs.begin(), extracted.region->inputs.end(), saved.primal) !=
                           extracted.region->inputs.end();
        if (!input) {
            protectedIntermediate = true;
            require(mapping->classification == backend::PhysicalValueClass::SavedForBackward &&
                    mapping->materialized && !mapping->reusable &&
                    mapping->lifetime.lastObligation == extracted.region->nodes.size(),
                    "actual TH-010 saved intermediate was not physically protected");
        }
    }
    require(protectedIntermediate, "AD-save fixture did not expose an intermediate save");
}

} // namespace

int main() {
    try {
        memoryPlanning();
        asyncPlanning();
        fusionPlanning();
        actualAdSaves();
        std::cout << "V0PhysicalPlanTests PASS " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0PhysicalPlanTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
