#include "backend/v0/PhysicalPlan.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace thiran::v0::backend {
namespace {

bool tensor(const semantic::Type& type) {
    return type.kind == semantic::TypeKind::Tensor && type.elements.size() == 1;
}

bool elementwise(const RegionNode& node) {
    if (node.op == RegionOp::Negate || node.op == RegionOp::Add ||
        node.op == RegionOp::Subtract || node.op == RegionOp::ElementMultiply) return true;
    return node.op==RegionOp::Extension;
}
bool fusible(const RegionNode& node) {
    return node.op!=RegionOp::Extension || (node.extensionOperation && node.extensionOperation->fusible &&
        node.extensionOperation->effect==extension::EffectV0::Pure);
}

bool kernelOp(const RegionNode& node) { return elementwise(node) || node.op == RegionOp::Index; }

std::string deviceName(PhysicalDevice device) {
    return device == PhysicalDevice::Host ? "host" : "gpu";
}

std::string className(PhysicalValueClass classification) {
    switch (classification) {
    case PhysicalValueClass::External: return "external";
    case PhysicalValueClass::Temporary: return "temporary";
    case PhysicalValueClass::Output: return "output";
    case PhysicalValueClass::Alias: return "alias";
    case PhysicalValueClass::View: return "view";
    case PhysicalValueClass::SavedForBackward: return "saved-for-backward";
    case PhysicalValueClass::Parameter: return "parameter";
    case PhysicalValueClass::TrainingState: return "training-state";
    case PhysicalValueClass::AsyncReserved: return "async-reserved";
    case PhysicalValueClass::ExternallyObserved: return "externally-observed";
    case PhysicalValueClass::FusedIntermediate: return "fused-intermediate";
    case PhysicalValueClass::ZeroSize: return "zero-size";
    }
    return "unknown";
}

std::string shapeText(const semantic::ShapeFact& shape) {
    std::string result = "[";
    for (std::size_t index = 0; index < shape.extents.size(); ++index) {
        if (index) result += ',';
        result += shape.extents[index] ? std::to_string(*shape.extents[index]) : "?";
    }
    return result + ']';
}

const RegionNode* node(const TensorRegion& region, semantic::ValueId id) {
    const auto found = std::find_if(region.nodes.begin(), region.nodes.end(),
        [&](const RegionNode& candidate) { return candidate.id == id; });
    return found == region.nodes.end() ? nullptr : &*found;
}

std::map<semantic::ValueId, std::size_t> positions(const TensorRegion& region) {
    std::map<semantic::ValueId, std::size_t> result;
    for (std::size_t index = 0; index < region.nodes.size(); ++index)
        result.emplace(region.nodes[index].id, index);
    return result;
}

semantic::ValueId canonicalAlias(const TensorRegion& region, semantic::ValueId id) {
    std::set<semantic::ValueId> seen;
    while (seen.insert(id).second) {
        const auto* current = node(region, id);
        if (!current || current->op != RegionOp::Alias || current->dependencies.size() != 1) break;
        id = current->dependencies.front();
    }
    return id;
}

std::map<semantic::ValueId, std::vector<semantic::ValueId>> fusionConsumers(const TensorRegion& region) {
    std::map<semantic::ValueId, std::vector<semantic::ValueId>> result;
    for (const auto& current : region.nodes) {
        if (current.op == RegionOp::Alias) continue;
        for (auto dependency : current.dependencies)
            result[canonicalAlias(region, dependency)].push_back(current.id);
    }
    return result;
}

bool protectedValue(semantic::ValueId id, const PhysicalPlanningObligations& obligations) {
    return obligations.savedForBackward.contains(id) || obligations.parameters.contains(id) ||
           obligations.trainingState.contains(id) || obligations.externallyObserved.contains(id) ||
           obligations.asyncReadReserved.contains(id) || obligations.asyncWriteReserved.contains(id) ||
           obligations.viewRoots.contains(id);
}

bool ownershipProtected(const RegionNode& node) {
    return node.provenance == analysis::ProvenanceKind::ReadViewOf ||
           node.provenance == analysis::ProvenanceKind::PossibleAlias;
}

PhysicalValueClass classification(semantic::ValueId id,
                                  semantic::ValueId root,
                                  const RegionNode& current,
                                  const TensorRegion& region,
                                  const PhysicalPlanningObligations& obligations,
                                  bool fusedIntermediate,
                                  bool zero) {
    if (zero) return PhysicalValueClass::ZeroSize;
    if (fusedIntermediate) return PhysicalValueClass::FusedIntermediate;
    if (obligations.savedForBackward.contains(id) || obligations.savedForBackward.contains(root))
        return PhysicalValueClass::SavedForBackward;
    if (obligations.parameters.contains(id) || obligations.parameters.contains(root))
        return PhysicalValueClass::Parameter;
    if (obligations.trainingState.contains(id) || obligations.trainingState.contains(root))
        return PhysicalValueClass::TrainingState;
    if (obligations.asyncReadReserved.contains(id) || obligations.asyncReadReserved.contains(root) ||
        obligations.asyncWriteReserved.contains(id) || obligations.asyncWriteReserved.contains(root))
        return PhysicalValueClass::AsyncReserved;
    if (obligations.externallyObserved.contains(id) || obligations.externallyObserved.contains(root))
        return PhysicalValueClass::ExternallyObserved;
    if (obligations.viewRoots.contains(id)) return PhysicalValueClass::View;
    if (current.provenance == analysis::ProvenanceKind::ReadViewOf) return PhysicalValueClass::View;
    if (current.provenance == analysis::ProvenanceKind::PossibleAlias)
        return PhysicalValueClass::ExternallyObserved;
    if (current.op == RegionOp::Alias) return PhysicalValueClass::Alias;
    if (id == region.output || root == region.output) return PhysicalValueClass::Output;
    if (current.op == RegionOp::Input) return PhysicalValueClass::External;
    return PhysicalValueClass::Temporary;
}

PhysicalRequirement requirement(const RegionNode& current, PhysicalDevice device) {
    if (!tensor(current.type)) throw std::invalid_argument("MP001 non-tensor physical requirement");
    PhysicalRequirement result;
    result.dtype = storage::fromSemantic(current.type.elements[0].kind);
    result.rank = current.type.rank;
    result.shape = current.shape;
    result.alignment = storage::elementWidth(result.dtype);
    result.device = device;
    if (result.shape.extents.size() != result.rank)
        throw std::invalid_argument("MP002 rank/shape mismatch");
    std::uint64_t count = 1;
    bool known = true;
    for (auto extent : result.shape.extents) {
        if (!extent) { known = false; continue; }
        if (*extent < 0) throw std::invalid_argument("MP003 negative extent");
        if (known) count = storage::checkedMultiply(count, static_cast<std::uint64_t>(*extent));
    }
    if (known) {
        result.elementCount = count;
        result.byteCount = storage::checkedByteCount(count, result.dtype);
    }
    return result;
}

bool sameSlotRequirement(const PhysicalSlot& slot, const PhysicalRequirement& requirement) {
    return slot.dtype == requirement.dtype && slot.rank == requirement.rank &&
           slot.shape == requirement.shape && slot.alignment >= requirement.alignment &&
           slot.layout == requirement.layout && slot.device == requirement.device &&
           (!slot.capacityBytes || !requirement.byteCount || *slot.capacityBytes >= *requirement.byteCount);
}

bool overlaps(const PhysicalLifetime& left, const PhysicalLifetime& right) {
    return !(left.lastObligation < right.definition || right.lastObligation < left.definition);
}

std::vector<FusionGroup> makeFusionGroups(const TensorRegion& region,
                                          const PhysicalPlanOptions& options,
                                          const PhysicalPlanningObligations& obligations) {
    const auto uses = fusionConsumers(region);
    const auto position = positions(region);
    auto protectedRoot = [&](semantic::ValueId root) {
        for (const auto& candidate : region.nodes)
            if (canonicalAlias(region, candidate.id) == root &&
                (protectedValue(candidate.id, obligations) ||
                 obligations.fusionBarriers.contains(candidate.id))) return true;
        return false;
    };
    std::vector<FusionGroup> groups;
    for (const auto& current : region.nodes) {
        if (!kernelOp(current)) continue;
        bool extend = false;
        if (options.enableFusion && elementwise(current) && fusible(current) && !groups.empty()) {
            auto& prior = groups.back();
            const auto* tail = node(region, prior.nodes.back());
            if (tail && elementwise(*tail) && fusible(*tail) && !protectedRoot(current.id) &&
                !protectedRoot(tail->id) && canonicalAlias(region, region.output) != tail->id &&
                tail->type == current.type && tail->shape == current.shape) {
                const auto use = uses.find(tail->id);
                const bool oneConsumer = use != uses.end() && use->second.size() == 1 &&
                                         use->second.front() == current.id;
                const auto occurrences = static_cast<std::size_t>(
                    std::count_if(current.dependencies.begin(), current.dependencies.end(),
                        [&](semantic::ValueId dependency) {
                            return canonicalAlias(region, dependency) == tail->id;
                        }));
                bool interveningMaterialization = false;
                for (auto dependency : current.dependencies) {
                    const auto canonical = canonicalAlias(region, dependency);
                    if (canonical != tail->id && position.at(canonical) > position.at(prior.nodes.front()))
                        interveningMaterialization = true;
                }
                extend = oneConsumer && occurrences == 1 && !interveningMaterialization;
            }
        }
        if (extend) {
            auto& group = groups.back();
            group.nodes.push_back(current.id);
            group.output = current.id;
            group.fused = true;
        } else {
            FusionGroup group;
            group.id = static_cast<FusionGroupId>(groups.size() + 1);
            group.nodes = {current.id};
            group.output = current.id;
            group.type = current.type;
            group.shape = current.shape;
            groups.push_back(std::move(group));
        }
    }
    return groups;
}

std::set<semantic::ValueId> fusedIntermediates(const std::vector<FusionGroup>& groups) {
    std::set<semantic::ValueId> result;
    for (const auto& group : groups)
        if (group.nodes.size() > 1)
            result.insert(group.nodes.begin(), group.nodes.end() - 1);
    return result;
}

std::map<semantic::ValueId, semantic::ValueId> roots(
    const TensorRegion& region, const PhysicalPlanningObligations& obligations) {
    std::map<semantic::ValueId, semantic::ValueId> result;
    for (const auto& current : region.nodes) {
        if (!tensor(current.type)) continue;
        if (auto view = obligations.viewRoots.find(current.id); view != obligations.viewRoots.end()) {
            if (!result.contains(view->second))
                throw std::invalid_argument("MP004 view root is not an earlier tensor value");
            result[current.id] = result.at(view->second);
        } else if (current.op == RegionOp::Alias) {
            if (current.dependencies.size() != 1 || !result.contains(current.dependencies.front()))
                throw std::invalid_argument("MP005 alias root is unavailable");
            result[current.id] = result.at(current.dependencies.front());
        } else {
            result[current.id] = current.id;
        }
    }
    return result;
}

PhysicalLifetime requiredLifetime(const TensorRegion& region,
                                  semantic::ValueId value,
                                  semantic::ValueId root,
                                  const std::map<semantic::ValueId, semantic::ValueId>& valueRoots,
                                  const std::map<semantic::ValueId, std::size_t>& position,
                                  const PhysicalPlanningObligations& obligations) {
    PhysicalLifetime lifetime{position.at(value), position.at(value)};
    for (const auto& current : region.nodes) {
        for (auto dependency : current.dependencies)
            if (valueRoots.contains(dependency) && valueRoots.at(dependency) == root)
                lifetime.lastObligation = std::max(lifetime.lastObligation, position.at(current.id));
        for (auto dependency : current.indices)
            if (valueRoots.contains(dependency) && valueRoots.at(dependency) == root)
                lifetime.lastObligation = std::max(lifetime.lastObligation, position.at(current.id));
    }
    for (const auto& [other, otherRoot] : valueRoots)
        if (otherRoot == root) lifetime.definition = std::min(lifetime.definition, position.at(other));
    bool retained = false;
    for (const auto& [other, otherRoot] : valueRoots)
        if (otherRoot == root && (other == region.output || protectedValue(other, obligations))) retained = true;
    if (retained) lifetime.lastObligation = region.nodes.size();
    return lifetime;
}

bool reusableClass(PhysicalValueClass classification) {
    return classification == PhysicalValueClass::Temporary;
}

bool protectsRoot(PhysicalValueClass classification) {
    return classification == PhysicalValueClass::Output ||
           classification == PhysicalValueClass::View ||
           classification == PhysicalValueClass::SavedForBackward ||
           classification == PhysicalValueClass::Parameter ||
           classification == PhysicalValueClass::TrainingState ||
           classification == PhysicalValueClass::AsyncReserved ||
           classification == PhysicalValueClass::ExternallyObserved;
}

void addError(PhysicalPlanVerification& result, std::string code, std::string message) {
    result.errors.push_back(std::move(code) + " " + std::move(message));
}

} // namespace

void addActiveAsyncObligation(PhysicalPlanningObligations& obligations,
                              semantic::ValueId value,
                              const runtime::AsyncResource& resource) {
    if (!resource.valid()) throw std::invalid_argument("MP006 invalid async resource");
    if (resource.activeReads() != 0) obligations.asyncReadReserved.insert(value);
    if (resource.activeWrites() != 0) obligations.asyncWriteReserved.insert(value);
}

const PhysicalValuePlan* PhysicalPlan::value(semantic::ValueId id) const noexcept {
    const auto found = std::find_if(values.begin(), values.end(),
        [&](const PhysicalValuePlan& candidate) { return candidate.value == id; });
    return found == values.end() ? nullptr : &*found;
}

const PhysicalSlot* PhysicalPlan::slot(PhysicalSlotId id) const noexcept {
    const auto found = std::find_if(slots.begin(), slots.end(),
        [&](const PhysicalSlot& candidate) { return candidate.id == id; });
    return found == slots.end() ? nullptr : &*found;
}

const FusionGroup* PhysicalPlan::groupFor(semantic::ValueId id) const noexcept {
    const auto found = std::find_if(fusionGroups.begin(), fusionGroups.end(),
        [&](const FusionGroup& group) {
            return std::find(group.nodes.begin(), group.nodes.end(), id) != group.nodes.end();
        });
    return found == fusionGroups.end() ? nullptr : &*found;
}

std::string PhysicalPlan::dump() const {
    std::ostringstream out;
    out << "physical-plan f" << function << " device=" << deviceName(device)
        << " reuse=" << (options.enableReuse ? "on" : "off")
        << " fusion=" << (options.enableFusion ? "on" : "off") << '\n';
    for (const auto& current : values) {
        out << "  value v" << current.value << " root=v" << current.root
            << " class=" << className(current.classification)
            << " dtype=" << storage::dtypeName(current.requirement.dtype)
            << " shape=" << shapeText(current.requirement.shape)
            << " bytes=" << (current.requirement.byteCount ? std::to_string(*current.requirement.byteCount) : "?")
            << " lifetime=[" << current.lifetime.definition << ',' << current.lifetime.lastObligation << ']'
            << " materialized=" << (current.materialized ? "yes" : "no")
            << " reusable=" << (current.reusable ? "yes" : "no")
            << " slot=" << (current.slot ? std::to_string(*current.slot) : "none") << '\n';
    }
    for (const auto& slot : slots) {
        out << "  slot s" << slot.id << " device=" << deviceName(slot.device)
            << " dtype=" << storage::dtypeName(slot.dtype) << " shape=" << shapeText(slot.shape)
            << " capacity=" << (slot.capacityBytes ? std::to_string(*slot.capacityBytes) : "?")
            << " external=" << (slot.external ? "yes" : "no")
            << " reusable=" << (slot.reusable ? "yes" : "no") << " values=";
        for (std::size_t index = 0; index < slot.values.size(); ++index) {
            if (index) out << ',';
            out << 'v' << slot.values[index];
        }
        out << '\n';
    }
    for (const auto& group : fusionGroups) {
        out << "  group g" << group.id << " nodes=";
        for (std::size_t index = 0; index < group.nodes.size(); ++index) {
            if (index) out << ',';
            out << 'v' << group.nodes[index];
        }
        out << " output=v" << group.output << " fused=" << (group.fused ? "yes" : "no") << '\n';
    }
    return out.str();
}

PhysicalPlanResult buildPhysicalPlan(const TensorRegion& region,
                                     PhysicalDevice device,
                                     PhysicalPlanOptions options,
                                     const PhysicalPlanningObligations& obligations) {
    PhysicalPlanResult result;
    const auto regionVerification = verifyRegion(region);
    if (!regionVerification.ok()) {
        for (const auto& error : regionVerification.errors) result.errors.push_back("MP-REGION " + error);
        return result;
    }
    try {
        PhysicalPlan plan;
        plan.function = region.function;
        plan.device = device;
        plan.options = options;
        plan.fusionGroups = makeFusionGroups(region, options, obligations);
        const auto fused = fusedIntermediates(plan.fusionGroups);
        const auto valueRoots = roots(region, obligations);
        const auto position = positions(region);

        for (const auto& current : region.nodes) {
            if (!tensor(current.type)) continue;
            PhysicalValuePlan value;
            value.value = current.id;
            value.root = valueRoots.at(current.id);
            value.requirement = requirement(current, device);
            value.lifetime = requiredLifetime(region, current.id, value.root, valueRoots, position, obligations);
            const bool zero = value.requirement.elementCount && *value.requirement.elementCount == 0;
            value.materialized = !fused.contains(current.id) && !fused.contains(value.root);
            value.classification = classification(current.id, value.root, current, region,
                                                  obligations, !value.materialized, zero);
            value.reusable = value.materialized && !zero && options.enableReuse &&
                             reusableClass(value.classification) && current.id == value.root &&
                             !ownershipProtected(current);
            plan.values.push_back(std::move(value));
        }

        // Root lifetimes/classifications govern every alias of the same physical contents.
        for (auto& current : plan.values) {
            const auto root = std::find_if(plan.values.begin(), plan.values.end(),
                [&](const PhysicalValuePlan& candidate) { return candidate.value == current.root; });
            if (root == plan.values.end()) throw std::invalid_argument("MP007 physical root missing");
            current.lifetime = root->lifetime = requiredLifetime(
                region, current.root, current.root, valueRoots, position, obligations);
            if (current.value != current.root) current.reusable = false;
        }
        for (auto& root : plan.values) {
            if (root.value != root.root) continue;
            const bool retained = std::any_of(plan.values.begin(), plan.values.end(),
                [&](const PhysicalValuePlan& candidate) {
                    return candidate.root == root.value &&
                           (candidate.value == region.output || protectsRoot(candidate.classification));
                });
            if (retained) root.reusable = false;
        }

        for (auto& current : plan.values) {
            if (current.value != current.root || !current.materialized ||
                (current.requirement.byteCount && *current.requirement.byteCount == 0)) continue;
            PhysicalSlot* selected = nullptr;
            if (current.reusable) {
                for (auto& candidate : plan.slots) {
                    if (!candidate.reusable || !sameSlotRequirement(candidate, current.requirement)) continue;
                    bool interferes = false;
                    for (auto priorId : candidate.values) {
                        const auto* prior = plan.value(priorId);
                        if (prior && prior->root == prior->value && overlaps(prior->lifetime, current.lifetime)) {
                            interferes = true;
                            break;
                        }
                    }
                    if (!interferes) { selected = &candidate; break; }
                }
            }
            if (!selected) {
                PhysicalSlot slot;
                slot.id = static_cast<PhysicalSlotId>(plan.slots.size() + 1);
                slot.device = device;
                slot.dtype = current.requirement.dtype;
                slot.rank = current.requirement.rank;
                slot.shape = current.requirement.shape;
                slot.capacityBytes = current.requirement.byteCount;
                slot.alignment = current.requirement.alignment;
                slot.layout = current.requirement.layout;
                const auto* rootNode = node(region, current.root);
                slot.external = rootNode && rootNode->op == RegionOp::Input;
                slot.reusable = current.reusable;
                plan.slots.push_back(std::move(slot));
                selected = &plan.slots.back();
            }
            current.slot = selected->id;
            selected->values.push_back(current.value);
        }
        for (auto& current : plan.values) {
            if (current.value == current.root || !current.materialized) continue;
            const auto* root = plan.value(current.root);
            if (!root) throw std::invalid_argument("MP007 physical root missing");
            current.slot = root->slot;
            if (current.slot) {
                auto slot = std::find_if(plan.slots.begin(), plan.slots.end(),
                    [&](const PhysicalSlot& candidate) { return candidate.id == *current.slot; });
                if (slot == plan.slots.end()) throw std::invalid_argument("MP008 alias slot missing");
                slot->values.push_back(current.value);
            }
        }

        const auto verified = verifyPhysicalPlan(region, plan, obligations);
        if (!verified.ok()) {
            result.errors = verified.errors;
            return result;
        }
        result.plan = std::move(plan);
    } catch (const std::exception& error) {
        result.errors.push_back(std::string("MP-BUILD ") + error.what());
    }
    return result;
}

PhysicalPlanVerification verifyPhysicalPlan(const TensorRegion& region,
                                            const PhysicalPlan& plan,
                                            const PhysicalPlanningObligations& obligations) {
    PhysicalPlanVerification result;
    const auto regionVerification = verifyRegion(region);
    if (!regionVerification.ok()) {
        addError(result, "MPV01", "source TensorRegion is invalid");
        return result;
    }
    if (plan.function != region.function) addError(result, "MPV02", "plan belongs to another function");
    std::map<PhysicalSlotId, const PhysicalSlot*> slots;
    for (const auto& slot : plan.slots) {
        if (slot.id == 0 || !slots.emplace(slot.id, &slot).second)
            addError(result, "MPV03", "zero or duplicate physical slot ID");
        if (slot.device != plan.device) addError(result, "MPV04", "slot device mismatch");
        if (slot.alignment == 0 || (slot.alignment & (slot.alignment - 1)) != 0)
            addError(result, "MPV05", "invalid slot alignment");
    }

    std::map<semantic::ValueId, const PhysicalValuePlan*> values;
    for (const auto& value : plan.values) {
        if (value.value == 0 || !values.emplace(value.value, &value).second)
            addError(result, "MPV06", "zero or duplicate physical value mapping");
        if (!node(region, value.value) || !tensor(node(region, value.value)->type))
            addError(result, "MPV07", "physical mapping references a non-tensor or unknown value");
        if (value.requirement.device != plan.device)
            addError(result, "MPV04", "value device mismatch");
        if (value.slot && !slots.contains(*value.slot))
            addError(result, "MPV08", "value references an invalid physical slot ID");
    }
    for (const auto& current : region.nodes)
        if (tensor(current.type) && !values.contains(current.id))
            addError(result, "MPV09", "tensor value has no physical-plan entry");

    std::map<semantic::ValueId, semantic::ValueId> expectedRoots;
    std::map<semantic::ValueId, std::size_t> position;
    try {
        expectedRoots = roots(region, obligations);
        position = positions(region);
    } catch (const std::exception& error) {
        addError(result, "MPV10", error.what());
        return result;
    }
    const auto expectedGroups = makeFusionGroups(region, plan.options, obligations);
    if (plan.fusionGroups != expectedGroups)
        addError(result, "FPV01", "fusion groups differ from independently recomputed legal groups");
    std::set<FusionGroupId> groupIds;
    std::set<semantic::ValueId> grouped;
    for (const auto& group : plan.fusionGroups) {
        if (group.id == 0 || !groupIds.insert(group.id).second || group.nodes.empty())
            addError(result, "FPV02", "malformed fusion group identity or membership");
        if (group.output != group.nodes.back())
            addError(result, "FPV03", "fusion output is not the ordered terminal node");
        for (auto id : group.nodes)
            if (!grouped.insert(id).second || !node(region, id) || !kernelOp(*node(region, id)))
                addError(result, "FPV04", "fusion group duplicates or references an ineligible node");
        if (group.fused != (group.nodes.size() > 1))
            addError(result, "FPV05", "fusion marker contradicts group size");
    }
    for (const auto& current : region.nodes)
        if (kernelOp(current) && !grouped.contains(current.id))
            addError(result, "FPV06", "kernel node is missing from fusion plan");

    const auto fused = fusedIntermediates(plan.fusionGroups);
    for (const auto& [id, value] : values) {
        const auto* current = node(region, id);
        if (!current) continue;
        if (value->root != expectedRoots.at(id)) addError(result, "MPV10", "alias/view root mismatch");
        try {
            const auto expectedRequirement = requirement(*current, plan.device);
            if (value->requirement != expectedRequirement)
                addError(result, "MPV11", "dtype/rank/shape/size/layout requirement mismatch");
            const auto expectedLifetime = requiredLifetime(region, id, value->root,
                                                           expectedRoots, position, obligations);
            if (value->lifetime != expectedLifetime)
                addError(result, "MPV12", "invalid or truncated physical lifetime");
            const bool zeroRequirement = expectedRequirement.byteCount && *expectedRequirement.byteCount == 0;
            const auto expectedClass = classification(id, value->root, *current, region, obligations,
                fused.contains(id) || fused.contains(value->root), zeroRequirement);
            if (value->classification != expectedClass)
                addError(result, "MPV23", "reusable/non-reusable classification mismatch");
        } catch (const std::exception& error) {
            addError(result, "MPV11", error.what());
        }
        const bool shouldMaterialize = !fused.contains(id) && !fused.contains(value->root);
        if (value->materialized != shouldMaterialize)
            addError(result, "MPV13", "materialization boundary disagrees with fusion plan");
        const bool zero = value->requirement.byteCount && *value->requirement.byteCount == 0;
        if ((!value->materialized || zero) && value->slot)
            addError(result, "MPV14", "non-materialized or zero-size value has an allocation");
        if (value->materialized && !zero && !value->slot)
            addError(result, "MPV15", "materialized value is missing a physical assignment");
        if (value->reusable && (!plan.options.enableReuse || id != value->root ||
            value->classification != PhysicalValueClass::Temporary))
            addError(result, "MPV16", "protected/alias value is incorrectly reusable");
        if ((id == region.output || protectedValue(id, obligations)) && value->reusable)
            addError(result, "MPV17", "output or retained obligation is reusable");
    }
    for (const auto& [id, value] : values) {
        if (id != value->root) continue;
        bool retained = false;
        for (const auto& [otherId, other] : values)
            if (other->root == id &&
                (otherId == region.output || protectsRoot(other->classification))) retained = true;
        const bool zero = value->requirement.byteCount && *value->requirement.byteCount == 0;
        const auto* rootNode = node(region, id);
        const bool expectedReusable = value->materialized && !zero && plan.options.enableReuse &&
            value->classification == PhysicalValueClass::Temporary && !retained && rootNode &&
            !ownershipProtected(*rootNode);
        if (value->reusable != expectedReusable)
            addError(result, "MPV16", "root reusable flag contradicts independently recomputed policy");
    }

    for (const auto& slot : plan.slots) {
        std::set<semantic::ValueId> listed;
        std::vector<const PhysicalValuePlan*> rootsInSlot;
        for (auto id : slot.values) {
            if (!listed.insert(id).second) addError(result, "MPV18", "duplicate contradictory slot member");
            const auto found = values.find(id);
            if (found == values.end() || !found->second->slot || *found->second->slot != slot.id) {
                addError(result, "MPV18", "slot membership and value mapping disagree");
                continue;
            }
            const auto& value = *found->second;
            if (!sameSlotRequirement(slot, value.requirement))
                addError(result, "MPV19", "slot capacity/dtype/layout/device is incompatible");
            if (value.value == value.root) rootsInSlot.push_back(&value);
        }
        for (const auto& [id, value] : values)
            if (value->slot && *value->slot == slot.id && !listed.contains(id))
                addError(result, "MPV18", "value mapping is absent from slot members");
        for (std::size_t left = 0; left < rootsInSlot.size(); ++left)
            for (std::size_t right = left + 1; right < rootsInSlot.size(); ++right)
                if (rootsInSlot[left]->root != rootsInSlot[right]->root &&
                    overlaps(rootsInSlot[left]->lifetime, rootsInSlot[right]->lifetime))
                    addError(result, "MPV20", "interfering values share a physical slot");
        if (slot.external && slot.reusable)
            addError(result, "MPV21", "external slot is marked reusable");
        if (rootsInSlot.size() > 1 && (!slot.reusable || slot.external))
            addError(result, "MPV21", "non-reusable/external slot has independent roots");
        for (const auto* value : rootsInSlot)
            if ((value->value == region.output || protectedValue(value->value, obligations)) &&
                (slot.reusable || rootsInSlot.size() > 1))
                addError(result, "MPV22", "output or retained root shares a reusable slot");
        for (const auto* value : rootsInSlot)
            if (slot.reusable != value->reusable)
                addError(result, "MPV24", "slot reusable flag contradicts its root assignment");
        for (const auto* value : rootsInSlot) {
            const auto* rootNode = node(region, value->root);
            const bool expectedExternal = rootNode && rootNode->op == RegionOp::Input;
            if (slot.external != expectedExternal)
                addError(result, "MPV25", "slot external flag contradicts root ownership");
        }
    }
    return result;
}

} // namespace thiran::v0::backend
