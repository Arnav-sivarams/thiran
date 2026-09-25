#pragma once

#include "backend/v0/TensorRegion.hpp"
#include "runtime/v0/Async.hpp"
#include "storage/v0/Storage.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace thiran::v0::backend {

using PhysicalSlotId = std::uint32_t;
using FusionGroupId = std::uint32_t;

enum class PhysicalDevice { Host, Gpu };
enum class PhysicalLayout { ContiguousRowMajor, Strided };
enum class PhysicalValueClass {
    External,
    Temporary,
    Output,
    Alias,
    View,
    SavedForBackward,
    Parameter,
    TrainingState,
    AsyncReserved,
    ExternallyObserved,
    FusedIntermediate,
    ZeroSize
};

struct PhysicalPlanOptions {
    bool enableReuse = true;
    bool enableFusion = true;
    bool operator==(const PhysicalPlanOptions&) const = default;
};

// Obligations are compiler facts, not source annotations. A protected value is
// retained through the region boundary and is never assigned a reusable slot.
struct PhysicalPlanningObligations {
    std::set<semantic::ValueId> savedForBackward;
    std::set<semantic::ValueId> parameters;
    std::set<semantic::ValueId> trainingState;
    std::set<semantic::ValueId> externallyObserved;
    std::set<semantic::ValueId> asyncReadReserved;
    std::set<semantic::ValueId> asyncWriteReserved;
    std::set<semantic::ValueId> fusionBarriers;
    std::map<semantic::ValueId, semantic::ValueId> viewRoots;
};

// Snapshots the TH-014 reservation state into explicit planning obligations.
// Completed-but-unobserved work remains protected because reservations are
// released only by observation.
void addActiveAsyncObligation(PhysicalPlanningObligations&,
                              semantic::ValueId,
                              const runtime::AsyncResource&);

struct PhysicalRequirement {
    storage::DType dtype = storage::DType::Invalid;
    std::uint32_t rank = 0;
    semantic::ShapeFact shape;
    std::optional<std::uint64_t> elementCount;
    std::optional<std::size_t> byteCount;
    std::size_t alignment = 0;
    PhysicalLayout layout = PhysicalLayout::ContiguousRowMajor;
    PhysicalDevice device = PhysicalDevice::Host;
    bool operator==(const PhysicalRequirement&) const = default;
};

struct PhysicalLifetime {
    std::size_t definition = 0;
    std::size_t lastObligation = 0; // Inclusive instruction position.
    bool operator==(const PhysicalLifetime&) const = default;
};

struct PhysicalValuePlan {
    semantic::ValueId value = 0;
    semantic::ValueId root = 0;
    PhysicalRequirement requirement;
    PhysicalLifetime lifetime;
    PhysicalValueClass classification = PhysicalValueClass::Temporary;
    bool materialized = true;
    bool reusable = false;
    std::optional<PhysicalSlotId> slot;
    bool operator==(const PhysicalValuePlan&) const = default;
};

struct PhysicalSlot {
    PhysicalSlotId id = 0;
    PhysicalDevice device = PhysicalDevice::Host;
    storage::DType dtype = storage::DType::Invalid;
    std::uint32_t rank = 0;
    semantic::ShapeFact shape;
    std::optional<std::size_t> capacityBytes;
    std::size_t alignment = 0;
    PhysicalLayout layout = PhysicalLayout::ContiguousRowMajor;
    bool external = false;
    bool reusable = false;
    std::vector<semantic::ValueId> values;
    bool operator==(const PhysicalSlot&) const = default;
};

struct FusionGroup {
    FusionGroupId id = 0;
    std::vector<semantic::ValueId> nodes;
    semantic::ValueId output = 0;
    semantic::Type type;
    semantic::ShapeFact shape;
    bool fused = false;
    bool operator==(const FusionGroup&) const = default;
};

struct PhysicalPlan {
    semantic::FunctionId function = 0;
    PhysicalDevice device = PhysicalDevice::Host;
    PhysicalPlanOptions options;
    std::vector<PhysicalValuePlan> values;
    std::vector<PhysicalSlot> slots;
    std::vector<FusionGroup> fusionGroups;
    std::string dump() const;
    const PhysicalValuePlan* value(semantic::ValueId) const noexcept;
    const PhysicalSlot* slot(PhysicalSlotId) const noexcept;
    const FusionGroup* groupFor(semantic::ValueId) const noexcept;
};

struct PhysicalPlanResult {
    std::optional<PhysicalPlan> plan;
    std::vector<std::string> errors;
    bool ok() const noexcept { return plan.has_value() && errors.empty(); }
};

struct PhysicalPlanVerification {
    std::vector<std::string> errors;
    bool ok() const noexcept { return errors.empty(); }
};

PhysicalPlanResult buildPhysicalPlan(const TensorRegion&,
                                     PhysicalDevice,
                                     PhysicalPlanOptions = {},
                                     const PhysicalPlanningObligations& = {});

// This routine recomputes requirements, roots, lifetimes, interference, and
// fusion legality from the region and obligations. It does not trust planner
// metadata merely because it was produced by buildPhysicalPlan.
PhysicalPlanVerification verifyPhysicalPlan(const TensorRegion&,
                                            const PhysicalPlan&,
                                            const PhysicalPlanningObligations& = {});

} // namespace thiran::v0::backend
