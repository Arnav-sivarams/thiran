#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "region/RegionGraph.hpp"
#include "region/StrategyClassification.hpp"

namespace thiran::region
{

enum class RegionPlanDiagnosticStage : std::uint8_t {
    Classification,
    Formation,
    Verification
};

std::string_view toString(RegionPlanDiagnosticStage stage) noexcept;

struct RegionPlanDiagnostic final {
    RegionPlanDiagnosticStage stage;
    std::string code;
    std::string message;
    std::optional<std::string> nodeName;
    std::optional<std::size_t> graphIndex;
    std::optional<std::size_t> assignmentIndex;
    std::optional<std::size_t> dimensionIndex;
    std::optional<RegionId> regionId;
};

class RegionPlan final {
public:
    ~RegionPlan() = default;
    RegionPlan(const RegionPlan&) = delete;
    RegionPlan& operator=(const RegionPlan&) = delete;
    RegionPlan(RegionPlan&&) noexcept = default;
    RegionPlan& operator=(RegionPlan&&) noexcept = default;

    const Graph& sourceGraph() const noexcept;
    const std::vector<NodeStrategyDecision>& decisions() const noexcept;
    const RegionGraph& regionGraph() const noexcept;
    const NodeStrategyDecision* decisionFor(const Node& node) const noexcept;
    const Region* regionFor(const Node& node) const noexcept;
    std::size_t nodeCount() const noexcept;
    std::size_t regionCount() const noexcept;
    bool empty() const noexcept;

private:
    friend class HybridRegionPlanner;
    friend class RegionPlanVerifier;

    RegionPlan(
        const Graph& graph,
        std::vector<NodeStrategyDecision> decisions,
        std::unique_ptr<RegionGraph> regionGraph
    );

    const Graph* sourceGraph_;
    std::vector<NodeStrategyDecision> decisions_;
    std::unique_ptr<RegionGraph> regionGraph_;
    std::unordered_map<const Node*, std::size_t> decisionIndices_;
};

struct RegionPlanResult final {
    std::unique_ptr<RegionPlan> plan;
    std::vector<RegionPlanDiagnostic> diagnostics;
    bool succeeded() const noexcept;
};

class HybridRegionPlanner final {
public:
    static RegionPlanResult build(const Graph& graph);
};

}
