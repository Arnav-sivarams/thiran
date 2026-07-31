#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ir/Graph.hpp"
#include "region/RegionFormation.hpp"

namespace thiran::region
{

enum class ShapeKnowledge : std::uint8_t {
    Static,
    RuntimeSpecializable,
    UnknownRank
};

std::string_view toString(ShapeKnowledge value) noexcept;

enum class StrategyDecisionReason : std::uint8_t {
    InputBoundary,
    OutputBoundary,
    RuntimeTransfer,
    StaticConstant,
    StaticOptimizedOperation,
    DynamicOptimizedOperation,
    UnknownRankFallback,
    FallbackOnlyOperation
};

std::string_view toString(StrategyDecisionReason value) noexcept;

struct NodeStrategyDecision final {
    const Node* node;
    RegionStrategy strategy;
    ShapeKnowledge shapeKnowledge;
    StrategyDecisionReason reason;
};

struct StrategyClassificationDiagnostic final {
    std::string code;
    std::string message;
    std::optional<std::string> nodeName;
    std::optional<std::size_t> graphIndex;
    std::optional<std::size_t> dimensionIndex;
};

struct StrategyClassificationResult final {
    std::vector<NodeStrategyDecision> decisions;
    std::vector<StrategyClassificationDiagnostic> diagnostics;

    bool succeeded() const noexcept;

    std::vector<NodeStrategyAssignment> assignments() const;
};

class BaselineStrategyClassifier final {
public:
    static StrategyClassificationResult classify(const Graph& graph);
};

}
