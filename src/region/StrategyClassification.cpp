#include "region/StrategyClassification.hpp"

#include <string>
#include <unordered_map>
#include <utility>

namespace thiran::region
{

namespace
{

void addDiagnostic(
    StrategyClassificationResult& result,
    std::string code,
    std::string message,
    std::optional<std::string> nodeName,
    std::optional<std::size_t> graphIndex,
    std::optional<std::size_t> dimensionIndex = std::nullopt
)
{
    result.diagnostics.push_back({
        std::move(code),
        std::move(message),
        std::move(nodeName),
        graphIndex,
        dimensionIndex
    });
}

ShapeKnowledge shapeKnowledge(const Node& node)
{
    if(node.shape.empty())
    {
        return node.op == Operation::Constant
            ? ShapeKnowledge::Static
            : ShapeKnowledge::UnknownRank;
    }
    for(const int dimension : node.shape)
    {
        if(dimension == -1)
        {
            return ShapeKnowledge::RuntimeSpecializable;
        }
    }
    return ShapeKnowledge::Static;
}

bool isOptimized(Operation operation) noexcept
{
    switch(operation)
    {
        case Operation::Add:
        case Operation::Subtract:
        case Operation::Multiply:
        case Operation::Divide:
        case Operation::MatMul:
        case Operation::ReLU:
        case Operation::Sigmoid:
        case Operation::Tanh:
        case Operation::Reshape:
        case Operation::Transpose:
        case Operation::FusedMatMulRelu:
            return true;
        default:
            return false;
    }
}

bool isFallbackOnly(Operation operation) noexcept
{
    switch(operation)
    {
        case Operation::Conv2D:
        case Operation::Softmax:
        case Operation::MaxPool:
        case Operation::AvgPool:
            return true;
        default:
            return false;
    }
}

}

std::string_view toString(ShapeKnowledge value) noexcept
{
    switch(value)
    {
        case ShapeKnowledge::Static: return "STATIC";
        case ShapeKnowledge::RuntimeSpecializable:
            return "RUNTIME_SPECIALIZABLE";
        case ShapeKnowledge::UnknownRank: return "UNKNOWN_RANK";
    }
    return "UNKNOWN";
}

std::string_view toString(StrategyDecisionReason value) noexcept
{
    switch(value)
    {
        case StrategyDecisionReason::InputBoundary: return "INPUT_BOUNDARY";
        case StrategyDecisionReason::OutputBoundary: return "OUTPUT_BOUNDARY";
        case StrategyDecisionReason::RuntimeTransfer: return "RUNTIME_TRANSFER";
        case StrategyDecisionReason::StaticConstant: return "STATIC_CONSTANT";
        case StrategyDecisionReason::StaticOptimizedOperation:
            return "STATIC_OPTIMIZED_OPERATION";
        case StrategyDecisionReason::DynamicOptimizedOperation:
            return "DYNAMIC_OPTIMIZED_OPERATION";
        case StrategyDecisionReason::UnknownRankFallback:
            return "UNKNOWN_RANK_FALLBACK";
        case StrategyDecisionReason::FallbackOnlyOperation:
            return "FALLBACK_ONLY_OPERATION";
    }
    return "UNKNOWN";
}

bool StrategyClassificationResult::succeeded() const noexcept
{
    return diagnostics.empty();
}

std::vector<NodeStrategyAssignment>
StrategyClassificationResult::assignments() const
{
    std::vector<NodeStrategyAssignment> result;
    if(!succeeded())
    {
        return result;
    }
    result.reserve(decisions.size());
    for(const auto& decision : decisions)
    {
        result.push_back({decision.node, decision.strategy});
    }
    return result;
}

StrategyClassificationResult BaselineStrategyClassifier::classify(
    const Graph& graph
)
{
    StrategyClassificationResult result;
    std::unordered_map<const Node*, std::size_t> firstIndices;

    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        if(graph.nodes[index] == nullptr)
        {
            addDiagnostic(
                result, "SC001",
                "Graph storage index " + std::to_string(index) +
                    " contains a null Node holder.",
                std::nullopt, index
            );
        }
    }
    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node* node = graph.nodes[index].get();
        if(node == nullptr)
        {
            continue;
        }
        const auto [first, inserted] = firstIndices.emplace(node, index);
        if(!inserted)
        {
            addDiagnostic(
                result, "SC002",
                "Graph Node first stored at index " +
                    std::to_string(first->second) +
                    " is duplicated at storage index " +
                    std::to_string(index) + ".",
                node->name, index
            );
        }
    }
    if(!result.diagnostics.empty())
    {
        return result;
    }

    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node& node = *graph.nodes[index];
        if(node.op == Operation::Unknown)
        {
            addDiagnostic(
                result, "SC003",
                "Node '" + node.name + "' has an unknown Operation.",
                node.name, index
            );
        }
    }
    std::vector<bool> invalidShape(graph.nodes.size(), false);
    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node& node = *graph.nodes[index];
        for(std::size_t dimensionIndex = 0;
            dimensionIndex < node.shape.size();
            ++dimensionIndex)
        {
            const int dimension = node.shape[dimensionIndex];
            if(dimension == 0 || dimension < -1)
            {
                invalidShape[index] = true;
                addDiagnostic(
                    result, "SC004",
                    "Node '" + node.name + "' has invalid dimension " +
                        std::to_string(dimension) + " at dimension index " +
                        std::to_string(dimensionIndex) + ".",
                    node.name, index, dimensionIndex
                );
            }
        }
    }
    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node& node = *graph.nodes[index];
        if(!invalidShape[index] &&
           node.op == Operation::Constant &&
           shapeKnowledge(node) == ShapeKnowledge::RuntimeSpecializable)
        {
            addDiagnostic(
                result, "SC005",
                "Node '" + node.name +
                    "': runtime-specializable Constant shapes are not "
                    "supported by the baseline classifier.",
                node.name, index
            );
        }
    }
    if(!result.diagnostics.empty())
    {
        return result;
    }

    result.decisions.reserve(graph.nodes.size());
    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node& node = *graph.nodes[index];
        const ShapeKnowledge knowledge = shapeKnowledge(node);
        NodeStrategyDecision decision{
            &node, RegionStrategy::Unresolved, knowledge,
            StrategyDecisionReason::FallbackOnlyOperation
        };

        if(node.op == Operation::Input)
            decision = {&node, RegionStrategy::Fallback, knowledge,
                        StrategyDecisionReason::InputBoundary};
        else if(node.op == Operation::Output)
            decision = {&node, RegionStrategy::Fallback, knowledge,
                        StrategyDecisionReason::OutputBoundary};
        else if(node.op == Operation::Transfer)
            decision = {&node, RegionStrategy::Fallback, knowledge,
                        StrategyDecisionReason::RuntimeTransfer};
        else if(node.op == Operation::Constant)
            decision = {&node, RegionStrategy::AheadOfTime, knowledge,
                        StrategyDecisionReason::StaticConstant};
        else if(isOptimized(node.op))
        {
            if(knowledge == ShapeKnowledge::Static)
                decision = {&node, RegionStrategy::AheadOfTime, knowledge,
                            StrategyDecisionReason::StaticOptimizedOperation};
            else if(knowledge == ShapeKnowledge::RuntimeSpecializable)
                decision = {&node, RegionStrategy::JustInTime, knowledge,
                            StrategyDecisionReason::DynamicOptimizedOperation};
            else
                decision = {&node, RegionStrategy::Fallback, knowledge,
                            StrategyDecisionReason::UnknownRankFallback};
        }
        else if(isFallbackOnly(node.op))
            decision = {&node, RegionStrategy::Fallback, knowledge,
                        StrategyDecisionReason::FallbackOnlyOperation};
        else
        {
            addDiagnostic(
                result, "SC006",
                "Node '" + node.name +
                    "' reached no classification branch for Operation " +
                    std::to_string(static_cast<int>(node.op)) + ".",
                node.name, index
            );
            result.decisions.clear();
            return result;
        }
        result.decisions.push_back(decision);
    }
    return result;
}

}
