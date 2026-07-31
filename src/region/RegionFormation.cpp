#include "region/RegionFormation.hpp"

#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>

#include "region/RegionVerifier.hpp"

namespace thiran::region
{

namespace
{

void addDiagnostic(
    RegionFormationResult& result,
    std::string code,
    std::string message,
    std::optional<std::string> nodeName = std::nullopt,
    std::optional<std::size_t> assignmentIndex = std::nullopt
)
{
    result.diagnostics.push_back({
        .code = std::move(code),
        .message = std::move(message),
        .nodeName = std::move(nodeName),
        .assignmentIndex = assignmentIndex
    });
}

RegionFormationResult fail(RegionFormationResult result)
{
    result.regionGraph.reset();
    return result;
}

void convertDiagnostics(
    RegionFormationResult& result,
    const RegionVerificationResult& verification
)
{
    for(const auto& diagnostic : verification.diagnostics)
    {
        addDiagnostic(
            result,
            "RF014",
            "RegionVerifier " + diagnostic.code + ": " +
                diagnostic.message,
            diagnostic.nodeName
        );
    }
}

}

bool RegionFormationResult::succeeded() const noexcept
{
    return regionGraph != nullptr && diagnostics.empty();
}

RegionFormationResult TopologicalRegionFormer::form(
    const Graph& graph,
    const std::vector<NodeStrategyAssignment>& assignments
)
{
    RegionFormationResult result{nullptr, {}};
    std::unordered_map<const Node*, std::size_t> storageIndices;

    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node* node = graph.nodes[index].get();
        if(node == nullptr)
        {
            addDiagnostic(
                result,
                "RF001",
                "Graph storage index " + std::to_string(index) +
                    " contains a null Node holder."
            );
        }
    }

    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node* node = graph.nodes[index].get();
        if(node != nullptr)
        {
            const auto [first, inserted] =
                storageIndices.emplace(node, index);
            if(!inserted)
            {
                addDiagnostic(
                    result,
                    "RF002",
                    "Graph Node first stored at index " +
                        std::to_string(first->second) +
                        " is duplicated at storage index " +
                        std::to_string(index) + ".",
                    node->name
                );
            }
        }
    }

    if(!result.diagnostics.empty())
    {
        return fail(std::move(result));
    }

    std::unordered_map<const Node*, RegionStrategy> strategies;
    std::unordered_map<const Node*, std::size_t> firstAssignments;

    for(std::size_t index = 0; index < assignments.size(); ++index)
    {
        if(assignments[index].node == nullptr)
        {
            addDiagnostic(
                result,
                "RF003",
                "Assignment " + std::to_string(index) +
                    " contains a null Node pointer.",
                std::nullopt,
                index
            );
        }
    }

    for(std::size_t index = 0; index < assignments.size(); ++index)
    {
        const Node* node = assignments[index].node;
        if(node != nullptr && !storageIndices.contains(node))
        {
            addDiagnostic(
                result,
                "RF004",
                "Assignment " + std::to_string(index) +
                    " pointer identity did not match a live source Graph Node.",
                std::nullopt,
                index
            );
        }
    }

    for(std::size_t index = 0; index < assignments.size(); ++index)
    {
        const Node* node = assignments[index].node;
        if(node == nullptr || !storageIndices.contains(node))
        {
            continue;
        }

        const auto [first, inserted] = firstAssignments.emplace(node, index);
        if(!inserted)
        {
            addDiagnostic(
                result,
                "RF005",
                "Source Node was first assigned at index " +
                    std::to_string(first->second) +
                    " and duplicated at assignment index " +
                    std::to_string(index) + ".",
                node->name,
                index
            );
        }
        else
        {
            strategies.emplace(node, assignments[index].strategy);
        }
    }

    for(const auto& holder : graph.nodes)
    {
        if(!firstAssignments.contains(holder.get()))
        {
            addDiagnostic(
                result,
                "RF006",
                "Source Graph Node has no strategy assignment.",
                holder->name
            );
        }
    }

    for(std::size_t index = 0; index < assignments.size(); ++index)
    {
        const Node* node = assignments[index].node;
        if(node != nullptr &&
           storageIndices.contains(node) &&
           assignments[index].strategy == RegionStrategy::Unresolved)
        {
            addDiagnostic(
                result,
                "RF007",
                "Formation requires a resolved RegionStrategy.",
                node->name,
                index
            );
        }
    }

    for(const auto& holder : graph.nodes)
    {
        const Node& consumer = *holder;
        for(std::size_t operand = 0;
            operand < consumer.inputs.size();
            ++operand)
        {
            if(consumer.inputs[operand] == nullptr)
            {
                addDiagnostic(
                    result,
                    "RF008",
                    "Consumer '" + consumer.name +
                        "' has a null input at operand index " +
                        std::to_string(operand) + ".",
                    consumer.name
                );
            }
        }
    }

    for(const auto& holder : graph.nodes)
    {
        const Node& consumer = *holder;
        for(std::size_t operand = 0;
            operand < consumer.inputs.size();
            ++operand)
        {
            const Node* producer = consumer.inputs[operand];
            if(producer != nullptr && !storageIndices.contains(producer))
            {
                addDiagnostic(
                    result,
                    "RF009",
                    "Consumer '" + consumer.name +
                        "' has a foreign input at operand index " +
                        std::to_string(operand) + ".",
                    consumer.name
                );
            }
        }
    }

    if(!result.diagnostics.empty())
    {
        return fail(std::move(result));
    }

    std::vector<std::size_t> indegree(graph.nodes.size(), 0);
    std::vector<std::vector<std::size_t>> consumers(graph.nodes.size());
    for(std::size_t consumerIndex = 0;
        consumerIndex < graph.nodes.size();
        ++consumerIndex)
    {
        for(const Node* producer : graph.nodes[consumerIndex]->inputs)
        {
            ++indegree[consumerIndex];
            consumers[storageIndices.at(producer)].push_back(consumerIndex);
        }
    }

    std::priority_queue<
        std::size_t,
        std::vector<std::size_t>,
        std::greater<>
    > ready;
    for(std::size_t index = 0; index < indegree.size(); ++index)
    {
        if(indegree[index] == 0)
        {
            ready.push(index);
        }
    }

    std::vector<const Node*> topologicalOrder;
    topologicalOrder.reserve(graph.nodes.size());
    while(!ready.empty())
    {
        const std::size_t index = ready.top();
        ready.pop();
        topologicalOrder.push_back(graph.nodes[index].get());

        for(const std::size_t consumer : consumers[index])
        {
            --indegree[consumer];
            if(indegree[consumer] == 0)
            {
                ready.push(consumer);
            }
        }
    }

    if(topologicalOrder.size() != graph.nodes.size())
    {
        addDiagnostic(
            result,
            "RF010",
            "Source Graph input adjacency contains a cycle."
        );
        return fail(std::move(result));
    }

    result.regionGraph = std::make_unique<RegionGraph>(graph);
    Region* currentRegion = nullptr;
    RegionStrategy currentStrategy = RegionStrategy::Unresolved;

    for(const Node* node : topologicalOrder)
    {
        const RegionStrategy strategy = strategies.at(node);
        bool connectedToCurrent = false;

        if(currentRegion != nullptr)
        {
            for(const Node* producer : node->inputs)
            {
                if(currentRegion->contains(*producer))
                {
                    connectedToCurrent = true;
                    break;
                }
            }
        }

        if(currentRegion == nullptr ||
           strategy != currentStrategy ||
           !connectedToCurrent)
        {
            currentRegion = &result.regionGraph->createRegion();
            currentStrategy = strategy;
            if(!result.regionGraph->setStrategy(
                currentRegion->id(),
                strategy
            ))
            {
                addDiagnostic(
                    result,
                    "RF012",
                    "RegionGraph rejected the strategy for a new Region.",
                    node->name
                );
                return fail(std::move(result));
            }
        }

        if(!result.regionGraph->appendNode(currentRegion->id(), *node))
        {
            addDiagnostic(
                result,
                "RF011",
                "RegionGraph rejected a validated source Node append.",
                node->name
            );
            return fail(std::move(result));
        }
    }

    if(!result.regionGraph->finalize())
    {
        addDiagnostic(
            result,
            "RF013",
            "RegionGraph failed its first finalization attempt."
        );
        return fail(std::move(result));
    }

    const auto structural = RegionVerifier::verify(
        graph,
        *result.regionGraph,
        RegionVerificationLevel::Structural
    );
    if(!structural.valid)
    {
        convertDiagnostics(result, structural);
        return fail(std::move(result));
    }

    const auto executable = RegionVerifier::verify(
        graph,
        *result.regionGraph,
        RegionVerificationLevel::Executable
    );
    if(!executable.valid)
    {
        convertDiagnostics(result, executable);
        return fail(std::move(result));
    }

    return result;
}

}
