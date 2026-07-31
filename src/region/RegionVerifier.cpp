#include "region/RegionVerifier.hpp"

#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace thiran::region
{

namespace
{

using DependencyKey = std::pair<RegionId, RegionId>;

void addDiagnostic(
    RegionVerificationResult& result,
    std::string code,
    std::string message,
    std::optional<RegionId> regionId = std::nullopt,
    std::optional<std::string> nodeName = std::nullopt
)
{
    result.diagnostics.push_back({
        .code = std::move(code),
        .message = std::move(message),
        .regionId = regionId,
        .nodeName = std::move(nodeName)
    });
}

bool equalNodes(
    const std::vector<const Node*>& left,
    const std::vector<const Node*>& right
)
{
    return left == right;
}

std::unordered_map<const Node*, RegionId> buildMembership(
    const RegionGraph& regionGraph
)
{
    std::unordered_map<const Node*, RegionId> membership;
    for(const auto& holder : regionGraph.regions())
    {
        for(const Node* node : holder->nodes())
        {
            if(node != nullptr && !membership.contains(node))
            {
                membership[node] = holder->id();
            }
        }
    }
    return membership;
}

std::vector<const Node*> expectedInputs(
    const Region& region,
    const std::unordered_map<const Node*, RegionId>& membership
)
{
    std::vector<const Node*> result;
    std::unordered_set<const Node*> seen;

    for(const Node* node : region.nodes())
    {
        if(node == nullptr)
        {
            continue;
        }

        for(const Node* producer : node->inputs)
        {
            if(producer == nullptr)
            {
                continue;
            }

            const auto found = membership.find(producer);
            if(found != membership.end() && found->second == region.id())
            {
                continue;
            }

            if(seen.insert(producer).second)
            {
                result.push_back(producer);
            }
        }
    }

    return result;
}

std::vector<const Node*> expectedOutputs(
    const Region& region,
    const std::unordered_map<const Node*, RegionId>& membership
)
{
    std::vector<const Node*> result;

    for(const Node* node : region.nodes())
    {
        if(node == nullptr)
        {
            continue;
        }

        bool externallyVisible = node->op == Operation::Output;
        for(const Node* consumer : node->outputs)
        {
            if(consumer == nullptr)
            {
                continue;
            }

            const auto found = membership.find(consumer);
            if(found != membership.end() && found->second != region.id())
            {
                externallyVisible = true;
                break;
            }
        }

        if(externallyVisible)
        {
            result.push_back(node);
        }
    }

    return result;
}

std::vector<RegionDependency> expectedDependencies(
    const Graph& graph,
    const RegionGraph& regionGraph,
    const std::unordered_map<const Node*, RegionId>& membership
)
{
    std::map<DependencyKey, std::unordered_set<const Node*>> buckets;

    for(const auto& destinationHolder : regionGraph.regions())
    {
        const Region& destination = *destinationHolder;
        for(const Node* consumer : destination.nodes())
        {
            if(consumer == nullptr)
            {
                continue;
            }

            for(const Node* producer : consumer->inputs)
            {
                if(producer == nullptr)
                {
                    continue;
                }

                const auto found = membership.find(producer);
                if(found == membership.end() ||
                   found->second == destination.id())
                {
                    continue;
                }

                buckets[{found->second, destination.id()}].insert(producer);
            }
        }
    }

    std::vector<RegionDependency> result;
    for(const auto& [key, values] : buckets)
    {
        RegionDependency dependency{
            .source = key.first,
            .destination = key.second,
            .values = {}
        };

        for(const auto& holder : graph.nodes)
        {
            if(values.contains(holder.get()))
            {
                dependency.values.push_back(holder.get());
            }
        }

        result.push_back(std::move(dependency));
    }

    return result;
}

bool equalDependencies(
    const std::vector<RegionDependency>& left,
    const std::vector<RegionDependency>& right
)
{
    if(left.size() != right.size())
    {
        return false;
    }

    for(std::size_t i = 0; i < left.size(); ++i)
    {
        if(left[i].source != right[i].source ||
           left[i].destination != right[i].destination ||
           left[i].values != right[i].values)
        {
            return false;
        }
    }

    return true;
}

}

RegionVerificationResult RegionVerifier::verify(
    const Graph& graph,
    const RegionGraph& regionGraph,
    RegionVerificationLevel level
)
{
    RegionVerificationResult result{
        .valid = true,
        .diagnostics = {}
    };

    const bool sourceMatches = &graph == &regionGraph.sourceGraph();
    if(!sourceMatches)
    {
        addDiagnostic(
            result,
            "RG001",
            "RegionGraph was constructed from another Graph."
        );
    }

    const bool finalized = regionGraph.finalized();
    if(!finalized)
    {
        addDiagnostic(
            result,
            "RG002",
            "RegionGraph must be finalized before verification."
        );
    }

    if(!graph.nodes.empty() && regionGraph.empty())
    {
        addDiagnostic(
            result,
            "RG003",
            "A nonempty source Graph requires at least one Region."
        );
    }

    for(std::size_t i = 0; i < regionGraph.regions().size(); ++i)
    {
        const Region& region = *regionGraph.regions()[i];
        if(region.id() != static_cast<RegionId>(i))
        {
            addDiagnostic(
                result,
                "RG004",
                "Region ID does not match its contiguous storage position.",
                region.id()
            );
        }
    }

    if(finalized)
    {
        for(const auto& holder : regionGraph.regions())
        {
            if(holder->empty())
            {
                addDiagnostic(
                    result,
                    "RG005",
                    "Finalized Region is empty.",
                    holder->id()
                );
            }
        }
    }

    std::unordered_set<const Node*> graphNodes;
    for(const auto& holder : graph.nodes)
    {
        graphNodes.insert(holder.get());
    }

    bool allRegionNodesOwned = sourceMatches;
    for(const auto& holder : regionGraph.regions())
    {
        for(const Node* node : holder->nodes())
        {
            if(node == nullptr)
            {
                allRegionNodesOwned = false;
                addDiagnostic(
                    result,
                    "RG006",
                    "Region contains a null Node pointer.",
                    holder->id()
                );
            }
        }
    }

    for(const auto& holder : regionGraph.regions())
    {
        for(const Node* node : holder->nodes())
        {
            if(node != nullptr && !graphNodes.contains(node))
            {
                allRegionNodesOwned = false;
                addDiagnostic(
                    result,
                    "RG007",
                    "Region contains a Node not owned by the supplied Graph.",
                    holder->id()
                );
            }
        }
    }

    for(const auto& holder : regionGraph.regions())
    {
        std::unordered_set<const Node*> seen;
        for(const Node* node : holder->nodes())
        {
            if(node != nullptr && !seen.insert(node).second)
            {
                addDiagnostic(
                    result,
                    "RG008",
                    "Region contains the same Node more than once.",
                    holder->id(),
                    graphNodes.contains(node)
                        ? std::optional<std::string>(node->name)
                        : std::nullopt
                );
            }
        }
    }

    std::unordered_map<const Node*, RegionId> assignedRegions;
    bool uniqueMembership = true;
    for(const auto& holder : regionGraph.regions())
    {
        for(const Node* node : holder->nodes())
        {
            if(node == nullptr)
            {
                continue;
            }

            const auto [found, inserted] =
                assignedRegions.emplace(node, holder->id());
            if(!inserted && found->second != holder->id())
            {
                uniqueMembership = false;
                addDiagnostic(
                    result,
                    "RG009",
                    "Node is assigned to more than one Region.",
                    holder->id(),
                    graphNodes.contains(node)
                        ? std::optional<std::string>(node->name)
                        : std::nullopt
                );
            }
        }
    }

    for(const auto& holder : graph.nodes)
    {
        if(!assignedRegions.contains(holder.get()))
        {
            addDiagnostic(
                result,
                "RG010",
                "Graph Node is not assigned to any Region.",
                std::nullopt,
                holder->name
            );
        }
    }

    const bool safeNodeTraversal =
        sourceMatches && allRegionNodesOwned && uniqueMembership;

    if(safeNodeTraversal)
    {
        for(const auto& holder : regionGraph.regions())
        {
            std::unordered_map<const Node*, std::size_t> indices;
            for(std::size_t i = 0; i < holder->nodes().size(); ++i)
            {
                if(holder->nodes()[i] != nullptr)
                {
                    indices[holder->nodes()[i]] = i;
                }
            }

            for(std::size_t consumerIndex = 0;
                consumerIndex < holder->nodes().size();
                ++consumerIndex)
            {
                const Node* consumer = holder->nodes()[consumerIndex];
                if(consumer == nullptr)
                {
                    continue;
                }

                for(const Node* producer : consumer->inputs)
                {
                    const auto found = indices.find(producer);
                    if(found != indices.end() &&
                       found->second >= consumerIndex)
                    {
                        addDiagnostic(
                            result,
                            "RG011",
                            "Internal producer '" + producer->name +
                                "' does not precede consumer '" +
                                consumer->name + "'.",
                            holder->id(),
                            consumer->name
                        );
                    }
                }
            }
        }
    }

    const auto membership = buildMembership(regionGraph);

    if(finalized && safeNodeTraversal)
    {
        for(const auto& holder : regionGraph.regions())
        {
            if(!equalNodes(
                holder->inputs(),
                expectedInputs(*holder, membership)
            ))
            {
                addDiagnostic(
                    result,
                    "RG012",
                    "Region input boundary does not match deterministic derivation.",
                    holder->id()
                );
            }
        }

        for(const auto& holder : regionGraph.regions())
        {
            if(!equalNodes(
                holder->outputs(),
                expectedOutputs(*holder, membership)
            ))
            {
                addDiagnostic(
                    result,
                    "RG013",
                    "Region output boundary does not match deterministic derivation.",
                    holder->id()
                );
            }
        }
    }

    for(const auto& dependency : regionGraph.dependencies())
    {
        if(regionGraph.findRegion(dependency.source) == nullptr)
        {
            addDiagnostic(
                result,
                "RG014",
                "RegionDependency source does not identify an existing Region.",
                dependency.source
            );
        }
        if(regionGraph.findRegion(dependency.destination) == nullptr)
        {
            addDiagnostic(
                result,
                "RG014",
                "RegionDependency destination does not identify an existing Region.",
                dependency.destination
            );
        }
    }

    for(const auto& dependency : regionGraph.dependencies())
    {
        if(dependency.source == dependency.destination)
        {
            addDiagnostic(
                result,
                "RG015",
                "RegionDependency source and destination are identical.",
                dependency.source
            );
        }
    }

    for(const auto& dependency : regionGraph.dependencies())
    {
        if(dependency.values.empty())
        {
            addDiagnostic(
                result,
                "RG016",
                "RegionDependency contains no crossing values.",
                dependency.source
            );
        }
    }

    std::set<DependencyKey> dependencyPairs;
    for(const auto& dependency : regionGraph.dependencies())
    {
        if(!dependencyPairs.insert({
            dependency.source,
            dependency.destination
        }).second)
        {
            addDiagnostic(
                result,
                "RG017",
                "Duplicate RegionDependency exists for the same endpoints.",
                dependency.source
            );
        }
    }

    for(const auto& dependency : regionGraph.dependencies())
    {
        const Region* source = regionGraph.findRegion(dependency.source);
        const Region* destination =
            regionGraph.findRegion(dependency.destination);

        for(const Node* value : dependency.values)
        {
            bool valid = value != nullptr &&
                         source != nullptr &&
                         destination != nullptr &&
                         graphNodes.contains(value) &&
                         source->contains(*value);

            bool consumed = false;
            if(valid && safeNodeTraversal)
            {
                for(const Node* consumer : destination->nodes())
                {
                    for(const Node* input : consumer->inputs)
                    {
                        if(input == value)
                        {
                            consumed = true;
                            break;
                        }
                    }
                    if(consumed)
                    {
                        break;
                    }
                }
            }

            if(!valid || !consumed)
            {
                addDiagnostic(
                    result,
                    "RG018",
                    "RegionDependency value is null, foreign, outside its source Region, or not consumed by its destination Region.",
                    dependency.source,
                    value != nullptr && graphNodes.contains(value)
                        ? std::optional<std::string>(value->name)
                        : std::nullopt
                );
            }
        }
    }

    for(const auto& dependency : regionGraph.dependencies())
    {
        std::unordered_set<const Node*> seen;
        for(const Node* value : dependency.values)
        {
            if(value != nullptr && !seen.insert(value).second)
            {
                addDiagnostic(
                    result,
                    "RG019",
                    "RegionDependency contains the same value more than once.",
                    dependency.source,
                    graphNodes.contains(value)
                        ? std::optional<std::string>(value->name)
                        : std::nullopt
                );
            }
        }
    }

    if(finalized && safeNodeTraversal)
    {
        const auto expected = expectedDependencies(
            graph,
            regionGraph,
            membership
        );
        if(!equalDependencies(regionGraph.dependencies(), expected))
        {
            addDiagnostic(
                result,
                "RG020",
                "RegionDependency order or content does not match deterministic derivation."
            );
        }
    }

    {
        std::vector<std::vector<RegionId>> adjacency(regionGraph.size());
        std::vector<std::size_t> indegree(regionGraph.size(), 0);

        for(const auto& dependency : regionGraph.dependencies())
        {
            if(dependency.source < regionGraph.size() &&
               dependency.destination < regionGraph.size())
            {
                adjacency[dependency.source].push_back(
                    dependency.destination
                );
                ++indegree[dependency.destination];
            }
        }

        std::priority_queue<
            RegionId,
            std::vector<RegionId>,
            std::greater<>
        > ready;
        for(RegionId id = 0; id < regionGraph.size(); ++id)
        {
            if(indegree[id] == 0)
            {
                ready.push(id);
            }
        }

        std::size_t visited = 0;
        while(!ready.empty())
        {
            const RegionId current = ready.top();
            ready.pop();
            ++visited;

            for(RegionId destination : adjacency[current])
            {
                --indegree[destination];
                if(indegree[destination] == 0)
                {
                    ready.push(destination);
                }
            }
        }

        if(visited != regionGraph.size())
        {
            addDiagnostic(
                result,
                "RG021",
                "Region dependency graph contains a cycle."
            );
        }
    }

    if(level == RegionVerificationLevel::Executable)
    {
        for(const auto& holder : regionGraph.regions())
        {
            if(holder->strategy() == RegionStrategy::Unresolved)
            {
                addDiagnostic(
                    result,
                    "RG022",
                    "Executable verification requires a resolved Region strategy.",
                    holder->id()
                );
            }
        }
    }

    result.valid = result.diagnostics.empty();
    return result;
}

}
