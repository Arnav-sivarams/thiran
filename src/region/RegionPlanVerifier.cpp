#include "region/RegionPlanVerifier.hpp"

#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "region/RegionVerifier.hpp"

namespace thiran::region
{
namespace
{

void add(
    RegionPlanVerificationResult& result,
    std::string code,
    std::string message,
    std::optional<std::string> nodeName = std::nullopt,
    std::optional<std::size_t> graphIndex = std::nullopt,
    std::optional<std::size_t> assignmentIndex = std::nullopt,
    std::optional<std::size_t> dimensionIndex = std::nullopt,
    std::optional<RegionId> regionId = std::nullopt
)
{
    result.diagnostics.push_back({
        RegionPlanDiagnosticStage::Verification,
        std::move(code),
        std::move(message),
        std::move(nodeName),
        graphIndex,
        assignmentIndex,
        dimensionIndex,
        regionId
    });
}

using RegionDiagnosticKey = std::tuple<
    std::string, std::string, std::optional<RegionId>,
    std::optional<std::string>>;

bool equalRegionGraphs(const RegionGraph& left, const RegionGraph& right)
{
    if(left.finalized() != right.finalized() ||
       left.regions().size() != right.regions().size() ||
       left.dependencies().size() != right.dependencies().size())
    {
        return false;
    }
    for(std::size_t index = 0; index < left.regions().size(); ++index)
    {
        const Region& a = *left.regions()[index];
        const Region& b = *right.regions()[index];
        if(a.id() != b.id() || a.strategy() != b.strategy() ||
           a.nodes() != b.nodes() || a.inputs() != b.inputs() ||
           a.outputs() != b.outputs())
        {
            return false;
        }
    }
    for(std::size_t index = 0; index < left.dependencies().size(); ++index)
    {
        const auto& a = left.dependencies()[index];
        const auto& b = right.dependencies()[index];
        if(a.source != b.source || a.destination != b.destination ||
           a.values != b.values)
        {
            return false;
        }
    }
    return true;
}

}

RegionPlanVerificationResult RegionPlanVerifier::verify(
    const Graph& graph,
    const RegionPlan& plan
)
{
    RegionPlanVerificationResult result{true, {}};

    if(&graph != &plan.sourceGraph())
        add(result, "RP001", "RegionPlan belongs to another Graph.");

    if(plan.decisions().size() != graph.nodes.size())
        add(result, "RP002",
            "RegionPlan decision count " +
            std::to_string(plan.decisions().size()) +
            " differs from Graph Node count " +
            std::to_string(graph.nodes.size()) + ".");

    std::unordered_set<const Node*> graphNodes;
    for(const auto& holder : graph.nodes)
        if(holder != nullptr) graphNodes.insert(holder.get());

    for(std::size_t index = 0; index < plan.decisions().size(); ++index)
        if(plan.decisions()[index].node == nullptr)
            add(result, "RP003", "Decision contains a null Node pointer.",
                std::nullopt, index);

    for(std::size_t index = 0; index < plan.decisions().size(); ++index)
    {
        const Node* node = plan.decisions()[index].node;
        if(node != nullptr && !graphNodes.contains(node))
            add(result, "RP004",
                "Decision Node is not owned by the supplied Graph.",
                std::nullopt, index);
    }

    const std::size_t common =
        std::min(plan.decisions().size(), graph.nodes.size());
    for(std::size_t index = 0; index < common; ++index)
    {
        const Node* node = plan.decisions()[index].node;
        const Node* expected = graph.nodes[index].get();
        if(node != expected)
        {
            std::optional<std::string> name;
            if(node != nullptr && graphNodes.contains(node)) name = node->name;
            else if(expected != nullptr) name = expected->name;
            add(result, "RP005",
                "Decision Node does not match Graph storage order.",
                std::move(name), index);
        }
    }

    std::unordered_set<const Node*> seen;
    for(std::size_t index = 0; index < plan.decisions().size(); ++index)
    {
        const Node* node = plan.decisions()[index].node;
        if(node != nullptr && !seen.insert(node).second)
        {
            std::optional<std::string> name;
            if(graphNodes.contains(node)) name = node->name;
            add(result, "RP006", "Decision Node is duplicated.",
                std::move(name), index);
        }
    }

    for(std::size_t index = 0; index < plan.decisions().size(); ++index)
    {
        const auto& decision = plan.decisions()[index];
        if(decision.strategy == RegionStrategy::Unresolved)
        {
            std::optional<std::string> name;
            if(decision.node != nullptr && graphNodes.contains(decision.node))
                name = decision.node->name;
            add(result, "RP007", "Decision strategy is unresolved.",
                std::move(name), index);
        }
    }

    if(&plan.regionGraph().sourceGraph() != &graph)
        add(result, "RP008", "RegionGraph belongs to another Graph.");

    std::vector<RegionDiagnosticKey> emitted;
    for(const auto level : {RegionVerificationLevel::Structural,
                            RegionVerificationLevel::Executable})
    {
        const auto verification =
            RegionVerifier::verify(graph, plan.regionGraph(), level);
        for(const auto& diagnostic : verification.diagnostics)
        {
            RegionDiagnosticKey key{
                diagnostic.code, diagnostic.message,
                diagnostic.regionId, diagnostic.nodeName};
            if(std::find(emitted.begin(), emitted.end(), key) != emitted.end())
                continue;
            emitted.push_back(key);
            add(result, "RP009",
                "RegionVerifier " + diagnostic.code + ": " +
                diagnostic.message,
                diagnostic.nodeName, std::nullopt, std::nullopt,
                std::nullopt, diagnostic.regionId);
        }
    }

    for(std::size_t index = 0; index < plan.decisions().size(); ++index)
    {
        const Node* node = plan.decisions()[index].node;
        if(node != nullptr && graphNodes.contains(node) &&
           plan.regionFor(*node) == nullptr)
            add(result, "RP010", "Decision Node has no Region.",
                node->name, index);
    }

    for(const auto& decision : plan.decisions())
    {
        const Node* node = decision.node;
        if(node == nullptr || !graphNodes.contains(node)) continue;
        const Region* region = plan.regionFor(*node);
        if(region != nullptr && decision.strategy != region->strategy())
            add(result, "RP011",
                "Decision strategy " + std::string(toString(decision.strategy)) +
                " differs from Region strategy " +
                std::string(toString(region->strategy())) + ".",
                node->name, std::nullopt, std::nullopt, std::nullopt,
                region->id());
    }

    auto replay = BaselineStrategyClassifier::classify(graph);
    if(!replay.succeeded())
    {
        for(const auto& diagnostic : replay.diagnostics)
            add(result, "RP012",
                "Classifier " + diagnostic.code + ": " + diagnostic.message,
                diagnostic.nodeName, diagnostic.graphIndex, std::nullopt,
                diagnostic.dimensionIndex);
    }
    else
    {
        const std::size_t positions =
            std::max(plan.decisions().size(), replay.decisions.size());
        for(std::size_t index = 0; index < positions; ++index)
        {
            bool mismatch =
                index >= plan.decisions().size() ||
                index >= replay.decisions.size();
            if(!mismatch)
            {
                const auto& stored = plan.decisions()[index];
                const auto& current = replay.decisions[index];
                mismatch = stored.node != current.node ||
                    stored.strategy != current.strategy ||
                    stored.shapeKnowledge != current.shapeKnowledge ||
                    stored.reason != current.reason;
            }
            if(mismatch)
            {
                std::optional<std::string> name;
                if(index < replay.decisions.size() &&
                   replay.decisions[index].node != nullptr &&
                   graphNodes.contains(replay.decisions[index].node))
                    name = replay.decisions[index].node->name;
                add(result, "RP013",
                    "Stored decision differs from classification replay.",
                    std::move(name), index);
            }
        }

        auto formation =
            TopologicalRegionFormer::form(graph, replay.assignments());
        if(!formation.succeeded())
        {
            for(const auto& diagnostic : formation.diagnostics)
                add(result, "RP014",
                    "Region formation " + diagnostic.code + ": " +
                    diagnostic.message,
                    diagnostic.nodeName, std::nullopt,
                    diagnostic.assignmentIndex);
        }
        else if(!equalRegionGraphs(plan.regionGraph(), *formation.regionGraph))
        {
            add(result, "RP014",
                "Stored RegionGraph differs from formation replay.");
        }
    }

    result.valid = result.diagnostics.empty();
    return result;
}

}
