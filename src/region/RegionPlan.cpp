#include "region/RegionPlan.hpp"

#include <utility>

#include "region/RegionPlanVerifier.hpp"

namespace thiran::region
{

std::string_view toString(RegionPlanDiagnosticStage stage) noexcept
{
    switch(stage)
    {
        case RegionPlanDiagnosticStage::Classification:
            return "CLASSIFICATION";
        case RegionPlanDiagnosticStage::Formation:
            return "FORMATION";
        case RegionPlanDiagnosticStage::Verification:
            return "VERIFICATION";
    }
    return "UNKNOWN";
}

RegionPlan::RegionPlan(
    const Graph& graph,
    std::vector<NodeStrategyDecision> decisions,
    std::unique_ptr<RegionGraph> regionGraph
)
    : sourceGraph_(&graph),
      decisions_(std::move(decisions)),
      regionGraph_(std::move(regionGraph))
{
    for(std::size_t index = 0; index < decisions_.size(); ++index)
    {
        if(decisions_[index].node != nullptr)
        {
            decisionIndices_.try_emplace(decisions_[index].node, index);
        }
    }
}

const Graph& RegionPlan::sourceGraph() const noexcept { return *sourceGraph_; }
const std::vector<NodeStrategyDecision>& RegionPlan::decisions() const noexcept
{
    return decisions_;
}
const RegionGraph& RegionPlan::regionGraph() const noexcept
{
    return *regionGraph_;
}
const NodeStrategyDecision* RegionPlan::decisionFor(const Node& node) const noexcept
{
    const auto found = decisionIndices_.find(&node);
    return found == decisionIndices_.end() ? nullptr : &decisions_[found->second];
}
const Region* RegionPlan::regionFor(const Node& node) const noexcept
{
    return regionGraph_->regionFor(node);
}
std::size_t RegionPlan::nodeCount() const noexcept { return decisions_.size(); }
std::size_t RegionPlan::regionCount() const noexcept { return regionGraph_->size(); }
bool RegionPlan::empty() const noexcept
{
    return decisions_.empty() && regionGraph_->empty();
}

bool RegionPlanResult::succeeded() const noexcept
{
    return plan != nullptr && diagnostics.empty();
}

RegionPlanResult HybridRegionPlanner::build(const Graph& graph)
{
    RegionPlanResult result{nullptr, {}};
    auto classification = BaselineStrategyClassifier::classify(graph);
    if(!classification.succeeded())
    {
        for(const auto& diagnostic : classification.diagnostics)
        {
            result.diagnostics.push_back({
                RegionPlanDiagnosticStage::Classification,
                diagnostic.code,
                diagnostic.message,
                diagnostic.nodeName,
                diagnostic.graphIndex,
                std::nullopt,
                diagnostic.dimensionIndex,
                std::nullopt
            });
        }
        return result;
    }

    auto formation =
        TopologicalRegionFormer::form(graph, classification.assignments());
    if(!formation.succeeded())
    {
        for(const auto& diagnostic : formation.diagnostics)
        {
            result.diagnostics.push_back({
                RegionPlanDiagnosticStage::Formation,
                diagnostic.code,
                diagnostic.message,
                diagnostic.nodeName,
                std::nullopt,
                diagnostic.assignmentIndex,
                std::nullopt,
                std::nullopt
            });
        }
        return result;
    }

    result.plan.reset(new RegionPlan(
        graph,
        std::move(classification.decisions),
        std::move(formation.regionGraph)
    ));
    auto verification = RegionPlanVerifier::verify(graph, *result.plan);
    if(!verification.valid)
    {
        result.plan.reset();
        result.diagnostics = std::move(verification.diagnostics);
    }
    return result;
}

}
