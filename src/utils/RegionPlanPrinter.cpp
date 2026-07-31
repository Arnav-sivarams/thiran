#include "utils/RegionPlanPrinter.hpp"

namespace thiran
{
namespace
{
void printNodes(const std::vector<const Node*>& nodes, std::ostream& output)
{
    if(nodes.empty())
    {
        output << "<none>";
        return;
    }
    for(std::size_t index = 0; index < nodes.size(); ++index)
    {
        if(index != 0) output << ", ";
        output << nodes[index]->name;
    }
}
}

void RegionPlanPrinter::print(
    const region::RegionPlan& plan,
    std::ostream& output
)
{
    output << "HybridRegionPlan\n"
           << "  Nodes: " << plan.nodeCount() << "\n"
           << "  Regions: " << plan.regionCount() << "\n"
           << "  Dependencies: "
           << plan.regionGraph().dependencies().size() << "\n\n"
           << "Decisions\n";
    if(plan.decisions().empty()) output << "  <none>\n";
    for(std::size_t index = 0; index < plan.decisions().size(); ++index)
    {
        const auto& decision = plan.decisions()[index];
        const region::Region* owner =
            decision.node == nullptr ? nullptr : plan.regionFor(*decision.node);
        output << "  [" << index << "] " << decision.node->name
               << " | Shape: " << region::toString(decision.shapeKnowledge)
               << " | Strategy: " << region::toString(decision.strategy)
               << " | Reason: " << region::toString(decision.reason)
               << " | Region: ";
        if(owner == nullptr) output << "<none>";
        else output << owner->id();
        output << "\n";
    }

    output << "\nRegions\n";
    if(plan.regionGraph().regions().empty()) output << "  <none>\n";
    for(const auto& holder : plan.regionGraph().regions())
    {
        const auto& value = *holder;
        output << "  Region " << value.id() << " ["
               << region::toString(value.strategy()) << "]\n"
               << "    Nodes: ";
        printNodes(value.nodes(), output);
        output << "\n    Inputs: ";
        printNodes(value.inputs(), output);
        output << "\n    Outputs: ";
        printNodes(value.outputs(), output);
        output << "\n";
    }

    output << "\nDependencies\n";
    if(plan.regionGraph().dependencies().empty()) output << "  <none>\n";
    for(const auto& dependency : plan.regionGraph().dependencies())
    {
        output << "  " << dependency.source << " -> "
               << dependency.destination << ": ";
        printNodes(dependency.values, output);
        output << "\n";
    }
}
}
