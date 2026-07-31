#include "utils/RegionPrinter.hpp"

#include <string_view>

namespace thiran
{

namespace
{

void printNodes(
    const std::vector<const Node*>& nodes,
    std::ostream& output
)
{
    if(nodes.empty())
    {
        output << "<none>";
        return;
    }

    for(std::size_t i = 0; i < nodes.size(); ++i)
    {
        if(i != 0)
        {
            output << ", ";
        }
        output << nodes[i]->name;
    }
}

}

void RegionPrinter::print(
    const region::RegionGraph& regionGraph,
    std::ostream& output
)
{
    output << "RegionGraph\n";
    output << "  Regions: " << regionGraph.size() << "\n";
    output << "  Dependencies: "
           << regionGraph.dependencies().size() << "\n\n";

    for(std::size_t i = 0; i < regionGraph.regions().size(); ++i)
    {
        const region::Region& region = *regionGraph.regions()[i];
        output << "Region " << region.id() << " ["
               << region::toString(region.strategy()) << "]\n";
        output << "  Nodes: ";
        printNodes(region.nodes(), output);
        output << "\n  Inputs: ";
        printNodes(region.inputs(), output);
        output << "\n  Outputs: ";
        printNodes(region.outputs(), output);
        output << "\n\n";
    }

    output << "Dependencies\n";
    if(regionGraph.dependencies().empty())
    {
        output << "  <none>\n";
        return;
    }

    for(const auto& dependency : regionGraph.dependencies())
    {
        output << "  " << dependency.source << " -> "
               << dependency.destination << ": ";
        printNodes(dependency.values, output);
        output << "\n";
    }
}

}
