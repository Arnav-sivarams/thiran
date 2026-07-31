#include "utils/RegionPlanDiagnosticPrinter.hpp"

namespace thiran
{

void RegionPlanDiagnosticPrinter::print(
    const std::vector<region::RegionPlanDiagnostic>& diagnostics,
    std::ostream& output
)
{
    output << "Region plan diagnostics\n";
    if(diagnostics.empty())
    {
        output << "  <none>\n";
        return;
    }

    for(const auto& diagnostic : diagnostics)
    {
        output << "  [" << region::toString(diagnostic.stage)
               << "][" << diagnostic.code << "]";

        bool hasMetadata = false;
        const auto separator = [&]()
        {
            output << (hasMetadata ? " " : " ");
            hasMetadata = true;
        };

        if(diagnostic.nodeName.has_value())
        {
            separator();
            output << "node=" << *diagnostic.nodeName;
        }
        if(diagnostic.graphIndex.has_value())
        {
            separator();
            output << "graph-index=" << *diagnostic.graphIndex;
        }
        if(diagnostic.assignmentIndex.has_value())
        {
            separator();
            output << "assignment-index="
                   << *diagnostic.assignmentIndex;
        }
        if(diagnostic.dimensionIndex.has_value())
        {
            separator();
            output << "dimension-index=" << *diagnostic.dimensionIndex;
        }
        if(diagnostic.regionId.has_value())
        {
            separator();
            output << "region=" << *diagnostic.regionId;
        }

        output << ": " << diagnostic.message << "\n";
    }
}

}
