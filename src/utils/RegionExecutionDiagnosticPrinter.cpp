#include "utils/RegionExecutionDiagnosticPrinter.hpp"

namespace thiran
{

void RegionExecutionDiagnosticPrinter::print(
    const std::vector<region::RegionExecutionDiagnostic>& diagnostics,
    std::ostream& output
)
{
    output << "Region executor diagnostics\n";
    if(diagnostics.empty())
    {
        output << "  <none>\n";
        return;
    }

    for(const auto& diagnostic : diagnostics)
    {
        output << "  [" << diagnostic.code << "]";
        if(diagnostic.regionId.has_value())
        {
            output << " region=" << *diagnostic.regionId;
        }
        if(diagnostic.nodeName.has_value())
        {
            output << " node=" << *diagnostic.nodeName;
        }
        output << ": " << diagnostic.message << "\n";
    }
}

}
