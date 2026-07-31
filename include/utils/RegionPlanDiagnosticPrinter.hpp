#pragma once

#include <ostream>
#include <vector>

#include "region/RegionPlan.hpp"

namespace thiran
{

class RegionPlanDiagnosticPrinter final
{
public:

    static void print(
        const std::vector<region::RegionPlanDiagnostic>& diagnostics,
        std::ostream& output
    );
};

}
