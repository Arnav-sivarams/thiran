#pragma once

#include <ostream>
#include <vector>

#include "region/RegionPythonEmitter.hpp"

namespace thiran
{

class RegionExecutionDiagnosticPrinter final
{
public:
    static void print(
        const std::vector<region::RegionExecutionDiagnostic>& diagnostics,
        std::ostream& output
    );
};

}
