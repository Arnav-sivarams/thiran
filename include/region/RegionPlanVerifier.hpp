#pragma once

#include <vector>

#include "region/RegionPlan.hpp"

namespace thiran::region
{

struct RegionPlanVerificationResult final {
    bool valid;
    std::vector<RegionPlanDiagnostic> diagnostics;
};

class RegionPlanVerifier final {
public:
    static RegionPlanVerificationResult verify(
        const Graph& graph,
        const RegionPlan& plan
    );
};

}
