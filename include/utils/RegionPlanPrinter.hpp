#pragma once

#include <ostream>

#include "region/RegionPlan.hpp"

namespace thiran
{

class RegionPlanPrinter final {
public:
    static void print(
        const region::RegionPlan& plan,
        std::ostream& output
    );
};

}
