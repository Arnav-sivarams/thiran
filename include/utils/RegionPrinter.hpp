#pragma once

#include <ostream>

#include "region/RegionGraph.hpp"

namespace thiran
{

class RegionPrinter final
{
public:

    static void print(
        const region::RegionGraph& regionGraph,
        std::ostream& output
    );
};

}
