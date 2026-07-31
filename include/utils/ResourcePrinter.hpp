#pragma once

#include "analysis/ResourceAnalysis.hpp"

namespace thiran
{

class ResourcePrinter
{
public:

    static void print(
        Graph& graph,
        ResourceAnalysis& resources
    );

};

}