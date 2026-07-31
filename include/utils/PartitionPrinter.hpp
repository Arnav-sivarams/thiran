#pragma once

#include "distributed/PartitionGraph.hpp"

namespace thiran
{

class PartitionPrinter
{
public:

    static void print(
        PartitionGraph& graph
    );

};

}