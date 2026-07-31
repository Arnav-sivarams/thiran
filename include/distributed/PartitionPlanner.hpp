#pragma once

#include "scheduler/ExecutionGraph.hpp"
#include "distributed/PartitionGraph.hpp"

namespace thiran
{

class PartitionPlanner
{
public:

    PartitionGraph partition(
        ExecutionGraph& graph
    );

};

}