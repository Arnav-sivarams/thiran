#pragma once

#include "distributed/PartitionGraph.hpp"
#include "distributed/CommunicationGraph.hpp"

namespace thiran
{

class CommunicationPlanner
{
public:

    CommunicationGraph build(
        PartitionGraph& graph
    );

};

}