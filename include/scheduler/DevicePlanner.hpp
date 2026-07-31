#pragma once

#include "scheduler/ExecutionGraph.hpp"

namespace thiran
{

class DevicePlanner
{
public:

    void assign(
        ExecutionGraph& graph
    );

};

}