#pragma once

#include "backend/BackendIR.hpp"
#include "scheduler/ExecutionGraph.hpp"

namespace thiran
{

class BackendBuilder
{
public:

    BackendIR build(
        ExecutionGraph& graph
    );

};

}