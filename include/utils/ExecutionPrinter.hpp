#pragma once

#include "scheduler/ExecutionGraph.hpp"

namespace thiran
{

class ExecutionPrinter
{
public:

    static void print(
        ExecutionGraph& graph
    );

};

}