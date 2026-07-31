#pragma once

#include "analysis/DependencyAnalysis.hpp"
#include "scheduler/ExecutionGraph.hpp"

namespace thiran
{

class ExecutionPlanner
{
public:

    ExecutionGraph build(Graph& graph);

};

}