#include "scheduler/ExecutionPlanner.hpp"

namespace thiran
{

ExecutionGraph ExecutionPlanner::build(Graph& graph)
{
    DependencyAnalysis dependency;

    auto levels =
        dependency.executionLevels(graph);

    ExecutionGraph execution;

    for(size_t stage = 0; stage < levels.size(); stage++)
    {
        for(auto node : levels[stage])
        {
            execution.add(node);

            execution.nodes.back()->stage = stage;

            execution.nodes.back()->parallel =
                levels[stage].size() > 1;
        }
    }

    return execution;
}

}