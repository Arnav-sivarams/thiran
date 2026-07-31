#include "scheduler/ExecutionGraph.hpp"

namespace thiran
{

void ExecutionGraph::clear()
{
    nodes.clear();
}

void ExecutionGraph::add(Node* node)
{
    nodes.push_back(
        std::make_shared<ExecutionNode>(node)
    );
}

}