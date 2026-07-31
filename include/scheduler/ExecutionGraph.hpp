#pragma once

#include <memory>
#include <vector>

#include "scheduler/ExecutionNode.hpp"

namespace thiran
{

class ExecutionGraph
{
public:

    std::vector<std::shared_ptr<ExecutionNode>> nodes;

    void clear();

    void add(Node* node);

};

}