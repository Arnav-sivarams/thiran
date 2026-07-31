#pragma once

#include <vector>

#include "scheduler/ExecutionNode.hpp"

namespace thiran
{

class Partition
{
public:

    int id;

    std::vector<ExecutionNode*> nodes;

    Partition(int id);

};

}