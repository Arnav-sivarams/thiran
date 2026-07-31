#pragma once

#include "ir/Node.hpp"
#include "scheduler/Device.hpp"

namespace thiran
{

class ExecutionNode
{
public:

    Node* node;

    int stage;

    Device device;

    bool parallel;

    ExecutionNode(Node* node);

};

}