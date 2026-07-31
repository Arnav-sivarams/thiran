#include "scheduler/ExecutionNode.hpp"

namespace thiran
{

ExecutionNode::ExecutionNode(Node* node)
{
    this->node = node;

    stage = 0;

    device = Device::Unknown;

    parallel = false;
}

}