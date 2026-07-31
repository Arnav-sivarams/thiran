#include "scheduler/DevicePlanner.hpp"

namespace thiran
{

void DevicePlanner::assign(
    ExecutionGraph& graph
)
{
    for(auto& node : graph.nodes)
    {
        switch(node->node->op)
        {
            case Operation::MatMul:

            case Operation::FusedMatMulRelu:

            case Operation::Conv2D:

                node->device = Device::GPU;

                break;

            case Operation::ReLU:

                node->device = Device::GPU;

                break;

            case Operation::Multiply:

                node->device = Device::GPU;

                break;

            case Operation::Add:

            case Operation::Subtract:

            case Operation::Divide:

            case Operation::Sigmoid:

            case Operation::Tanh:

            case Operation::Softmax:

            case Operation::MaxPool:

            case Operation::AvgPool:

                node->device = Device::GPU;

                break;

            case Operation::Input:

                node->device = Device::CPU;

                break;

            default:

                node->device = Device::CPU;

                break;
        }

        node->node->device = node->device;
    }
}

}
