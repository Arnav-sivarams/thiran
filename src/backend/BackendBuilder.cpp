#include "backend/BackendBuilder.hpp"

#include <algorithm>
#include <memory>
#include <unordered_set>

namespace thiran
{

BackendIR BackendBuilder::build(
    ExecutionGraph& graph
)
{
    BackendIR backend;

    int id = 0;

    std::shared_ptr<Kernel> current;

    for(size_t i = 0; i < graph.nodes.size(); i++)
    {
        auto exec = graph.nodes[i];

        if(!current)
        {
            current =
                std::make_shared<Kernel>();

            current->name =
                "kernel_" + std::to_string(id++) + "_" +
                (exec->device == Device::GPU ? "gpu" : "cpu");

            current->device =
                exec->device;
        }

        current->operations.push_back(
            exec->node
        );

        bool endKernel = true;

        if(i + 1 < graph.nodes.size())
        {
            auto next = graph.nodes[i + 1];

            if(exec->device == next->device)
            {
                if(exec->node->op == Operation::MatMul &&
                next->node->op == Operation::ReLU)
                {
                    if(!next->node->inputs.empty() &&
                    next->node->inputs[0] == exec->node)
                    {
                        endKernel = false;
                    }
                }
            }
        }

        if(endKernel)
        {
            backend.kernels.push_back(
                current
            );

            current = nullptr;
        }
    }

    for(auto& kernel : backend.kernels)
    {
        std::unordered_set<Node*> members(
            kernel->operations.begin(),
            kernel->operations.end()
        );

        for(auto operation : kernel->operations)
        {
            for(auto input : operation->inputs)
            {
                if(!members.count(input) &&
                   std::find(kernel->inputs.begin(), kernel->inputs.end(), input) == kernel->inputs.end())
                {
                    kernel->inputs.push_back(input);
                }
            }

            bool escapes = operation->op == Operation::Output;
            for(auto output : operation->outputs)
            {
                escapes = escapes || !members.count(output);
            }
            if(escapes)
            {
                kernel->outputs.push_back(operation);
            }
        }
    }

    return backend;
}

}
