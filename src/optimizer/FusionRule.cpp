#include "optimizer/FusionRule.hpp"

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace thiran
{

bool FusionRule::run(Graph& graph)
{
    bool changed = false;

    const auto candidates = graph.nodes;

    for(const auto& holder : candidates)
    {
        Node* node = holder.get();
        if(!graph.containsNode(node))
        {
            continue;
        }
        if(node->op != Operation::MatMul)
        {
            continue;
        }

        if(node->outputs.size() != 1)
        {
            continue;
        }

        Node* next = node->outputs[0];

        if(next->op != Operation::ReLU)
        {
            continue;
        }

        if(next->inputs.size() != 1)
        {
            continue;
        }

        const auto consumers = next->outputs;
        // The Graph cannot represent the same producer in two operand slots.
        if(std::any_of(consumers.begin(), consumers.end(),
            [node](Node* consumer)
            {
                return std::find(consumer->inputs.begin(), consumer->inputs.end(), node)
                    != consumer->inputs.end();
            }))
        {
            continue;
        }

        for(auto consumer : consumers)
        {
            if(!graph.replaceInput(consumer, next, node))
            {
                throw std::runtime_error("fusion input replacement failed");
            }
        }

        node->op = Operation::FusedMatMulRelu;
        graph.removeNode(next);

        changed = true;
    }

    return changed;
}

}
