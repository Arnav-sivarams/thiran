#include "optimizer/FusionRule.hpp"

#include <vector>

namespace thiran
{

bool FusionRule::run(Graph& graph)
{
    bool changed = false;

    std::vector<Node*> candidates;
    for(auto& node : graph.nodes)
    {
        candidates.push_back(node.get());
    }

    for(auto node : candidates)
    {
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
        graph.disconnect(node, next);
        node->op = Operation::FusedMatMulRelu;

        for(auto consumer : consumers)
        {
            graph.disconnect(next, consumer);
            graph.connect(node, consumer);
        }

        graph.removeNode(next);

        changed = true;
    }

    return changed;
}

}
