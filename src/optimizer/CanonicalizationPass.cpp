#include "optimizer/CanonicalizationPass.hpp"

#include <algorithm>
#include <iostream>

namespace thiran
{

std::string CanonicalizationPass::getName()
{
    return "Canonicalization";
}

void CanonicalizationPass::sortInputs(Node* node)
{
    std::sort(
        node->inputs.begin(),
        node->inputs.end(),
        [](Node* a, Node* b)
        {
            return a->name < b->name;
        }
    );
}

void CanonicalizationPass::run(Graph& graph)
{
    std::cout
        << "Running "
        << getName()
        << "\n";

    for(auto& node : graph.nodes)
    {
        switch(node->op)
        {
            case Operation::Add:
            case Operation::Multiply:
                sortInputs(node.get());
                break;

            default:
                break;
        }
    }
}

}