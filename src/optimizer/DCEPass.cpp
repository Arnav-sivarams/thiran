#include "optimizer/DCEPass.hpp"

namespace thiran
{

std::string DCEPass::getName()
{
    return "Dead Node Elimination";
}

void DCEPass::run(Graph& graph)
{
    bool changed = true;
    while(changed)
    {
        changed = false;
        std::vector<Node*> dead;
        for(auto& holder : graph.nodes)
        {
            Node* node = holder.get();
            if(node->outputs.empty() && node->op != Operation::Output && node->op != Operation::Input)
            {
                dead.push_back(node);
            }
        }
        for(auto node : dead)
        {
            graph.removeNode(node);
            changed = true;
        }
    }
}

}
