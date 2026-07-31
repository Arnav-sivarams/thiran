#include "optimizer/RewriteEngine.hpp"

#include <iostream>

namespace thiran
{

void RewriteEngine::addRule(std::shared_ptr<RewriteRule> rule)
{
    rules.push_back(rule);
}

void RewriteEngine::run(Graph& graph)
{
    std::cout << "Running Rewrite Engine\n";

    for(auto& node : graph.nodes)
    {
        for(auto& rule : rules)
        {
            if(rule->match(node.get()))
            {
                rule->rewrite(graph, node.get());
            }
        }
    }
}

}