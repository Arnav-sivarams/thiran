#include "optimizer/RewriteEngine.hpp"

#include <iostream>
#include <stdexcept>
#include "optimizer/GraphVerifier.hpp"

namespace thiran
{

void RewriteEngine::addRule(std::shared_ptr<RewriteRule> rule)
{
    rules.push_back(rule);
}

void RewriteEngine::run(Graph& graph)
{
    std::cout << "Running Rewrite Engine\n";
    // One successful rewrite ends a round. The next round observes the current
    // graph in vector order, including newly created nodes. Owning snapshots
    // keep every candidate alive even when a rule erases it from graph.nodes.
    constexpr std::size_t maxRewrites = 10000;
    std::size_t rewrites = 0;
    while(true)
    {
        const auto candidates = graph.nodes;
        bool changed = false;
        for(const auto& holder : candidates)
        {
            Node* node = holder.get();
            if(!graph.containsNode(node))
            {
                continue;
            }
            for(const auto& rule : rules)
            {
                if(rule->match(node) && rule->rewrite(graph, node))
                {
                    if(!GraphVerifier{}.verify(graph))
                    {
                        throw std::runtime_error("rewrite produced an invalid graph");
                    }
                    if(++rewrites > maxRewrites)
                    {
                        throw std::runtime_error("rewrite engine did not reach a fixed point");
                    }
                    changed = true;
                    break;
                }
            }
            if(changed)
            {
                break;
            }
        }
        if(!changed)
        {
            return;
        }
    }
}

}
