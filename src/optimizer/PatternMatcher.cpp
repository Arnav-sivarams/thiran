#include "optimizer/PatternMatcher.hpp"

namespace thiran
{

std::vector<Node*> PatternMatcher::match(
    Graph& graph,   
    Pattern& pattern
)
{
    std::vector<Node*> matches;

    for(auto& node : graph.nodes)
    {
        if(matchNode(node.get(), pattern.root))
        {
            matches.push_back(node.get());
        }
    }

    return matches;
}

bool PatternMatcher::matchNode(
    Node* graphNode,
    PatternNode* patternNode
)
{
    if(graphNode == nullptr)
    {
        return false;
    }

    if(patternNode == nullptr)
    {
        return false;
    }

    return graphNode->op == patternNode->op;
}

}