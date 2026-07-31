#pragma once

#include <vector>

#include "ir/Graph.hpp"
#include "optimizer/Pattern.hpp"

namespace thiran
{

class PatternMatcher
{
public:

    std::vector<Node*> match(
        Graph& graph,
        Pattern& pattern
    );

private:

    bool matchNode(
        Node* graphNode,
        PatternNode* patternNode
    );
};

}