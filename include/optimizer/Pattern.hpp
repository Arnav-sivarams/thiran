#pragma once

#include <memory>
#include <vector>

#include "optimizer/PatternNode.hpp"

namespace thiran
{

class Pattern
{
public:

    std::vector<std::shared_ptr<PatternNode>> nodes;

    PatternNode* root;

    Pattern();

    PatternNode* createNode(Operation op);

    void connect(
        PatternNode* from,
        PatternNode* to
    );
};

}