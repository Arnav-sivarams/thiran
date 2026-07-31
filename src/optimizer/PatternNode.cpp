#include "optimizer/PatternNode.hpp"

namespace thiran
{

PatternNode::PatternNode(Operation op)
{
    this->op = op;
}

void PatternNode::connect(PatternNode* node)
{
    outputs.push_back(node);
}

}