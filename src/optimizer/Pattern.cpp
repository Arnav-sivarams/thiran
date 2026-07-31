#include "optimizer/Pattern.hpp"

namespace thiran
{

Pattern::Pattern()
{
    root = nullptr;
}

PatternNode* Pattern::createNode(Operation op)
{
    auto node =
        std::make_shared<PatternNode>(op);

    if(root == nullptr)
    {
        root = node.get();
    }

    nodes.push_back(node);

    return node.get();
}

void Pattern::connect(
    PatternNode* from,
    PatternNode* to
)
{
    from->connect(to);
}

}