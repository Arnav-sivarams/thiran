#pragma once

#include "optimizer/RewriteRule.hpp"

namespace thiran
{

class RemoveDoubleReluRule : public RewriteRule
{
public:

    std::string getName() override;

    bool match(Node* node) override;

    bool rewrite(Graph& graph, Node* node) override;
};

}