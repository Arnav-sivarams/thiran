#pragma once

#include <string>

#include "ir/Graph.hpp"

namespace thiran
{

class RewriteRule
{
public:

    virtual ~RewriteRule() = default;

    virtual std::string getName() = 0;

    virtual bool match(Node* node) = 0;

    virtual bool rewrite(Graph& graph, Node* node) = 0;
};

}