#pragma once

#include <memory>
#include <vector>

#include "optimizer/RewriteRule.hpp"

namespace thiran
{

class RewriteEngine
{
private:

    std::vector<std::shared_ptr<RewriteRule>> rules;

public:

    void addRule(std::shared_ptr<RewriteRule> rule);

    void run(Graph& graph);
};

}