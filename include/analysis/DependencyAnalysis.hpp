#pragma once

#include <unordered_map>
#include <vector>

#include "ir/Graph.hpp"

namespace thiran
{

class DependencyAnalysis
{
public:

    std::vector<std::vector<Node*>> executionLevels(Graph& graph);

private:

    std::unordered_map<Node*,int> indegree(Graph& graph);

};

}