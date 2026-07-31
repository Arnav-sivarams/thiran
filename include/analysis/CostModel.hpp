#pragma once

#include <unordered_map>

#include "ir/Graph.hpp"

namespace thiran
{

class CostModel
{
public:

    CostModel();

    int operationCost(Operation op);

    int graphCost(Graph& graph);

private:

    std::unordered_map<Operation,int> costs;

};

}