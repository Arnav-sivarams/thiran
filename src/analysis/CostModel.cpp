#include "analysis/CostModel.hpp"

namespace thiran
{

CostModel::CostModel()
{
    costs[Operation::Input] = 0;

    costs[Operation::Add] = 2;

    costs[Operation::Multiply] = 3;

    costs[Operation::MatMul] = 100;

    costs[Operation::ReLU] = 5;

    costs[Operation::Unknown] = 0;
}

int CostModel::operationCost(Operation op)
{
    return costs[op];
}

int CostModel::graphCost(Graph& graph)
{
    int total = 0;

    for(auto& node : graph.nodes)
    {
        total += operationCost(node->op);
    }

    return total;
}

}