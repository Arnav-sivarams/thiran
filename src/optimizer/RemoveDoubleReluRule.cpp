#include "optimizer/RemoveDoubleReluRule.hpp"

#include <iostream>

namespace thiran
{

std::string RemoveDoubleReluRule::getName()
{
    return "RemoveDoubleRelu";
}

bool RemoveDoubleReluRule::match(Node* node)
{
    if(node->op != Operation::ReLU)
    {
        return false;
    }

    if(node->inputs.empty())
    {
        return false;
    }

    return node->inputs[0]->op == Operation::ReLU;
}

bool RemoveDoubleReluRule::rewrite(Graph& graph, Node* node)
{
    std::cout
        << "Matched "
        << getName()
        << " at "
        << node->name
        << "\n";

    Node* previous = node->inputs[0];
    const auto consumers = node->outputs;

    for(auto consumer : consumers)
    {
        graph.disconnect(node, consumer);
        graph.connect(previous, consumer);
    }

    graph.removeNode(node);
    return true;
}

}
