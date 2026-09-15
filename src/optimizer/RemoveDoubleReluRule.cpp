#include "optimizer/RemoveDoubleReluRule.hpp"

#include <iostream>
#include <algorithm>
#include <stdexcept>

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

    if(node->inputs.size() != 1 || node->inputs[0] == nullptr)
    {
        return false;
    }

    Node* previous = node->inputs[0];
    if(previous->op != Operation::ReLU)
    {
        return false;
    }
    // Graph edges are unique producer/consumer pairs. Replacing this input
    // cannot merge two operand positions into one edge.
    for(Node* consumer : node->outputs)
    {
        if(std::find(consumer->inputs.begin(), consumer->inputs.end(), previous)
            != consumer->inputs.end())
        {
            return false;
        }
    }
    return true;
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
        if(!graph.replaceInput(consumer, node, previous))
        {
            throw std::runtime_error("double-ReLU input replacement failed");
        }
    }

    graph.removeNode(node);
    return true;
}

}
