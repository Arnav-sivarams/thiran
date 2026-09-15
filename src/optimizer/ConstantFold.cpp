#include "optimizer/ConstantFold.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace thiran
{

namespace
{

bool isConstant(Node* node)
{
    return node && node->op == Operation::Constant && node->shape.empty();
}

bool replaceWithInput(Graph& graph, Node* node, Node* replacement)
{
    if(!graph.containsNode(node) || !graph.containsNode(replacement) ||
       node == replacement)
    {
        return false;
    }

    const auto consumers = node->outputs;
    // Preflight the entire fan-out: rejecting one unrepresentable consumer must
    // not leave earlier consumers rewired or the folded node removed.
    for(Node* consumer : consumers)
    {
        const auto edgeCount = std::count_if(graph.edges.begin(), graph.edges.end(),
            [node, consumer](const Edge& edge)
            {
                return edge.source == node && edge.destination == consumer;
            });
        if(!graph.containsNode(consumer) ||
           std::count(consumer->inputs.begin(), consumer->inputs.end(), node) != 1 ||
           std::count(node->outputs.begin(), node->outputs.end(), consumer) != 1 ||
           edgeCount != 1 ||
           std::find(consumer->inputs.begin(), consumer->inputs.end(), replacement) != consumer->inputs.end() ||
           std::find(replacement->outputs.begin(), replacement->outputs.end(), consumer) != replacement->outputs.end() ||
           std::any_of(graph.edges.begin(), graph.edges.end(),
               [replacement, consumer](const Edge& edge)
               {
                   return edge.source == replacement && edge.destination == consumer;
               }))
        {
            return false;
        }
    }

    for(Node* consumer : consumers)
    {
        if(!graph.replaceInput(consumer, node, replacement))
        {
            throw std::logic_error("ConstantFold replacement failed after preflight");
        }
    }
    graph.removeNode(node);
    return true;
}

}

std::string ConstantFold::getName()
{
    return "Constant Folding";
}

void ConstantFold::run(Graph& graph)
{
    std::vector<Node*> nodes;
    for(auto& holder : graph.nodes) nodes.push_back(holder.get());

    for(auto node : nodes)
    {
        if(node->op == Operation::Transpose && node->inputs.size() == 1 &&
           node->inputs[0]->op == Operation::Transpose &&
           node->inputs[0]->inputs.size() == 1)
        {
            replaceWithInput(graph, node, node->inputs[0]->inputs[0]);
            continue;
        }

        if(node->op == Operation::Reshape && node->inputs.size() == 1 &&
           !node->shape.empty() && node->shape == node->inputs[0]->shape)
        {
            replaceWithInput(graph, node, node->inputs[0]);
            continue;
        }

        if(node->op == Operation::Transfer && node->inputs.size() == 1 &&
           node->inputs[0]->op == Operation::Transfer &&
           node->attributes["target"] == node->inputs[0]->attributes["target"])
        {
            replaceWithInput(graph, node, node->inputs[0]);
            continue;
        }

        if(node->inputs.size() != 2) continue;
        Node* left = node->inputs[0];
        Node* right = node->inputs[1];

        if(isConstant(left) && isConstant(right))
        {
            float value = 0.0f;
            bool supported = true;
            switch(node->op)
            {
                case Operation::Add: value = left->constantValue + right->constantValue; break;
                case Operation::Subtract: value = left->constantValue - right->constantValue; break;
                case Operation::Multiply: value = left->constantValue * right->constantValue; break;
                case Operation::Divide:
                    if(right->constantValue == 0.0f) supported = false;
                    else value = left->constantValue / right->constantValue;
                    break;
                default: supported = false; break;
            }
            if(supported)
            {
                graph.disconnect(left, node);
                graph.disconnect(right, node);
                node->op = Operation::Constant;
                node->isConstant = true;
                node->constantValue = value;
            }
            continue;
        }

        if(node->op == Operation::Add)
        {
            if(isConstant(right) && right->constantValue == 0.0f) replaceWithInput(graph, node, left);
            else if(isConstant(left) && left->constantValue == 0.0f) replaceWithInput(graph, node, right);
        }
        else if(node->op == Operation::Multiply)
        {
            if(isConstant(right) && right->constantValue == 1.0f) replaceWithInput(graph, node, left);
            else if(isConstant(left) && left->constantValue == 1.0f) replaceWithInput(graph, node, right);
        }
    }
}

}
