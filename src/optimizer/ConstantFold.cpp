#include "optimizer/ConstantFold.hpp"

#include <cmath>

namespace thiran
{

namespace
{

bool isConstant(Node* node)
{
    return node && node->op == Operation::Constant && node->shape.empty();
}

void replaceWithInput(Graph& graph, Node* node, Node* replacement)
{
    const auto consumers = node->outputs;
    for(auto consumer : consumers)
    {
        graph.disconnect(node, consumer);
        graph.connect(replacement, consumer);
    }
    graph.removeNode(node);
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
