#include "ir/Graph.hpp"

#include <algorithm>

namespace thiran
{

Graph::Graph()
{
    nextId = 0;
}

Node* Graph::createNode(
    const std::string& name,
    Operation op
)
{
    auto node =
        std::make_shared<Node>(
            nextId,
            name,
            op
        );

    nextId++;

    nodes.push_back(node);

    symbolTable[name] = node.get();

    return node.get();
}

void Graph::connect(
    Node* from,
    Node* to
)
{
    if(from == nullptr || to == nullptr)
    {
        return;
    }

    for(const auto& edge : edges)
    {
        if(edge.source == from && edge.destination == to)
        {
            return;
        }
    }

    from->addOutput(to);

    to->addInput(from);

    edges.emplace_back(from, to);
}

void Graph::disconnect(Node* from, Node* to)
{
    if(from == nullptr || to == nullptr)
    {
        return;
    }

    auto eraseNode = [](std::vector<Node*>& values, Node* value)
    {
        values.erase(
            std::remove(values.begin(), values.end(), value),
            values.end()
        );
    };

    eraseNode(from->outputs, to);
    eraseNode(to->inputs, from);

    edges.erase(
        std::remove_if(
            edges.begin(),
            edges.end(),
            [from, to](const Edge& edge)
            {
                return edge.source == from && edge.destination == to;
            }
        ),
        edges.end()
    );
}

void Graph::removeNode(Node* node)
{
    if(node == nullptr)
    {
        return;
    }

    const auto inputs = node->inputs;
    const auto outputs = node->outputs;

    for(auto input : inputs)
    {
        disconnect(input, node);
    }

    for(auto output : outputs)
    {
        disconnect(node, output);
    }

    auto symbol = symbolTable.find(node->name);
    if(symbol != symbolTable.end() && symbol->second == node)
    {
        symbolTable.erase(symbol);
    }

    nodes.erase(
        std::remove_if(
            nodes.begin(),
            nodes.end(),
            [node](const std::shared_ptr<Node>& candidate)
            {
                return candidate.get() == node;
            }
        ),
        nodes.end()
    );
}

Node* Graph::findNode(
    const std::string& name
)
{
    auto it = symbolTable.find(name);

    if(it == symbolTable.end())
    {
        return nullptr;
    }

    return it->second;
}

}
