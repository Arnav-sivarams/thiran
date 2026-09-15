#include "optimizer/GraphVerifier.hpp"

#include <algorithm>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace thiran
{

bool GraphVerifier::verify(Graph& graph)
{
    bool ok = true;

    ok &= verifyDuplicateNames(graph);

    ok &= verifyEdges(graph);

    ok &= verifyOperations(graph);

    ok &= verifyArity(graph);

    ok &= verifyAcyclic(graph);

    if(ok)
    {
        std::cout << "Graph Verification : PASSED\n";
    }
    else
    {
        std::cout << "Graph Verification : FAILED\n";
    }

    return ok;
}

bool GraphVerifier::verifyDuplicateNames(Graph& graph)
{
    std::unordered_set<std::string> names;

    for(auto& node : graph.nodes)
    {
        if(names.count(node->name))
        {
            std::cout
                << "Duplicate Node : "
                << node->name
                << "\n";

            return false;
        }

        names.insert(node->name);
    }

    return true;
}

bool GraphVerifier::verifyEdges(Graph& graph)
{
    std::unordered_set<Node*> nodes;
    for(auto& node : graph.nodes)
    {
        nodes.insert(node.get());
    }

    for(auto& edge : graph.edges)
    {
        if(edge.source == nullptr)
        {
            return false;
        }

        if(edge.destination == nullptr)
        {
            return false;
        }

        if(!nodes.count(edge.source) || !nodes.count(edge.destination))
        {
            std::cout << "Edge references a node outside the graph\n";
            return false;
        }

        if(std::find(edge.source->outputs.begin(), edge.source->outputs.end(), edge.destination)
            == edge.source->outputs.end() ||
           std::find(edge.destination->inputs.begin(), edge.destination->inputs.end(), edge.source)
            == edge.destination->inputs.end())
        {
            std::cout << "Edge adjacency is inconsistent\n";
            return false;
        }
    }

    for(auto& node : graph.nodes)
    {
        for(auto input : node->inputs)
        {
            if(!input || !nodes.count(input))
            {
                std::cout << "Invalid input edge on " << node->name << "\n";
                return false;
            }
            if(std::none_of(graph.edges.begin(), graph.edges.end(),
                [input, destination = node.get()](const Edge& edge)
                {
                    return edge.source == input && edge.destination == destination;
                }))
            {
                std::cout << "Input adjacency has no edge on " << node->name << "\n";
                return false;
            }
        }
        for(auto output : node->outputs)
        {
            if(!output || !nodes.count(output))
            {
                std::cout << "Invalid output edge on " << node->name << "\n";
                return false;
            }
            if(std::none_of(graph.edges.begin(), graph.edges.end(),
                [source = node.get(), output](const Edge& edge)
                {
                    return edge.source == source && edge.destination == output;
                }))
            {
                std::cout << "Output adjacency has no edge on " << node->name << "\n";
                return false;
            }
        }
    }

    return true;
}

bool GraphVerifier::verifyOperations(Graph& graph)
{
    for(auto& node : graph.nodes)
    {
        if(node->op == Operation::Unknown)
        {
            std::cout
                << "Unknown Operation : "
                << node->name
                << "\n";

            return false;
        }
    }

    return true;
}

bool GraphVerifier::verifyArity(Graph& graph)
{
    for(auto& holder : graph.nodes)
    {
        Node* node = holder.get();
        size_t expected = 0;
        bool exact = true;

        switch(node->op)
        {
            case Operation::Input:
            case Operation::Constant:
                expected = 0;
                break;
            case Operation::Output:
            case Operation::ReLU:
            case Operation::Sigmoid:
            case Operation::Tanh:
            case Operation::Softmax:
            case Operation::MaxPool:
            case Operation::AvgPool:
            case Operation::Reshape:
            case Operation::Transpose:
            case Operation::Transfer:
                expected = 1;
                break;
            case Operation::Add:
            case Operation::Subtract:
            case Operation::Multiply:
            case Operation::Divide:
            case Operation::MatMul:
            case Operation::Conv2D:
                expected = 2;
                break;
            case Operation::FusedMatMulRelu:
                expected = 2;
                break;
            default:
                exact = false;
                break;
        }

        if(exact && node->inputs.size() != expected)
        {
            std::cout << "Invalid input count for " << node->name
                      << ": expected " << expected
                      << ", got " << node->inputs.size() << "\n";
            return false;
        }
    }

    return true;
}

bool GraphVerifier::verifyAcyclic(Graph& graph)
{
    enum class State { Unvisited, Visiting, Done };
    std::unordered_map<Node*, State> state;

    std::function<bool(Node*)> visit = [&](Node* node)
    {
        if(state[node] == State::Visiting)
        {
            std::cout << "Cycle detected at " << node->name << "\n";
            return false;
        }
        if(state[node] == State::Done)
        {
            return true;
        }

        state[node] = State::Visiting;
        for(auto output : node->outputs)
        {
            if(!visit(output))
            {
                return false;
            }
        }
        state[node] = State::Done;
        return true;
    };

    for(auto& node : graph.nodes)
    {
        if(!visit(node.get()))
        {
            return false;
        }
    }

    return true;
}

}
