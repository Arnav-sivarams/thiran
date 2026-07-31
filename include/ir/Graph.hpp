#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "ir/Edge.hpp"
#include "ir/Node.hpp"

namespace thiran
{

class Graph
{
private:

    int nextId;

public:

    std::vector<std::shared_ptr<Node>> nodes;

    std::vector<Edge> edges;

    std::unordered_map<std::string, Node*> symbolTable;

    Graph();

    Node* createNode(
        const std::string& name,
        Operation op
    );

    void connect(
        Node* from,
        Node* to
    );

    void disconnect(Node* from, Node* to);

    void removeNode(Node* node);

    Node* findNode(
        const std::string& name
    );
};

}
