#pragma once

namespace thiran
{

class Node;

class Edge
{
public:

    Node* source;

    Node* destination;

    Edge();

    Edge(
        Node* source,
        Node* destination
    );
};

}