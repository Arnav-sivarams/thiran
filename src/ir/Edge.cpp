#include "ir/Edge.hpp"

namespace thiran
{

Edge::Edge()
{
    source = nullptr;

    destination = nullptr;
}

Edge::Edge(
    Node* source,
    Node* destination
)
{
    this->source = source;

    this->destination = destination;
}

}