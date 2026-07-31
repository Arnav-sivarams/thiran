#include "ir/Node.hpp"

namespace thiran
{

Node::Node()
{
    id = -1;

    name = "";

    op = Operation::Unknown;

    device = Device::Unknown;

    dtype = "float32";
}

Node::Node(
    int id,
    const std::string& name,
    Operation op
)
{
    this->id = id;

    this->name = name;

    this->op = op;

    device = Device::Unknown;

    dtype = "float32";
}

void Node::addInput(Node* node)
{
    inputs.push_back(node);
}

void Node::addOutput(Node* node)
{
    outputs.push_back(node);
}

}