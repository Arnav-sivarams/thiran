#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "ir/Operation.hpp"
#include "scheduler/Device.hpp"
#include "ir/TensorType.hpp"

namespace thiran
{

class Node
{
public:

    int id;

    std::string name;

    Operation op;

    bool isConstant = false;

    float constantValue = 0.0f;

    Device device;

    std::vector<int> shape;

    std::string dtype;

    std::unordered_map<std::string, std::string> attributes;

    std::vector<Node*> inputs;

    std::vector<Node*> outputs;

    TensorType tensor;

    Node();

    Node(
        int id,
        const std::string& name,
        Operation op
    );

    void addInput(Node* node);

    void addOutput(Node* node);
};

}