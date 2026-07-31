#pragma once

#include <string>
#include <vector>

#include "ir/Node.hpp"
#include "scheduler/Device.hpp"

namespace thiran
{

class Kernel
{
public:

    std::string name;

    Device device;

    std::vector<Node*> operations;

    std::vector<Node*> inputs;

    std::vector<Node*> outputs;
};

}