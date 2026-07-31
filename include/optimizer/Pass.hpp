#pragma once

#include <string>

#include "ir/Graph.hpp"

namespace thiran
{

class Pass
{
public:

    virtual ~Pass() = default;

    virtual std::string getName() = 0;

    virtual void run(Graph& graph) = 0;
};

}