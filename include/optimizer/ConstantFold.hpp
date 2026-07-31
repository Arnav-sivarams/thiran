#pragma once

#include "optimizer/Pass.hpp"

namespace thiran
{

class ConstantFold : public Pass
{
public:
    std::string getName() override;
    void run(Graph& graph) override;
};

}
