#pragma once

#include "optimizer/Pass.hpp"

namespace thiran
{

class DCEPass : public Pass
{
public:
    std::string getName() override;
    void run(Graph& graph) override;
};

}
