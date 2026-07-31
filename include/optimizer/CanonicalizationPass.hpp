#pragma once

#include "optimizer/Pass.hpp"

namespace thiran
{

class CanonicalizationPass : public Pass
{
public:

    std::string getName() override;

    void run(Graph& graph) override;

private:

    void sortInputs(Node* node);
};

}