#include "utils/FusionPrinter.hpp"

#include <iostream>

namespace thiran
{

void FusionPrinter::print(Graph& graph)
{
    std::cout << "\n";

    std::cout
        << "===== FUSED GRAPH =====\n\n";

    for(auto& node : graph.nodes)
    {
        std::cout
            << node->name
            << " : "
            << toString(node->op)
            << "\n";
    }
}

}