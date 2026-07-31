#include "utils/ExecutionPrinter.hpp"

#include <iostream>
#include "scheduler/Device.hpp"

namespace thiran
{

void ExecutionPrinter::print(
    ExecutionGraph& graph
)
{
    std::cout << "\n";

    std::cout
        << "===== EXECUTION GRAPH =====\n\n";

    for(auto& node : graph.nodes)
    {
        std::cout
            << node->node->name
            << "  Stage "
            << node->stage
            << "  Device "
            << toString(node->device);

        if(node->parallel)
        {
            std::cout << "  Parallel";
        }

        std::cout << "\n";
    }
}

}