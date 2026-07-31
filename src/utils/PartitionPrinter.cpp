#include "utils/PartitionPrinter.hpp"

#include <iostream>

namespace thiran
{

void PartitionPrinter::print(
    PartitionGraph& graph)
{
    std::cout << "\n";

    std::cout
        << "===== PARTITIONS =====\n\n";

    for(auto& partition : graph.partitions)
    {
        std::cout
            << "Partition "
            << partition->id
            << "\n";

        for(auto node : partition->nodes)
        {
            std::cout
                << "  "
                << node->node->name
                << "\n";
        }

        std::cout << "\n";
    }
}

}