#include "utils/CommunicationPrinter.hpp"

#include <iostream>

namespace thiran
{

void CommunicationPrinter::print(
    CommunicationGraph& graph
)
{
    std::cout << "\n";

    std::cout
        << "===== COMMUNICATION =====\n\n";

    for(auto& edge : graph.edges)
    {
        std::cout
            << "Partition "
            << edge->fromPartition
            << " -> Partition "
            << edge->toPartition
            << " ("
            << edge->tensorBytes
            << " bytes)\n";
    }

    if(graph.edges.empty())
    {
        std::cout
            << "No communication required.\n";
    }
}

}