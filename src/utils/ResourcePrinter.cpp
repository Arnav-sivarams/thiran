#include "utils/ResourcePrinter.hpp"

#include <iostream>

namespace thiran
{

void ResourcePrinter::print(
    Graph& graph,
    ResourceAnalysis& resources
)
{
    std::cout << "\n";

    std::cout
        << "===== RESOURCE ANALYSIS =====\n\n";

    for(auto& node : graph.nodes)
    {
        auto r =
            resources.resource(node.get());

        std::cout
            << node->name
            << "\n";

        std::cout
            << "  FLOPs : "
            << r.flops
            << "\n";

        std::cout
            << "  Memory: "
            << r.memoryBytes / 1024.0
            << " KB\n";

        std::cout
            << "  Time  : "
            << r.estimatedTime
            << " s\n\n";
    }

    std::cout << "Summary\n";
    std::cout << "  Total FLOPs : " << resources.totalFlops() << "\n";
    std::cout << "  Peak Memory : " << resources.peakMemoryBytes() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  Est. Runtime: " << resources.estimatedRuntime() << " s\n";
}

}
