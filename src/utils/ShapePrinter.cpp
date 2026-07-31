#include "utils/ShapePrinter.hpp"

#include <iostream>

namespace thiran
{

void ShapePrinter::print(
    Graph& graph,
    ShapeInference& inference
)
{
    std::cout << "\n";

    std::cout
        << "===== SHAPES =====\n\n";

    for(auto& node : graph.nodes)
    {
        auto shape =
            inference.shape(node.get());

        std::cout
            << node->name
            << " : [";

        for(size_t i = 0; i < shape.dimensions.size(); i++)
        {
            std::cout
                << shape.dimensions[i];

            if(i + 1 != shape.dimensions.size())
            {
                std::cout << ", ";
            }
        }

        std::cout << "]\n";
    }
}

}