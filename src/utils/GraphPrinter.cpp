#include "utils/GraphPrinter.hpp"

#include <iostream>

namespace thiran
{

void GraphPrinter::print(const Graph& graph)
{
    std::cout << "\n";
    std::cout << "=========== GRAPH ===========\n\n";

    for(const auto& node : graph.nodes)
    {
        std::cout
            << node->name
            << " ("
            << toString(node->op)
            << ")"
            << "\n";

        if(node->inputs.empty())
        {
            std::cout << "    Inputs : None\n";
        }
        else
        {
            std::cout << "    Inputs : ";

            for(auto input : node->inputs)
            {
                std::cout << input->name << " ";
            }

            std::cout << "\n";
        }

        if(node->outputs.empty())
        {
            std::cout << "    Outputs: None\n";
        }
        else
        {
            std::cout << "    Outputs: ";

            for(auto output : node->outputs)
            {
                std::cout << output->name << " ";
            }

            std::cout << "\n";
        }

        std::cout << "\n";
    }

    std::cout << "=============================\n";
}

}