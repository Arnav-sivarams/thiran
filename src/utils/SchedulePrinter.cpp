#include "utils/SchedulePrinter.hpp"

#include <iostream>

namespace thiran
{

void SchedulePrinter::print(
    const std::vector<std::vector<Node*>>& levels
)
{
    std::cout << "\n";

    std::cout << "===== EXECUTION SCHEDULE =====\n\n";

    for(size_t i = 0; i < levels.size(); i++)
    {
        std::cout
            << "Stage "
            << i
            << "\n";

        for(auto node : levels[i])
        {
            std::cout
                << "  "
                << node->name
                << "\n";
        }

        std::cout << "\n";
    }
}

}