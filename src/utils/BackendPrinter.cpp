#include "utils/BackendPrinter.hpp"

#include <iostream>

namespace thiran
{

void BackendPrinter::print(
    BackendIR& backend
)
{
    std::cout << "\n";

    std::cout
        << "===== BACKEND IR =====\n\n";

    for(auto& kernel : backend.kernels)
    {
        std::cout
            << kernel->name
            << "\n";

        std::cout
            << "Device : "
            << toString(kernel->device)
            << "\n";

        std::cout
            << "Operations\n";

        for(auto op : kernel->operations)
        {
            std::cout
                << "  "
                << op->name
                << " : "
                << toString(op->op)
                << "\n";
        }

        std::cout << "\n";
    }
}

}