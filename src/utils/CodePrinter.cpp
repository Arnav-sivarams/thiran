#include "utils/CodePrinter.hpp"

#include <iostream>

namespace thiran
{

void CodePrinter::print(
    const std::string& code
)
{
    std::cout
        << "\n===== GENERATED TRITON =====\n\n";

    std::cout
        << code
        << "\n";
}

}