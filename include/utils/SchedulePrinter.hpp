#pragma once

#include <vector>

#include "ir/Node.hpp"

namespace thiran
{

class SchedulePrinter
{
public:

    static void print(
        const std::vector<std::vector<Node*>>& levels
    );

};

}