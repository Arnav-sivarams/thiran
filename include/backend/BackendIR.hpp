#pragma once

#include <memory>
#include <vector>

#include "backend/Kernel.hpp"

namespace thiran
{

class BackendIR
{
public:

    std::vector<
        std::shared_ptr<Kernel>
    > kernels;

};

}