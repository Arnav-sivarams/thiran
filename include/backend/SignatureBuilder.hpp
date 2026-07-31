#pragma once

#include "backend/Kernel.hpp"
#include "backend/KernelSignature.hpp"

namespace thiran
{

class SignatureBuilder
{
public:

    KernelSignature build(
        Kernel& kernel
    );

};

}