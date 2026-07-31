#pragma once

#include "backend/BackendIR.hpp"

namespace thiran
{

class BackendPrinter
{
public:

    static void print(
        BackendIR& backend
    );

};

}