#pragma once

#include "backend/SymbolTable.hpp"
#include "backend/BackendIR.hpp"

namespace thiran
{

class SymbolBuilder
{
public:

    SymbolTable build(
        BackendIR& backend
    );

};

}