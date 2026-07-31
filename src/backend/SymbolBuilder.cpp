#include "backend/SymbolBuilder.hpp"

namespace thiran
{

SymbolTable SymbolBuilder::build(
    BackendIR& backend
)
{
    SymbolTable symbols;

    int id = 0;

    for(auto& kernel : backend.kernels)
    {
        for(auto op : kernel->operations)
        {
            symbols.bind(
                op,
                "t" + std::to_string(id++)
            );
        }
    }

    return symbols;
}

}