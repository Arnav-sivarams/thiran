#pragma once

#include <string>

#include "ir/Node.hpp"
#include "backend/SymbolTable.hpp"

namespace thiran
{

class TritonEmitter
{
public:

    static std::string emit(
        Node* node,
        SymbolTable& symbols
    );

};

}