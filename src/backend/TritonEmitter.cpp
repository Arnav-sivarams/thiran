#include "backend/TritonEmitter.hpp"

namespace thiran
{

std::string TritonEmitter::emit(Node* node, SymbolTable&)
{
    return "# Lowered operation: " + node->name + " (" + toString(node->op) + ")\n";
}

}
