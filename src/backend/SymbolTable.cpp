#include "backend/SymbolTable.hpp"

#include "ir/Node.hpp"

namespace thiran
{

void SymbolTable::bind(
    Node* node,
    const std::string& symbol
)
{
    table[node] = symbol;
}

std::string SymbolTable::lookup(
    Node* node
) const
{
    auto it = table.find(node);

    if(it == table.end())
        return "";

    return it->second;
}

}