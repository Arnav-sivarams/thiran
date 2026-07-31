#pragma once

#include <string>
#include <unordered_map>

namespace thiran
{

class Node;

class SymbolTable
{
public:

    void bind(
        Node* node,
        const std::string& symbol
    );

    std::string lookup(
        Node* node
    ) const;

private:

    std::unordered_map<
        Node*,
        std::string
    > table;

};

}