#include "backend/PyTorchEmitter.hpp"

#include <sstream>

namespace thiran
{

std::string PyTorchEmitter::emit(
    Node* node,
    SymbolTable& symbols
)
{
    std::stringstream out;

    std::string self =
        symbols.lookup(node);

    switch(node->op)
    {
        case Operation::Input:

            break;

        case Operation::MatMul:

            out
                << self
                << " = torch.matmul(t0, t1)\n";

            break;

        case Operation::ReLU:

            out
                << self
                << " = torch.relu(t2)\n";

            break;

        case Operation::Add:

            out
                << self
                << " = t2 + t3\n";

            break;

        default:

            break;
    }

    return out.str();
}

}