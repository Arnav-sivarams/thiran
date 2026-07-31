#include "backend/SignatureBuilder.hpp"

namespace thiran
{

KernelSignature SignatureBuilder::build(
    Kernel& kernel
)
{
    KernelSignature signature;

    for(auto op : kernel.operations)
    {
        if(op->op == Operation::Input)
        {
            signature.parameters.push_back(
                op->name + "_ptr"
            );
        }
    }

    return signature;
}

}