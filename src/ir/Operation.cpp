#include "ir/Operation.hpp"

namespace thiran
{

std::string toString(Operation op)
{
    switch(op)
    {
        case Operation::Input:
            return "Input";

        case Operation::Output:
            return "Output";

        case Operation::Constant:
            return "Constant";

        case Operation::Add:
            return "Add";

        case Operation::Subtract:
            return "Subtract";

        case Operation::Multiply:
            return "Multiply";

        case Operation::Divide:
            return "Divide";

        case Operation::MatMul:
            return "MatMul";

        case Operation::Conv2D:
            return "Conv2D";

        case Operation::ReLU:
            return "ReLU";

        case Operation::Sigmoid:
            return "Sigmoid";

        case Operation::Tanh:
            return "Tanh";

        case Operation::Softmax:
            return "Softmax";

        case Operation::MaxPool:
            return "MaxPool";

        case Operation::AvgPool:
            return "AvgPool";

        case Operation::Reshape:
            return "Reshape";

        case Operation::Transpose:
            return "Transpose";

        case Operation::Transfer:
            return "Transfer";

        case Operation::FusedMatMulRelu:
            return "FusedMatMulRelu";

        default:
            return "Unknown";
    }
}

}