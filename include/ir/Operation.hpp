#pragma once

#include <string>

namespace thiran
{

enum class Operation
{
    Input,
    Output,

    Constant,

    Add,
    Subtract,
    Multiply,
    Divide,

    MatMul,

    Conv2D,

    ReLU,
    Sigmoid,
    Tanh,
    Softmax,

    MaxPool,
    AvgPool,

    Reshape,
    Transpose,

    Transfer,
    FusedMatMulRelu,

    Unknown
};

std::string toString(Operation op);

}