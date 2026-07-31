#include "frontend/BuiltinOperations.hpp"

#include <map>

namespace thiran::frontend
{
Operation builtinOperation(const std::string& name)
{
    static const std::map<std::string, Operation> values{
        {"Input", Operation::Input}, {"Output", Operation::Output}, {"Constant", Operation::Constant},
        {"Add", Operation::Add}, {"Subtract", Operation::Subtract}, {"Multiply", Operation::Multiply},
        {"Divide", Operation::Divide}, {"MatMul", Operation::MatMul}, {"Conv2D", Operation::Conv2D},
        {"ReLU", Operation::ReLU}, {"Sigmoid", Operation::Sigmoid}, {"Tanh", Operation::Tanh},
        {"Softmax", Operation::Softmax}, {"MaxPool", Operation::MaxPool}, {"AvgPool", Operation::AvgPool},
        {"Reshape", Operation::Reshape}, {"Transpose", Operation::Transpose}, {"Transfer", Operation::Transfer}
    };
    const auto found = values.find(name);
    return found == values.end() ? Operation::Unknown : found->second;
}

bool isBuiltinOperation(const std::string& name)
{
    return builtinOperation(name) != Operation::Unknown;
}
}
