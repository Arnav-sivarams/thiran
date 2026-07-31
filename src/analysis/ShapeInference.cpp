#include "analysis/ShapeInference.hpp"

#include <string>

namespace thiran
{

void ShapeInference::infer(Graph& graph)
{
    shapes.clear();
    errors.clear();

    for(auto& holder : graph.nodes)
    {
        Node* node = holder.get();

        auto inputShape = [&](size_t index, TensorShape& result) -> bool
        {
            if(node->inputs.size() <= index || node->inputs[index] == nullptr)
            {
                report("Missing input " + std::to_string(index) + " for " + node->name);
                return false;
            }

            auto found = shapes.find(node->inputs[index]);
            if(found == shapes.end())
            {
                report("Shape for an input of " + node->name + " is unavailable");
                return false;
            }

            result = found->second;
            return true;
        };

        auto store = [&](const TensorShape& result)
        {
            shapes[node] = result;
            node->shape = result.dimensions;
            node->tensor.shape = result.dimensions;
        };

        switch(node->op)
        {
            case Operation::Input:
            {
                TensorShape result;
                result.dimensions = node->shape.empty() ? std::vector<int>{1, 1} : node->shape;
                store(result);
                break;
            }

            case Operation::Constant:
            {
                TensorShape result;
                result.dimensions = node->shape;
                store(result);
                break;
            }

            case Operation::MatMul:
            case Operation::FusedMatMulRelu:
            {
                TensorShape left, right;
                if(!inputShape(0, left) || !inputShape(1, right)) break;
                if(left.rank() != 2 || right.rank() != 2 || left.dimensions[1] != right.dimensions[0])
                {
                    report("Invalid MatMul dimensions for " + node->name);
                    break;
                }
                store(TensorShape({left.dimensions[0], right.dimensions[1]}));
                break;
            }

            case Operation::Add:
            case Operation::Subtract:
            case Operation::Multiply:
            case Operation::Divide:
            {
                TensorShape left, right;
                if(!inputShape(0, left) || !inputShape(1, right)) break;
                if(left.dimensions.empty()) { store(right); break; }
                if(right.dimensions.empty()) { store(left); break; }
                if(left.dimensions != right.dimensions)
                {
                    report("Elementwise dimension mismatch for " + node->name);
                    break;
                }
                store(left);
                break;
            }

            case Operation::ReLU:
            case Operation::Sigmoid:
            case Operation::Tanh:
            case Operation::Softmax:
            case Operation::Output:
            case Operation::Transfer:
            {
                TensorShape input;
                if(inputShape(0, input)) store(input);
                break;
            }

            case Operation::Conv2D:
            {
                TensorShape input, filter;
                if(!inputShape(0, input) || !inputShape(1, filter)) break;
                if(input.rank() != 4 || filter.rank() != 4 || input.dimensions[1] != filter.dimensions[1] ||
                   input.dimensions[2] < filter.dimensions[2] || input.dimensions[3] < filter.dimensions[3])
                {
                    report("Invalid Conv2D dimensions for " + node->name);
                    break;
                }
                store(TensorShape({input.dimensions[0], filter.dimensions[0],
                                   input.dimensions[2] - filter.dimensions[2] + 1,
                                   input.dimensions[3] - filter.dimensions[3] + 1}));
                break;
            }

            case Operation::MaxPool:
            case Operation::AvgPool:
            {
                TensorShape input;
                if(!inputShape(0, input)) break;
                if(input.rank() == 4)
                {
                    if(input.dimensions[2] < 2 || input.dimensions[3] < 2)
                    {
                        report("Pool input is too small for " + node->name);
                        break;
                    }
                    store(TensorShape({input.dimensions[0], input.dimensions[1],
                                       input.dimensions[2] / 2, input.dimensions[3] / 2}));
                }
                else store(input);
                break;
            }

            case Operation::Reshape:
            {
                TensorShape input;
                if(!inputShape(0, input)) break;
                const auto target = node->shape;
                if(target.empty()) { store(input); break; }
                long long inputElements = 1;
                long long outputElements = 1;
                for(int dimension : input.dimensions) inputElements *= dimension;
                for(int dimension : target)
                {
                    if(dimension <= 0)
                    {
                        report("Invalid Reshape dimension for " + node->name);
                        outputElements = -1;
                        break;
                    }
                    outputElements *= dimension;
                }
                if(outputElements == inputElements)
                {
                    TensorShape result;
                    result.dimensions = target;
                    store(result);
                }
                else if(outputElements >= 0)
                    report("Reshape element count mismatch for " + node->name);
                break;
            }

            case Operation::Transpose:
            {
                TensorShape input;
                if(!inputShape(0, input)) break;
                if(input.rank() != 2)
                {
                    report("Transpose currently requires a rank-2 tensor for " + node->name);
                    break;
                }
                store(TensorShape({input.dimensions[1], input.dimensions[0]}));
                break;
            }

            default:
                report("Cannot infer shape for " + node->name);
                break;
        }
    }
}

TensorShape ShapeInference::shape(Node* node)
{
    auto found = shapes.find(node);
    return found == shapes.end() ? TensorShape{} : found->second;
}

bool ShapeInference::hasErrors() const
{
    return !errors.empty();
}

const std::vector<std::string>& ShapeInference::diagnostics() const
{
    return errors;
}

void ShapeInference::report(const std::string& message)
{
    errors.push_back(message);
}

}
