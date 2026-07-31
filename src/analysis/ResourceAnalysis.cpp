#include "analysis/ResourceAnalysis.hpp"

#include <algorithm>

namespace thiran
{

void ResourceAnalysis::analyze(Graph& graph, ShapeInference& shapes)
{
    data.clear();
    totalFlopsValue = 0.0;
    totalMemoryValue = 0.0;
    peakMemoryValue = 0.0;
    estimatedRuntimeValue = 0.0;

    double liveMemory = 0.0;

    for(auto& holder : graph.nodes)
    {
        Node* node = holder.get();
        ResourceInfo info;
        const auto shape = shapes.shape(node);

        long long elements = 1;
        for(int dimension : shape.dimensions)
        {
            if(dimension > 0) elements *= dimension;
        }
        info.memoryBytes = static_cast<double>(elements) * 4.0;

        switch(node->op)
        {
            case Operation::MatMul:
            case Operation::FusedMatMulRelu:
                if(node->inputs.size() == 2)
                {
                    const auto left = shapes.shape(node->inputs[0]);
                    const auto right = shapes.shape(node->inputs[1]);
                    if(left.rank() == 2 && right.rank() == 2)
                        info.flops = 2.0 * left.dimensions[0] * left.dimensions[1] * right.dimensions[1];
                }
                if(node->op == Operation::FusedMatMulRelu) info.flops += elements;
                break;
            case Operation::Conv2D:
                if(node->inputs.size() == 2)
                {
                    const auto filter = shapes.shape(node->inputs[1]);
                    if(filter.rank() == 4)
                        info.flops = 2.0 * elements * filter.dimensions[1] * filter.dimensions[2] * filter.dimensions[3];
                }
                break;
            case Operation::Add:
            case Operation::Subtract:
            case Operation::Multiply:
            case Operation::Divide:
            case Operation::ReLU:
            case Operation::Tanh:
                info.flops = elements;
                break;
            case Operation::Sigmoid:
                info.flops = 4.0 * elements;
                break;
            case Operation::Softmax:
                info.flops = 5.0 * elements;
                break;
            case Operation::MaxPool:
            case Operation::AvgPool:
                info.flops = 4.0 * elements;
                break;
            case Operation::Transpose:
            case Operation::Reshape:
            case Operation::Transfer:
                info.flops = 0.0;
                break;
            default:
                break;
        }

        // A conservative demo estimate: 1 TFLOP/s compute plus 500 GB/s memory traffic.
        info.estimatedTime = info.flops / 1.0e12 + info.memoryBytes / 5.0e11;
        data[node] = info;
        totalFlopsValue += info.flops;
        totalMemoryValue += info.memoryBytes;
        estimatedRuntimeValue += info.estimatedTime;
        liveMemory += info.memoryBytes;
        peakMemoryValue = std::max(peakMemoryValue, liveMemory);
    }
}

ResourceInfo ResourceAnalysis::resource(Node* node)
{
    auto found = data.find(node);
    return found == data.end() ? ResourceInfo{} : found->second;
}

double ResourceAnalysis::totalFlops() const { return totalFlopsValue; }
double ResourceAnalysis::totalMemoryBytes() const { return totalMemoryValue; }
double ResourceAnalysis::peakMemoryBytes() const { return peakMemoryValue; }
double ResourceAnalysis::estimatedRuntime() const { return estimatedRuntimeValue; }

}
