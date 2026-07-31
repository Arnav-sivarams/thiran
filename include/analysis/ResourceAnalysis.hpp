#pragma once

#include <unordered_map>

#include "analysis/ResourceInfo.hpp"
#include "analysis/ShapeInference.hpp"

namespace thiran
{

class ResourceAnalysis
{
public:

    void analyze(
        Graph& graph,
        ShapeInference& shapes
    );

    ResourceInfo resource(
        Node* node
    );

    double totalFlops() const;

    double totalMemoryBytes() const;

    double peakMemoryBytes() const;

    double estimatedRuntime() const;

private:

    std::unordered_map<Node*, ResourceInfo> data;

    double totalFlopsValue = 0.0;
    double totalMemoryValue = 0.0;
    double peakMemoryValue = 0.0;
    double estimatedRuntimeValue = 0.0;

};

}
