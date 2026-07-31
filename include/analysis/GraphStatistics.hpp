#pragma once

#include "ir/Graph.hpp"

namespace thiran
{

class GraphStatistics
{
public:

    int nodeCount(Graph& graph);

    int edgeCount(Graph& graph);

    int inputCount(Graph& graph);

    int outputCount(Graph& graph);

    int maxDepth(Graph& graph);
};

}