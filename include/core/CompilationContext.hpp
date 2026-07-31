#pragma once

#include "analysis/ShapeInference.hpp"
#include "analysis/ResourceAnalysis.hpp"

#include "distributed/CommunicationGraph.hpp"
#include "distributed/PartitionGraph.hpp"

#include "ir/Graph.hpp"

#include "scheduler/ExecutionGraph.hpp"

namespace thiran
{

class CompilationContext
{
public:

    Graph graph;

    ShapeInference shapes;

    ResourceAnalysis resources;

    ExecutionGraph execution;

    PartitionGraph partitions;

    CommunicationGraph communication;

};

}