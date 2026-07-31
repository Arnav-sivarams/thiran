#pragma once

#include "ir/Graph.hpp"

namespace thiran
{

class GraphVerifier
{
public:

    bool verify(Graph& graph);

private:

    bool verifyDuplicateNames(Graph& graph);

    bool verifyEdges(Graph& graph);

    bool verifyOperations(Graph& graph);

    bool verifyArity(Graph& graph);

    bool verifyAcyclic(Graph& graph);

};

}
