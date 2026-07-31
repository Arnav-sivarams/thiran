#include "distributed/CommunicationGraph.hpp"

namespace thiran
{

void CommunicationGraph::add(
    int from,
    int to,
    int bytes
)
{
    edges.push_back(
        std::make_shared<CommunicationEdge>(
            from,
            to,
            bytes
        )
    );
}

}