#include "distributed/CommunicationEdge.hpp"

namespace thiran
{

CommunicationEdge::CommunicationEdge(
    int from,
    int to,
    int bytes
)
{
    fromPartition = from;

    toPartition = to;

    tensorBytes = bytes;
}

}