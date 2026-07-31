#pragma once

namespace thiran
{

class CommunicationEdge
{
public:

    int fromPartition;

    int toPartition;

    int tensorBytes;

    CommunicationEdge(
        int from,
        int to,
        int bytes
    );

};

}