#pragma once

#include <memory>
#include <vector>

#include "distributed/CommunicationEdge.hpp"

namespace thiran
{

class CommunicationGraph
{
public:

    std::vector<
        std::shared_ptr<CommunicationEdge>
    > edges;

    void add(
        int from,
        int to,
        int bytes
    );

};

}