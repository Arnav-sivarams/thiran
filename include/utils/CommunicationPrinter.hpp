#pragma once

#include "distributed/CommunicationGraph.hpp"

namespace thiran
{

class CommunicationPrinter
{
public:

    static void print(
        CommunicationGraph& graph
    );

};

}