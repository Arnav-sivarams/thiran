#include "distributed/PartitionGraph.hpp"

namespace thiran
{

void PartitionGraph::addPartition(int id)
{
    partitions.push_back(
        std::make_shared<Partition>(id)
    );
}

}