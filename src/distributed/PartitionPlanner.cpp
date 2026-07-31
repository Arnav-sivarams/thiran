#include "distributed/PartitionPlanner.hpp"

namespace thiran
{

PartitionGraph PartitionPlanner::partition(
    ExecutionGraph& graph
)
{
    PartitionGraph result;

    result.addPartition(0);
    result.addPartition(1);

    for(auto& node : graph.nodes)
    {
        if(node->device == Device::CPU)
        {
            result.partitions[0]->nodes.push_back(
                node.get()
            );
        }
        else
        {
            result.partitions[1]->nodes.push_back(
                node.get()
            );
        }
    }

    return result;
}

}