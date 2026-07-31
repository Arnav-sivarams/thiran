#include "distributed/CommunicationPlanner.hpp"

#include <unordered_map>
#include <unordered_set>

namespace thiran
{

CommunicationGraph CommunicationPlanner::build(
    PartitionGraph& graph
)
{
    CommunicationGraph communication;

    std::unordered_map<
        ExecutionNode*,
        int
    > location;

    std::unordered_map<Node*, int> tensorBytes;
    std::unordered_set<std::string> emitted;

    for(auto& partition : graph.partitions)
    {
        for(auto node : partition->nodes)
        {
            location[node] = partition->id;

            int elements = 1;
            for(int dimension : node->node->shape)
            {
                if(dimension > 0) elements *= dimension;
            }
            tensorBytes[node->node] = elements * 4;
        }
    }

    for(auto& partition : graph.partitions)
    {
        for(auto node : partition->nodes)
        {
            for(auto input : node->node->inputs)
            {
                ExecutionNode* producer = nullptr;

                for(auto& other : graph.partitions)
                {
                    for(auto n : other->nodes)
                    {
                        if(n->node == input)
                        {
                            producer = n;
                        }
                    }
                }

                if(producer == nullptr)
                {
                    continue;
                }

                if(location[producer] != location[node])
                {
                    const std::string key = std::to_string(producer->node->id) + ":" +
                                            std::to_string(node->node->id);
                    if(emitted.insert(key).second)
                    {
                        communication.add(
                            location[producer],
                            location[node],
                            tensorBytes[producer->node]
                        );
                    }
                }
            }
        }
    }

    return communication;
}

}
