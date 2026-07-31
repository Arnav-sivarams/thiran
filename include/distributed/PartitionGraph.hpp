#pragma once

#include <memory>
#include <vector>

#include "distributed/Partition.hpp"

namespace thiran
{

class PartitionGraph
{
public:

    std::vector<std::shared_ptr<Partition>> partitions;

    void addPartition(int id);

};

}