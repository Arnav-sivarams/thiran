#include "region/RegionGraph.hpp"

#include <cassert>
#include <map>
#include <unordered_set>
#include <utility>

namespace thiran::region
{

RegionGraph::RegionGraph(const Graph& graph)
    : sourceGraph_(&graph),
      finalized_(false)
{
}

const Graph& RegionGraph::sourceGraph() const noexcept
{
    return *sourceGraph_;
}

Region& RegionGraph::createRegion()
{
    assert(!finalized_);

    const auto id = static_cast<RegionId>(regions_.size());
    regions_.push_back(std::unique_ptr<Region>(new Region(id)));
    return *regions_.back();
}

bool RegionGraph::appendNode(RegionId regionId, const Node& node)
{
    if(finalized_)
    {
        return false;
    }

    Region* region = findRegion(regionId);
    if(region == nullptr)
    {
        return false;
    }

    bool owned = false;
    for(const auto& holder : sourceGraph_->nodes)
    {
        if(holder.get() == &node)
        {
            owned = true;
            break;
        }
    }

    if(!owned || membership_.contains(&node) || region->contains(node))
    {
        return false;
    }

    region->nodes_.push_back(&node);
    membership_[&node] = regionId;
    return true;
}

bool RegionGraph::setStrategy(
    RegionId regionId,
    RegionStrategy strategy
) noexcept
{
    Region* region = findRegion(regionId);
    if(region == nullptr)
    {
        return false;
    }

    region->strategy_ = strategy;
    return true;
}

bool RegionGraph::finalize()
{
    if(finalized_)
    {
        return false;
    }

    for(auto& holder : regions_)
    {
        Region& region = *holder;
        region.inputs_.clear();
        std::unordered_set<const Node*> seen;

        for(const Node* node : region.nodes_)
        {
            if(node == nullptr)
            {
                continue;
            }

            for(const Node* producer : node->inputs)
            {
                if(producer == nullptr)
                {
                    continue;
                }

                const auto found = membership_.find(producer);
                if(found != membership_.end() && found->second == region.id_)
                {
                    continue;
                }

                if(seen.insert(producer).second)
                {
                    region.inputs_.push_back(producer);
                }
            }
        }
    }

    for(auto& holder : regions_)
    {
        Region& region = *holder;
        region.outputs_.clear();

        for(const Node* node : region.nodes_)
        {
            if(node == nullptr)
            {
                continue;
            }

            bool externallyVisible = node->op == Operation::Output;
            for(const Node* consumer : node->outputs)
            {
                if(consumer == nullptr)
                {
                    continue;
                }

                const auto found = membership_.find(consumer);
                if(found != membership_.end() && found->second != region.id_)
                {
                    externallyVisible = true;
                    break;
                }
            }

            if(externallyVisible)
            {
                region.outputs_.push_back(node);
            }
        }
    }

    dependencies_.clear();
    using DependencyKey = std::pair<RegionId, RegionId>;
    std::map<DependencyKey, std::unordered_set<const Node*>> buckets;

    for(const auto& destinationHolder : regions_)
    {
        const Region& destination = *destinationHolder;
        for(const Node* consumer : destination.nodes_)
        {
            if(consumer == nullptr)
            {
                continue;
            }

            for(const Node* producer : consumer->inputs)
            {
                if(producer == nullptr)
                {
                    continue;
                }

                const auto found = membership_.find(producer);
                if(found == membership_.end() ||
                   found->second == destination.id_)
                {
                    continue;
                }

                buckets[{found->second, destination.id_}].insert(producer);
            }
        }
    }

    for(const auto& [key, values] : buckets)
    {
        RegionDependency dependency{
            .source = key.first,
            .destination = key.second,
            .values = {}
        };

        for(const auto& node : sourceGraph_->nodes)
        {
            if(values.contains(node.get()))
            {
                dependency.values.push_back(node.get());
            }
        }

        dependencies_.push_back(std::move(dependency));
    }

    finalized_ = true;
    return true;
}

bool RegionGraph::finalized() const noexcept
{
    return finalized_;
}

std::size_t RegionGraph::size() const noexcept
{
    return regions_.size();
}

bool RegionGraph::empty() const noexcept
{
    return regions_.empty();
}

const Region* RegionGraph::findRegion(RegionId id) const noexcept
{
    if(id >= regions_.size())
    {
        return nullptr;
    }

    return regions_[id].get();
}

Region* RegionGraph::findRegion(RegionId id) noexcept
{
    if(id >= regions_.size())
    {
        return nullptr;
    }

    return regions_[id].get();
}

const Region* RegionGraph::regionFor(const Node& node) const noexcept
{
    const auto found = membership_.find(&node);
    return found == membership_.end() ? nullptr : findRegion(found->second);
}

const std::vector<std::unique_ptr<Region>>& RegionGraph::regions()
    const noexcept
{
    return regions_;
}

const std::vector<RegionDependency>& RegionGraph::dependencies()
    const noexcept
{
    return dependencies_;
}

}
