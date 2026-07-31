#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ir/Graph.hpp"
#include "region/Region.hpp"

namespace thiran::region
{

// Owns Regions while retaining a non-owning reference to the source Graph.
// The source Graph and its Nodes must outlive this object.
class RegionGraph final
{
public:

    explicit RegionGraph(const Graph& graph);

    ~RegionGraph() = default;

    RegionGraph(const RegionGraph&) = delete;
    RegionGraph& operator=(const RegionGraph&) = delete;

    RegionGraph(RegionGraph&&) noexcept = default;
    RegionGraph& operator=(RegionGraph&&) noexcept = default;

    const Graph& sourceGraph() const noexcept;

    Region& createRegion();

    bool appendNode(RegionId regionId, const Node& node);

    bool setStrategy(
        RegionId regionId,
        RegionStrategy strategy
    ) noexcept;

    bool finalize();

    bool finalized() const noexcept;

    std::size_t size() const noexcept;

    bool empty() const noexcept;

    const Region* findRegion(RegionId id) const noexcept;

    Region* findRegion(RegionId id) noexcept;

    const Region* regionFor(const Node& node) const noexcept;

    const std::vector<std::unique_ptr<Region>>& regions()
        const noexcept;

    const std::vector<RegionDependency>& dependencies()
        const noexcept;

private:

    const Graph* sourceGraph_;
    std::vector<std::unique_ptr<Region>> regions_;
    std::vector<RegionDependency> dependencies_;
    std::unordered_map<const Node*, RegionId> membership_;
    bool finalized_;
};

}
