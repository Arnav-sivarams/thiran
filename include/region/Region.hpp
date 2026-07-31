#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "ir/Node.hpp"

namespace thiran::region
{

using RegionId = std::uint32_t;

enum class RegionStrategy : std::uint8_t
{
    Unresolved,
    AheadOfTime,
    JustInTime,
    Fallback
};

std::string_view toString(RegionStrategy strategy) noexcept;

// Holds non-owning, read-only references to Nodes owned by a source Graph.
class Region final
{
public:

    RegionId id() const noexcept;

    RegionStrategy strategy() const noexcept;

    const std::vector<const Node*>& nodes() const noexcept;

    const std::vector<const Node*>& inputs() const noexcept;

    const std::vector<const Node*>& outputs() const noexcept;

    bool contains(const Node& node) const noexcept;

    std::size_t size() const noexcept;

    bool empty() const noexcept;

private:

    friend class RegionGraph;

    explicit Region(RegionId id);

    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;

    Region(Region&&) = delete;
    Region& operator=(Region&&) = delete;

    RegionId id_;
    RegionStrategy strategy_;
    std::vector<const Node*> nodes_;
    std::vector<const Node*> inputs_;
    std::vector<const Node*> outputs_;
};

struct RegionDependency final
{
    RegionId source;
    RegionId destination;
    std::vector<const Node*> values;
};

}
