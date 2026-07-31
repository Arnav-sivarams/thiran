#include "region/Region.hpp"

#include <algorithm>

namespace thiran::region
{

std::string_view toString(RegionStrategy strategy) noexcept
{
    switch(strategy)
    {
        case RegionStrategy::Unresolved:
            return "UNRESOLVED";
        case RegionStrategy::AheadOfTime:
            return "AOT";
        case RegionStrategy::JustInTime:
            return "JIT";
        case RegionStrategy::Fallback:
            return "FALLBACK";
    }

    return "UNKNOWN";
}

Region::Region(RegionId id)
    : id_(id),
      strategy_(RegionStrategy::Unresolved)
{
}

RegionId Region::id() const noexcept
{
    return id_;
}

RegionStrategy Region::strategy() const noexcept
{
    return strategy_;
}

const std::vector<const Node*>& Region::nodes() const noexcept
{
    return nodes_;
}

const std::vector<const Node*>& Region::inputs() const noexcept
{
    return inputs_;
}

const std::vector<const Node*>& Region::outputs() const noexcept
{
    return outputs_;
}

bool Region::contains(const Node& node) const noexcept
{
    return std::find(nodes_.begin(), nodes_.end(), &node) != nodes_.end();
}

std::size_t Region::size() const noexcept
{
    return nodes_.size();
}

bool Region::empty() const noexcept
{
    return nodes_.empty();
}

}
