#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "region/Region.hpp"
#include "region/RegionGraph.hpp"

namespace thiran::region
{

struct NodeStrategyAssignment final
{
    const Node* node;
    RegionStrategy strategy;
};

struct RegionFormationDiagnostic final
{
    std::string code;
    std::string message;
    std::optional<std::string> nodeName;
    std::optional<std::size_t> assignmentIndex;
};

struct RegionFormationResult final
{
    std::unique_ptr<RegionGraph> regionGraph;
    std::vector<RegionFormationDiagnostic> diagnostics;

    bool succeeded() const noexcept;
};

class TopologicalRegionFormer final
{
public:

    static RegionFormationResult form(
        const Graph& graph,
        const std::vector<NodeStrategyAssignment>& assignments
    );
};

}
