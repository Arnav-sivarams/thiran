#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "frontend/LinkedProgram.hpp"
#include "frontend/LoweringProvenance.hpp"
#include "frontend/SourceManager.hpp"
#include "ir/Graph.hpp"

namespace thiran::frontend
{
struct FrontendResult final
{
    std::optional<Graph> graph;
    std::optional<LinkedProgram> program;
    std::vector<std::string> diagnostics;
    LoweringProvenanceTable provenance;
    bool succeeded() const noexcept { return graph.has_value() && diagnostics.empty(); }
};

class ModuleLinker final
{
public:
    static FrontendResult compile(const std::filesystem::path& entryPath);
};
}
