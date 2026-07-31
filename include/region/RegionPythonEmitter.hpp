#pragma once

#include <optional>
#include <string>
#include <vector>

#include "region/RegionPlan.hpp"

namespace thiran::region
{

struct RegionExecutionDiagnostic final
{
    std::string code;
    std::string message;
    std::optional<RegionId> regionId;
    std::optional<std::string> nodeName;
};

struct RegionPythonArtifact final
{
    std::string source;
};

struct RegionPythonEmissionResult final
{
    std::optional<RegionPythonArtifact> artifact;
    std::vector<RegionExecutionDiagnostic> diagnostics;

    bool succeeded() const noexcept;
};

class RegionPythonEmitter final
{
public:
    static RegionPythonEmissionResult emit(
        const Graph& graph,
        const RegionPlan& plan
    );
};

}
