#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "region/RegionGraph.hpp"

namespace thiran::region
{

enum class RegionVerificationLevel : std::uint8_t
{
    Structural,
    Executable
};

struct RegionDiagnostic final
{
    std::string code;
    std::string message;
    std::optional<RegionId> regionId;
    std::optional<std::string> nodeName;
};

struct RegionVerificationResult final
{
    bool valid;
    std::vector<RegionDiagnostic> diagnostics;
};

class RegionVerifier final
{
public:

    static RegionVerificationResult verify(
        const Graph& graph,
        const RegionGraph& regionGraph,
        RegionVerificationLevel level =
            RegionVerificationLevel::Structural
    );
};

}
