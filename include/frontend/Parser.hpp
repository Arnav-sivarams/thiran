#pragma once

#include <optional>
#include <string>
#include <vector>

#include "frontend/Ast.hpp"
#include "frontend/SourceManager.hpp"

namespace thiran::frontend
{
class Parser final
{
public:
    static std::optional<ModuleAst> parse(const SourceManager& sources, SourceId source,
                                          std::vector<std::string>& diagnostics);
};
}
