#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "frontend/Ast.hpp"

namespace thiran::frontend
{
struct ResolvedImport final
{
    std::string alias;
    SourceId target;
    SourceSpan span;
};

struct LinkedModule final
{
    std::size_t ordinal;
    ModuleAst ast;
    std::vector<ResolvedImport> imports;
};

struct LinkedProgram final
{
    SourceId entry;
    std::vector<LinkedModule> modules;
};
}
