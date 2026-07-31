#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "frontend/SourceLocation.hpp"

namespace thiran::frontend
{
struct FunctionIdentity final
{
    SourceId module;
    std::size_t declaration;
    bool operator<(const FunctionIdentity& other) const noexcept
    { return module < other.module || (module == other.module && declaration < other.declaration); }
};

struct CallGraphEdge final
{
    std::size_t callerOrdinal;
    std::size_t calleeOrdinal;
    std::size_t statementOrdinal;
    SourceSpan callSite;
};

struct CallGraph final
{
    std::vector<FunctionIdentity> functions;
    std::vector<CallGraphEdge> edges;
};
}
