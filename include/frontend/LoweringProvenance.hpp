#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "frontend/SourceLocation.hpp"

namespace thiran::frontend
{
struct SourceProvenance final
{
    std::string path;
    SourceSpan span;
};

struct InlineFrame final
{
    std::string function;
    SourceProvenance callSite;
};

struct LoweringProvenance final
{
    std::size_t valueId = 0;
    std::string definingModule;
    SourceSpan functionDeclaration;
    SourceSpan localDeclaration;
    SourceProvenance immediateCallSite;
    std::vector<InlineFrame> inlineChain;
};

using LoweringProvenanceTable = std::map<std::string, LoweringProvenance>;
}
