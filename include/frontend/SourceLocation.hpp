#pragma once

#include <cstddef>
#include <limits>

namespace thiran::frontend
{
using SourceId = std::size_t;
inline constexpr SourceId invalidSourceId = std::numeric_limits<SourceId>::max();

struct SourceLocation final
{
    SourceId source = invalidSourceId;
    std::size_t offset = 0;
    std::size_t line = 1;
    std::size_t column = 1;
};

struct SourceSpan final
{
    SourceLocation begin;
    SourceLocation end;
};
}
