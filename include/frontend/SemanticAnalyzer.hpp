#pragma once

#include <cstddef>

namespace thiran::frontend
{
struct FunctionLimits final
{
    static constexpr std::size_t functionsPerModule = 4096;
    static constexpr std::size_t parametersPerFunction = 256;
    static constexpr std::size_t statementsPerFunction = 4096;
    static constexpr std::size_t expansionDepth = 256;
    static constexpr std::size_t expandedNodes = 1000000;
    static constexpr std::size_t diagnostics = 100;
    static constexpr std::size_t generatedNameLength = 1024;
};
}
