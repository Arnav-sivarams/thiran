#pragma once

#include <string>
#include <vector>

#include "frontend/SourceManager.hpp"
#include "frontend/Token.hpp"

namespace thiran::frontend
{
class Lexer final
{
public:
    static std::vector<Token> tokenize(const SourceManager& sources, SourceId source,
                                       std::vector<std::string>& diagnostics);
};
}
