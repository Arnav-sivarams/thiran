#pragma once

#include <string_view>
#include <vector>
#include "frontend/v0/Token.hpp"

namespace thiran::v0 {
struct LexResult {
    std::vector<Token> tokens;
    std::vector<Diagnostic> diagnostics;
};

LexResult lex(std::string_view source, std::string sourceName = "<memory>");
}
