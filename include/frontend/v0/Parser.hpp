#pragma once

#include <optional>
#include <string_view>
#include "frontend/v0/Ast.hpp"

namespace thiran::v0 {
struct ParseResult {
    std::optional<Module> module;
    std::vector<Diagnostic> diagnostics;
};
ParseResult parse(std::string_view source, std::string sourceName = "<memory>");
}
