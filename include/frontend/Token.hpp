#pragma once

#include <string>

#include "frontend/SourceLocation.hpp"

namespace thiran::frontend
{
enum class TokenKind
{
    Identifier, Number, String, Equals, LeftParen, RightParen, Comma, Minus,
    Dot, LeftBrace, RightBrace, Newline, EndOfFile, Unknown
};

struct Token final
{
    TokenKind kind;
    std::string text;
    SourceSpan span;
};
}
