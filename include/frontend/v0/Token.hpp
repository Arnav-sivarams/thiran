#pragma once

#include <string>
#include "frontend/SourceLocation.hpp"

namespace thiran::v0 {
using frontend::SourceLocation;
using frontend::SourceSpan;

enum class TokenKind {
    Identifier, Integer, Real, String, Newline, End,
    Let, Mut, Fn, Export, Return, Import, As, True, False,
    If, Else, For, In, While, Break, Continue, Struct, Borrow, Copy, Move,
    Plus, Minus, Star, DotStar, Slash, DotSlash, Equal, Colon,
    Comma, Semicolon, Dot, LeftParen, RightParen, LeftBracket,
    RightBracket, LeftBrace, RightBrace, Less, Greater, Arrow
};

struct Token {
    TokenKind kind;
    std::string text;
    SourceSpan span;
};

struct Diagnostic {
    std::string source;
    SourceSpan span;
    std::string message;
    std::string format() const;
};
}
