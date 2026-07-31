#pragma once

#include <string>

namespace thiran
{

enum class TokenType
{
    Identifier,
    Number,

    Equals,

    LeftParen,
    RightParen,

    Comma,
    Minus,

    EndOfFile,

    Unknown
};

class Token
{
public:

    TokenType type;

    std::string text;

    Token();

    Token(
        TokenType type,
        const std::string& text
    );
};

}
