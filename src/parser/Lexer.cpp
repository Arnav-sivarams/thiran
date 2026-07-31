#include "parser/Lexer.hpp"

#include <cctype>

namespace thiran
{

Lexer::Lexer(
    const std::string& source
)
{
    this->source = source;
    position = 0;
}

char Lexer::currentChar()
{
    if(position >= source.size())
    {
        return '\0';
    }

    return source[position];
}

void Lexer::advance()
{
    position++;
}

void Lexer::skipWhitespace()
{
    while(std::isspace(currentChar()))
    {
        advance();
    }
}

Token Lexer::identifier()
{
    std::string value;

    while(std::isalnum(currentChar()) || currentChar() == '_')
    {
        value += currentChar();
        advance();
    }

    return Token(
        TokenType::Identifier,
        value
    );
}

std::vector<Token> Lexer::tokenize()
{
    std::vector<Token> tokens;

    while(currentChar() != '\0')
    {
        if(std::isspace(currentChar()))
        {
            skipWhitespace();
            continue;
        }

        if(std::isalpha(currentChar()) || currentChar() == '_')
        {
            tokens.push_back(identifier());
            continue;
        }

        if(std::isdigit(currentChar()) || currentChar() == '.')
        {
            std::string value;
            bool seenDot = false;

            while(std::isdigit(currentChar()) ||
                  (!seenDot && currentChar() == '.'))
            {
                seenDot = seenDot || currentChar() == '.';
                value += currentChar();
                advance();
            }

            tokens.emplace_back(TokenType::Number, value);
            continue;
        }

        if(currentChar() == '=')
        {
            tokens.push_back(
                Token(
                    TokenType::Equals,
                    "="
                )
            );

            advance();
            continue;
        }

        if(currentChar() == '(')
        {
            tokens.push_back(
                Token(
                    TokenType::LeftParen,
                    "("
                )
            );

            advance();
            continue;
        }

        if(currentChar() == ')')
        {
            tokens.push_back(
                Token(
                    TokenType::RightParen,
                    ")"
                )
            );

            advance();
            continue;
        }

        if(currentChar() == ',')
        {
            tokens.push_back(
                Token(
                    TokenType::Comma,
                    ","
                )
            );

            advance();
            continue;
        }

        if(currentChar() == '-')
        {
            tokens.emplace_back(TokenType::Minus, "-");
            advance();
            continue;
        }

        tokens.push_back(
            Token(
                TokenType::Unknown,
                std::string(1, currentChar())
            )
        );

        advance();
    }

    tokens.push_back(
        Token(
            TokenType::EndOfFile,
            ""
        )
    );

    return tokens;
}

}
