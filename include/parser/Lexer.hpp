#pragma once

#include <string>
#include <vector>

#include "parser/Token.hpp"

namespace thiran
{

class Lexer
{
private:

    std::string source;

    int position;

    char currentChar();

    void advance();

    void skipWhitespace();

    Token identifier();

public:

    Lexer(
        const std::string& source
    );

    std::vector<Token> tokenize();
};

}