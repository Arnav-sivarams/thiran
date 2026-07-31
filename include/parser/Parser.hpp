#pragma once

#include <vector>

#include "parser/Token.hpp"
#include "ir/Graph.hpp"

namespace thiran
{

class Parser
{
private:

    std::vector<Token> tokens;

    int position;

    std::vector<std::string> errors;

    Token currentToken();

    void advance();

    bool match(TokenType type);

    void parseStatement(Graph& graph);

public:

    Parser(
        const std::vector<Token>& tokens
    );

    Graph parse();

    bool hasErrors() const;

    const std::vector<std::string>& diagnostics() const;
};

}
