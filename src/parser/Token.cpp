#include "parser/Token.hpp"

namespace thiran
{

Token::Token()
{
    type = TokenType::Unknown;
    text = "";
}

Token::Token(
    TokenType type,
    const std::string& text
)
{
    this->type = type;
    this->text = text;
}

}