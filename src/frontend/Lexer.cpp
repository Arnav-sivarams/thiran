#include "frontend/Lexer.hpp"

#include <cctype>

namespace thiran::frontend
{
std::vector<Token> Lexer::tokenize(const SourceManager& sources, SourceId source,
                                   std::vector<std::string>& diagnostics)
{
    std::vector<Token> tokens;
    const auto* file = sources.source(source);
    if(file == nullptr) return tokens;
    const auto add = [&](TokenKind kind, std::string text, std::size_t begin, std::size_t end)
    { tokens.push_back({kind, std::move(text), {sources.location(source, begin), sources.location(source, end)}}); };
    const auto error = [&](std::size_t offset, const std::string& message)
    {
        const auto location = sources.location(source, offset);
        diagnostics.push_back(file->displayPath + ":" + std::to_string(location.line) + ":" +
                              std::to_string(location.column) + ": error: " + message);
    };
    std::size_t i = 0;
    while(i < file->contents.size())
    {
        const unsigned char value = static_cast<unsigned char>(file->contents[i]);
        if(value == 0) { error(i++, "embedded NUL in source"); continue; }
        if(file->contents[i] == '\n') { add(TokenKind::Newline, "\n", i, i + 1); ++i; continue; }
        if(file->contents[i] == '\r' || file->contents[i] == ' ' || file->contents[i] == '\t') { ++i; continue; }
        if(file->contents[i] == '/' && i + 1 < file->contents.size() && file->contents[i + 1] == '/')
        { while(i < file->contents.size() && file->contents[i] != '\n') ++i; continue; }
        const auto begin = i;
        if(std::isalpha(value) || file->contents[i] == '_')
        {
            while(i < file->contents.size() && (std::isalnum(static_cast<unsigned char>(file->contents[i])) || file->contents[i] == '_')) ++i;
            add(TokenKind::Identifier, file->contents.substr(begin, i - begin), begin, i); continue;
        }
        if(std::isdigit(value))
        {
            bool dot = false;
            while(i < file->contents.size() && (std::isdigit(static_cast<unsigned char>(file->contents[i])) || (!dot && file->contents[i] == '.')))
            { dot = dot || file->contents[i] == '.'; ++i; }
            add(TokenKind::Number, file->contents.substr(begin, i - begin), begin, i); continue;
        }
        if(file->contents[i] == '"')
        {
            ++i; std::string decoded; bool valid = true;
            while(i < file->contents.size() && file->contents[i] != '"' && file->contents[i] != '\n')
            {
                char c = file->contents[i++];
                if(c == '\\')
                {
                    if(i >= file->contents.size() || (file->contents[i] != '\\' && file->contents[i] != '"'))
                    { error(i - 1, "invalid import string escape"); valid = false; break; }
                    c = file->contents[i++];
                }
                decoded += c;
            }
            if(i >= file->contents.size() || file->contents[i] != '"') { if(valid) error(begin, "unterminated import string"); }
            else { ++i; if(valid) add(TokenKind::String, decoded, begin, i); }
            continue;
        }
        TokenKind kind = TokenKind::Unknown;
        switch(file->contents[i])
        {
            case '=': kind = TokenKind::Equals; break; case '(': kind = TokenKind::LeftParen; break;
            case ')': kind = TokenKind::RightParen; break; case ',': kind = TokenKind::Comma; break;
            case '-': kind = TokenKind::Minus; break; case '.': kind = TokenKind::Dot; break;
            case '{': kind = TokenKind::LeftBrace; break; case '}': kind = TokenKind::RightBrace; break;
            default: error(i, "unexpected character '" + std::string(1, file->contents[i]) + "'"); break;
        }
        add(kind, std::string(1, file->contents[i]), i, i + 1); ++i;
    }
    add(TokenKind::EndOfFile, {}, file->contents.size(), file->contents.size());
    return tokens;
}
}
