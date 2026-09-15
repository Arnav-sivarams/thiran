#include "frontend/v0/Lexer.hpp"

#include <unordered_map>

namespace thiran::v0 {
namespace {
bool digit(char c) { return c >= '0' && c <= '9'; }
bool letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool identStart(char c) { return letter(c) || c == '_'; }
bool identPart(char c) { return letter(c) || digit(c) || c == '_'; }
const std::unordered_map<std::string, TokenKind> keywords = {
    {"let", TokenKind::Let}, {"mut", TokenKind::Mut}, {"fn", TokenKind::Fn},
    {"export", TokenKind::Export}, {"return", TokenKind::Return},
    {"import", TokenKind::Import}, {"as", TokenKind::As},
    {"true", TokenKind::True}, {"false", TokenKind::False},
    {"if", TokenKind::If}, {"else", TokenKind::Else},
    {"for", TokenKind::For}, {"in", TokenKind::In},
    {"while", TokenKind::While}, {"break", TokenKind::Break},
    {"continue", TokenKind::Continue}, {"struct", TokenKind::Struct}
};
}

LexResult lex(std::string_view source, std::string sourceName) {
    LexResult result;
    std::size_t offset = 0, line = 1, column = 1;
    // Source 0 is the sole in-memory file in each V0 parse; Module::source owns its display identity.
    auto location = [&] { return SourceLocation{0, offset, line, column}; };
    auto advance = [&] {
        char c = source[offset++];
        if (c == '\n') { ++line; column = 1; } else { ++column; }
    };
    auto emit = [&](TokenKind kind, SourceLocation begin, std::size_t start) {
        result.tokens.push_back({kind, std::string(source.substr(start, offset - start)), {begin, location()}});
    };
    auto error = [&](SourceLocation begin, std::string message) {
        result.diagnostics.push_back({sourceName, {begin, location()}, std::move(message)});
    };
    while (offset < source.size()) {
        char c = source[offset];
        if (c == ' ' || c == '\t' || c == '\r') { advance(); continue; }
        auto begin = location();
        auto start = offset;
        if (c == '\n') { advance(); emit(TokenKind::Newline, begin, start); continue; }
        if (c == '/' && offset + 1 < source.size() && source[offset + 1] == '/') {
            while (offset < source.size() && source[offset] != '\n') advance();
            continue;
        }
        if (identStart(c)) {
            do { advance(); } while (offset < source.size() && identPart(source[offset]));
            auto word = std::string(source.substr(start, offset - start));
            auto found = keywords.find(word);
            emit(found == keywords.end() ? TokenKind::Identifier : found->second, begin, start);
            continue;
        }
        if (digit(c)) {
            do { advance(); } while (offset < source.size() && digit(source[offset]));
            TokenKind kind = TokenKind::Integer;
            if (offset + 1 < source.size() && source[offset] == '.' &&
                digit(source[offset + 1])) {
                kind = TokenKind::Real;
                advance();
                do { advance(); } while (offset < source.size() && digit(source[offset]));
            }
            emit(kind, begin, start);
            continue;
        }
        if (c == '"') {
            advance();
            bool closed = false;
            while (offset < source.size() && source[offset] != '\n') {
                if (source[offset] == '"') { advance(); closed = true; break; }
                if (source[offset] == '\\') {
                    advance();
                    if (offset >= source.size() || source[offset] == '\n') break;
                }
                advance();
            }
            if (!closed) error(begin, "unterminated string literal");
            else emit(TokenKind::String, begin, start);
            continue;
        }
        if (c == '.' || c == '-') {
            advance();
            if (offset < source.size() && ((c == '.' && (source[offset] == '*' || source[offset] == '/')) ||
                (c == '-' && source[offset] == '>'))) {
                char second = source[offset]; advance();
                emit(c == '-' ? TokenKind::Arrow : second == '*' ? TokenKind::DotStar : TokenKind::DotSlash, begin, start);
            } else emit(c == '.' ? TokenKind::Dot : TokenKind::Minus, begin, start);
            continue;
        }
        TokenKind kind;
        switch (c) {
            case '+': kind = TokenKind::Plus; break;
            case '*': kind = TokenKind::Star; break;
            case '/': kind = TokenKind::Slash; break;
            case '=': kind = TokenKind::Equal; break;
            case ':': kind = TokenKind::Colon; break;
            case ',': kind = TokenKind::Comma; break;
            case ';': kind = TokenKind::Semicolon; break;
            case '(': kind = TokenKind::LeftParen; break;
            case ')': kind = TokenKind::RightParen; break;
            case '[': kind = TokenKind::LeftBracket; break;
            case ']': kind = TokenKind::RightBracket; break;
            case '{': kind = TokenKind::LeftBrace; break;
            case '}': kind = TokenKind::RightBrace; break;
            case '<': kind = TokenKind::Less; break;
            case '>': kind = TokenKind::Greater; break;
            default: advance(); error(begin, "malformed or unsupported token"); continue;
        }
        advance(); emit(kind, begin, start);
    }
    auto end = location();
    result.tokens.push_back({TokenKind::End, "", {end, end}});
    return result;
}
}
