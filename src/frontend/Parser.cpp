#include "frontend/Parser.hpp"

#include <cctype>
#include <string_view>

#include "frontend/Lexer.hpp"

namespace thiran::frontend
{
namespace
{
bool reserved(const std::string& name)
{
    return name.rfind("__thiran_module_", 0) == 0 || name.rfind("__thiran_call_", 0) == 0;
}

struct LineParser
{
    const SourceManager& sources;
    SourceId source;
    std::string_view text;
    std::size_t base;
    std::size_t position = 0;
    std::vector<std::string>& diagnostics;

    void spaces() { while(position < text.size() && (text[position] == ' ' || text[position] == '\t' || text[position] == '\r')) ++position; }
    SourceSpan span(std::size_t begin, std::size_t end) const { return {sources.location(source, base + begin), sources.location(source, base + end)}; }
    void error(std::size_t at, const std::string& message)
    {
        const auto* file = sources.source(source);
        const auto location = sources.location(source, base + at);
        diagnostics.push_back((file ? file->displayPath : "<invalid>") + ":" + std::to_string(location.line) + ":" +
                              std::to_string(location.column) + ": error: " + message);
    }
    std::pair<std::string, SourceSpan> identifierWithSpan()
    {
        spaces(); const auto begin = position;
        if(position >= text.size() || !(std::isalpha(static_cast<unsigned char>(text[position])) || text[position] == '_')) return {{}, span(begin, begin)};
        while(position < text.size() && (std::isalnum(static_cast<unsigned char>(text[position])) || text[position] == '_')) ++position;
        return {std::string(text.substr(begin, position - begin)), span(begin, position)};
    }
    std::string identifier() { return identifierWithSpan().first; }
    bool character(char value) { spaces(); if(position < text.size() && text[position] == value) { ++position; return true; } return false; }
    bool done() { spaces(); return position == text.size() || (position + 1 < text.size() && text[position] == '/' && text[position + 1] == '/'); }
    std::optional<std::string> stringLiteral()
    {
        spaces(); if(position >= text.size() || text[position] != '"') return std::nullopt;
        ++position; std::string value;
        while(position < text.size() && text[position] != '"')
        {
            char c = text[position++];
            if(c == '\0') { error(position - 1, "embedded NUL in import string"); return std::nullopt; }
            if(c == '\\')
            {
                if(position >= text.size()) { error(position - 1, "unterminated import string"); return std::nullopt; }
                const char escaped = text[position++];
                if(escaped != '\\' && escaped != '"') { error(position - 2, "invalid import string escape"); return std::nullopt; }
                c = escaped;
            }
            value += c;
        }
        if(position >= text.size()) { error(position, "unterminated import string"); return std::nullopt; }
        ++position; return value;
    }
};

std::optional<AssignmentStmt> assignment(LineParser& line, std::string name, bool exported, std::size_t start)
{
    if(name.empty()) { line.error(start, "expected a node name"); return std::nullopt; }
    if(reserved(name)) { line.error(start, "identifier uses reserved internal prefix"); return std::nullopt; }
    if(!line.character('=')) { line.error(line.position, "expected '=' after " + name); return std::nullopt; }
    auto [callable, callableSpan] = line.identifierWithSpan();
    if(callable.empty()) { line.error(line.position, "expected an operation after " + name + " ="); return std::nullopt; }
    std::string alias;
    if(line.character('.'))
    {
        alias = callable; const auto member = line.identifier();
        if(member.empty()) { line.error(line.position, "expected function name after '.'"); return std::nullopt; }
        callable += "." + member;
    }
    if(!line.character('(')) { line.error(line.position, "expected '(' after operation " + callable); return std::nullopt; }
    AssignmentStmt statement{name, callable, {}, exported, line.span(start, line.text.size())};
    line.spaces();
    while(line.position < line.text.size() && line.text[line.position] != ')')
    {
        const auto argumentStart = line.position; line.spaces(); std::string spelling; bool numeric = false;
        if(line.position < line.text.size() && (line.text[line.position] == '-' || std::isdigit(static_cast<unsigned char>(line.text[line.position])) || line.text[line.position] == '.'))
        {
            numeric = true; if(line.text[line.position] == '-') spelling += line.text[line.position++];
            while(line.position < line.text.size() && (std::isdigit(static_cast<unsigned char>(line.text[line.position])) || line.text[line.position] == '.')) spelling += line.text[line.position++];
        }
        else spelling = line.identifier();
        ValueReference reference{{}, spelling, line.span(argumentStart, line.position)};
        if(!numeric && line.character('.'))
        {
            reference.alias = spelling; reference.name = line.identifier(); spelling += "." + reference.name;
            if(reference.name.empty()) line.error(line.position, "expected exported name after '.'");
        }
        if(spelling.empty()) { line.error(argumentStart, "expected operation argument"); return std::nullopt; }
        statement.arguments.push_back({spelling, numeric, reference, line.span(argumentStart, line.position)});
        line.spaces();
        if(line.position < line.text.size() && line.text[line.position] != ')')
        {
            if(!line.character(',')) { line.error(line.position, "expected ',' between arguments"); return std::nullopt; }
            line.spaces();
            if(line.position < line.text.size() && line.text[line.position] == ')') { line.error(line.position, "trailing commas are not allowed"); return std::nullopt; }
        }
    }
    if(!line.character(')')) { line.error(line.position, "expected ')' after operation arguments"); return std::nullopt; }
    if(!line.done()) { line.error(line.position, "unexpected text after assignment"); return std::nullopt; }
    (void)callableSpan; (void)alias;
    return statement;
}
}

std::optional<ModuleAst> Parser::parse(const SourceManager& sources, SourceId source,
                                       std::vector<std::string>& diagnostics)
{
    const auto* file = sources.source(source);
    if(file == nullptr) { diagnostics.push_back("<invalid>:1:1: error: invalid source identifier"); return std::nullopt; }
    const auto tokens = Lexer::tokenize(sources, source, diagnostics);
    if(tokens.empty() || !diagnostics.empty()) return std::nullopt;
    ModuleAst module{source, {}, {}, {}};
    bool declarationsSeen = false, assignmentsSeen = false;
    FunctionDecl* function = nullptr;
    std::size_t start = 0;
    while(start <= file->contents.size())
    {
        auto end = file->contents.find('\n', start); if(end == std::string::npos) end = file->contents.size();
        LineParser line{sources, source, std::string_view(file->contents).substr(start, end - start), start, 0, diagnostics};
        line.spaces();
        if(!line.done())
        {
            const auto statementStart = line.position; auto [first, firstSpan] = line.identifierWithSpan(); line.spaces();
            if(function != nullptr)
            {
                if(!function->returns.empty() && first != "return" && first.empty() == false)
                    line.error(statementStart, "return must be the final statement in function '" + function->name + "'");
                if(first.empty() && line.character('}'))
                {
                    if(!line.done()) line.error(line.position, "unexpected text after function body");
                    function->bodySpan.end = sources.location(source, start + line.position);
                    function->span.end = function->bodySpan.end; function = nullptr;
                }
                else if(first == "return" && (line.position >= line.text.size() || line.text[line.position] != '='))
                {
                    auto [value, valueSpan] = line.identifierWithSpan();
                    if(value.empty()) line.error(line.position, "return requires a local tensor value");
                    else if(!line.done()) line.error(line.position, "return expressions are not allowed; return a name");
                    else function->returns.push_back({value, line.span(statementStart, line.position), valueSpan});
                }
                else
                {
                    if(first == "export") line.error(statementStart, "local bindings cannot be exported");
                    else if(auto parsed = assignment(line, first, false, statementStart)) function->locals.push_back(std::move(*parsed));
                }
            }
            else
            {
                const bool importDeclaration = first == "import" && line.position < line.text.size() && line.text[line.position] == '"';
                if(importDeclaration)
                {
                    if(declarationsSeen) line.error(statementStart, "imports must precede assignments");
                    auto path = line.stringLiteral(); const auto asPosition = line.position; const auto as = line.identifier();
                    const auto aliasPosition = line.position; const auto alias = line.identifier();
                    if(path && (as != "as" || alias.empty())) line.error(asPosition, "import requires 'as <alias>'");
                    else if(path && reserved(alias)) line.error(aliasPosition, "import alias uses reserved internal prefix");
                    else if(path && !line.done()) line.error(line.position, "unexpected text after import");
                    else if(path) module.imports.push_back({*path, alias, line.span(statementStart, aliasPosition)});
                }
                else
                {
                    declarationsSeen = true; bool exported = false; std::string name = first;
                    if(first == "export" && line.position < line.text.size() &&
                       (std::isalpha(static_cast<unsigned char>(line.text[line.position])) || line.text[line.position] == '_'))
                    { exported = true; name = line.identifier(); }
                    line.spaces();
                    const bool looksFunctionDeclaration = name == "fn" && line.position < line.text.size() &&
                        (std::isalpha(static_cast<unsigned char>(line.text[line.position])) || line.text[line.position] == '_');
                    const bool functionDeclaration = looksFunctionDeclaration && !assignmentsSeen;
                    if(functionDeclaration)
                    {
                        auto [functionName, nameSpan] = line.identifierWithSpan();
                        if(functionName.empty()) line.error(line.position, "expected function name after 'fn'");
                        else if(reserved(functionName)) line.error(line.position, "function name uses reserved internal prefix");
                        else if(!line.character('(')) line.error(line.position, "malformed parameter list for function '" + functionName + "'");
                        else
                        {
                            FunctionDecl declaration; declaration.name = functionName; declaration.exported = exported;
                            declaration.fnSpan = firstSpan; declaration.nameSpan = nameSpan; declaration.span.begin = firstSpan.begin;
                            line.spaces();
                            while(line.position < line.text.size() && line.text[line.position] != ')')
                            {
                                auto [parameter, parameterSpan] = line.identifierWithSpan();
                                if(parameter.empty()) { line.error(line.position, "malformed parameter list for function '" + functionName + "'"); break; }
                                declaration.parameters.push_back({parameter, parameterSpan}); line.spaces();
                                if(line.position < line.text.size() && line.text[line.position] != ')')
                                {
                                    if(!line.character(',')) { line.error(line.position, "malformed parameter list for function '" + functionName + "'"); break; }
                                    line.spaces();
                                    if(line.position < line.text.size() && line.text[line.position] == ')') { line.error(line.position, "trailing commas are not allowed"); break; }
                                }
                            }
                            if(!line.character(')') || !line.character('{')) line.error(line.position, "malformed function declaration");
                            else if(!line.done()) line.error(line.position, "unexpected text after function declaration");
                            else
                            {
                                declaration.bodySpan.begin = sources.location(source, start + line.position - 1);
                                module.functions.push_back(std::move(declaration)); function = &module.functions.back();
                            }
                        }
                    }
                    else
                    {
                        assignmentsSeen = true;
                        if(looksFunctionDeclaration) line.error(statementStart, "function declarations must precede module assignments");
                        else if(first == "return" && !exported && (line.position >= line.text.size() || line.text[line.position] != '=')) line.error(statementStart, "return is only allowed inside a function");
                        else if(auto parsed = assignment(line, name, exported, statementStart)) module.assignments.push_back(std::move(*parsed));
                    }
                }
            }
        }
        if(end == file->contents.size()) break;
        start = end + 1;
    }
    if(function != nullptr) diagnostics.push_back(file->displayPath + ":" + std::to_string(function->span.begin.line) + ":" +
        std::to_string(function->span.begin.column) + ": error: unterminated function body");
    return diagnostics.empty() ? std::optional<ModuleAst>(std::move(module)) : std::nullopt;
}
}
