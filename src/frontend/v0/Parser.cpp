#include "frontend/v0/Parser.hpp"
#include "frontend/v0/Lexer.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace thiran::v0 {
namespace {
struct ParseError { Diagnostic diagnostic; };

class ParserImpl {
public:
    ParserImpl(std::vector<Token> tokens, std::string source)
        : tokens_(std::move(tokens)), source_(std::move(source)) {}

    Module module() {
        Module result;
        result.source = source_;
        result.span.begin = peek().span.begin;
        separators();
        while (!at(TokenKind::End)) {
            if (at(TokenKind::Import)) result.items.emplace_back(importDecl());
            else if (at(TokenKind::Fn) || at(TokenKind::Export)) result.items.emplace_back(functionDecl());
            else if (at(TokenKind::Let)) result.items.emplace_back(letStmt());
            else fail("unexpected token at module scope");
            requireSeparator(false);
            separators();
        }
        result.span.end = peek().span.end;
        return result;
    }

private:
    std::vector<Token> tokens_;
    std::string source_;
    std::size_t pos_ = 0;

    const Token& peek(std::size_t lookahead = 0) const {
        return tokens_[std::min(pos_ + lookahead, tokens_.size() - 1)];
    }
    bool at(TokenKind kind) const { return peek().kind == kind; }
    Token take() { return tokens_[pos_++]; }
    bool match(TokenKind kind) { if (!at(kind)) return false; take(); return true; }
    [[noreturn]] void fail(const std::string& message) const {
        throw ParseError{{source_, peek().span, message}};
    }
    Token expect(TokenKind kind, const std::string& message) {
        if (!at(kind)) fail(message);
        return take();
    }
    void softNewlines() { while (match(TokenKind::Newline)) {} }
    void separators() { while (at(TokenKind::Newline) || at(TokenKind::Semicolon)) take(); }
    void requireSeparator(bool body) {
        if (at(TokenKind::End) || (body && at(TokenKind::RightBrace))) return;
        if (!at(TokenKind::Newline) && !at(TokenKind::Semicolon)) fail("expected newline or semicolon between statements");
    }
    static SourceSpan joined(const SourceSpan& first, const SourceSpan& last) {
        return {first.begin, last.end};
    }
    template<typename Node>
    ExprPtr make(SourceSpan span, Node node) {
        return std::make_unique<Expr>(Expr{span, std::move(node)});
    }
    static int precedence(TokenKind kind) {
        switch (kind) {
            case TokenKind::Star: case TokenKind::DotStar:
            case TokenKind::Slash: case TokenKind::DotSlash: return 20;
            case TokenKind::Plus: case TokenKind::Minus: return 10;
            default: return -1;
        }
    }
    ExprPtr expression(int minPrecedence = 0, bool soft = false) {
        if (soft) softNewlines();
        auto left = prefix(soft);
        while (true) {
            if (soft) softNewlines();
            if (at(TokenKind::LeftParen) || at(TokenKind::LeftBracket) || at(TokenKind::Dot)) {
                left = postfix(std::move(left));
                continue;
            }
            int priority = precedence(peek().kind);
            if (priority < minPrecedence) break;
            auto operation = take();
            auto right = expression(priority + 1, soft);
            auto span = joined(left->span, right->span);
            left = make(span, BinaryExpr{operation.kind, std::move(left), std::move(right)});
        }
        return left;
    }
    ExprPtr prefix(bool soft) {
        if (soft) softNewlines();
        if (at(TokenKind::Minus)) {
            auto minus = take();
            auto operand = expression(30, soft);
            auto span = joined(minus.span, operand->span);
            return make(span, UnaryExpr{TokenKind::Minus, std::move(operand)});
        }
        if (at(TokenKind::Identifier)) {
            auto token = take(); return make(token.span, IdentifierExpr{token.text});
        }
        if (at(TokenKind::Integer)) {
            auto token = take(); return make(token.span, IntegerLiteralExpr{token.text});
        }
        if (at(TokenKind::True) || at(TokenKind::False)) {
            auto token = take(); return make(token.span, BooleanLiteralExpr{token.kind == TokenKind::True});
        }
        if (at(TokenKind::LeftParen)) {
            auto open = take(); softNewlines();
            if (at(TokenKind::RightParen)) fail("expected expression in parentheses");
            auto first = expression(0, true); softNewlines();
            if (!match(TokenKind::Comma)) {
                auto close = expect(TokenKind::RightParen, "missing ')' delimiter");
                first->span = joined(open.span, close.span);
                return first;
            }
            TupleExpr tuple;
            tuple.elements.push_back(std::move(first));
            softNewlines();
            if (!at(TokenKind::RightParen)) {
                do {
                    tuple.elements.push_back(expression(0, true)); softNewlines();
                } while (match(TokenKind::Comma) && (softNewlines(), !at(TokenKind::RightParen)));
            }
            auto close = expect(TokenKind::RightParen, "missing ')' delimiter in tuple");
            return make(joined(open.span, close.span), std::move(tuple));
        }
        if (at(TokenKind::LeftBracket)) return tensor();
        fail("unexpected token: expected expression");
    }
    ExprPtr tensor() {
        auto open = take();
        TensorLiteralExpr tensor;
        tensor.rows.emplace_back();
        softNewlines();
        if (at(TokenKind::RightBracket)) fail("malformed tensor literal: empty literal");
        while (true) {
            tensor.rows.back().push_back(expression(0, true));
            softNewlines();
            if (match(TokenKind::Comma)) {
                softNewlines();
                if (at(TokenKind::Semicolon) || at(TokenKind::RightBracket))
                    fail("malformed tensor literal: missing element after comma");
                continue;
            }
            if (match(TokenKind::Semicolon)) {
                if (tensor.rows.size() >= 2) fail("rank > 2 bracket tensor literal is unsupported");
                tensor.rows.emplace_back(); softNewlines();
                if (at(TokenKind::Semicolon) || at(TokenKind::RightBracket))
                    fail("malformed tensor literal: missing row after semicolon");
                continue;
            }
            break;
        }
        auto close = expect(TokenKind::RightBracket, "missing ']' delimiter in tensor literal");
        if (tensor.rows.size() == 2 && tensor.rows[0].size() != tensor.rows[1].size())
            throw ParseError{{source_, joined(open.span, close.span), "malformed tensor literal: ragged rows"}};
        return make(joined(open.span, close.span), std::move(tensor));
    }
    ExprPtr postfix(ExprPtr object) {
        if (at(TokenKind::Dot)) {
            take();
            auto member = expect(TokenKind::Identifier, "expected member name after '.'");
            auto span = joined(object->span, member.span);
            return make(span, MemberExpr{std::move(object), member.text});
        }
        if (at(TokenKind::LeftParen)) {
            take(); CallExpr call{std::move(object), {}};
            softNewlines();
            if (!at(TokenKind::RightParen)) {
                do {
                    call.arguments.push_back(expression(0, true)); softNewlines();
                } while (match(TokenKind::Comma) && (softNewlines(), !at(TokenKind::RightParen)));
            }
            auto close = expect(TokenKind::RightParen, "missing ')' delimiter in call");
            auto span = joined(call.callee->span, close.span);
            return make(span, std::move(call));
        }
        take();
        IndexExpr index{std::move(object), {}};
        softNewlines();
        if (at(TokenKind::RightBracket)) fail("invalid index/slice syntax: empty selector list");
        do {
            index.axes.push_back(selector()); softNewlines();
            if (!match(TokenKind::Comma)) break;
            softNewlines();
            if (at(TokenKind::RightBracket)) fail("invalid index/slice syntax: missing axis after comma");
        } while (true);
        auto close = expect(TokenKind::RightBracket, "missing ']' delimiter in index/slice");
        auto span = joined(index.object->span, close.span);
        return make(span, std::move(index));
    }
    AxisSelector selector() {
        softNewlines();
        SourceSpan first = peek().span;
        ExprPtr start;
        if (!at(TokenKind::Colon)) {
            if (at(TokenKind::Comma) || at(TokenKind::RightBracket)) fail("invalid index/slice syntax: missing selector");
            start = expression(0, true); softNewlines();
        }
        if (!match(TokenKind::Colon)) return IndexSelector{start->span, std::move(start)};
        SliceSelector slice;
        slice.span.begin = start ? start->span.begin : first.begin;
        softNewlines();
        if (!at(TokenKind::Colon) && !at(TokenKind::Comma) && !at(TokenKind::RightBracket)) {
            slice.end = expression(0, true); softNewlines();
        }
        if (match(TokenKind::Colon)) {
            softNewlines();
            if (!at(TokenKind::Comma) && !at(TokenKind::RightBracket) && !at(TokenKind::Colon)) {
                slice.step = expression(0, true); softNewlines();
            }
            if (at(TokenKind::Colon)) fail("invalid index/slice syntax: too many colons");
        }
        slice.start = std::move(start);
        slice.span.end = peek().span.begin;
        return slice;
    }
    TypeSyntax typeSyntax(bool allowTuple = true) {
        softNewlines();
        if (allowTuple && at(TokenKind::LeftParen)) {
            auto open = take(); TypeSyntax tuple; tuple.span.begin = open.span.begin;
            softNewlines();
            tuple.elements.push_back(typeSyntax(false)); softNewlines();
            expect(TokenKind::Comma, "malformed tuple type: expected ','"); softNewlines();
            do { tuple.elements.push_back(typeSyntax(false)); softNewlines(); }
            while (match(TokenKind::Comma) && (softNewlines(), !at(TokenKind::RightParen)));
            tuple.span.end = expect(TokenKind::RightParen, "missing ')' delimiter in tuple type").span.end;
            return tuple;
        }
        auto name = expect(TokenKind::Identifier, "malformed type syntax: expected type name");
        TypeSyntax type; type.name = name.text; type.span = name.span;
        if (match(TokenKind::Less)) {
            if (type.name != "Tensor" && type.name != "Buffer")
                fail("malformed type syntax: generic form only supported for Tensor/Buffer");
            type.elements.push_back(typeSyntax(false)); softNewlines();
            if (type.name == "Tensor") {
                expect(TokenKind::Comma, "malformed Tensor type: expected ',' before rank"); softNewlines();
                type.rank = expect(TokenKind::Integer, "malformed Tensor type: expected integer rank").text;
                softNewlines();
            }
            type.span.end = expect(TokenKind::Greater, "malformed type syntax: missing '>' generic delimiter").span.end;
        } else if (type.name == "Tensor" || type.name == "Buffer") {
            fail("malformed type syntax: built-in generic requires '<...>'");
        }
        return type;
    }
    LetStmt letStmt() {
        auto begin = take(); bool mutableBinding = match(TokenKind::Mut);
        auto name = expect(TokenKind::Identifier, "malformed let binding: expected identifier");
        expect(TokenKind::Equal, "malformed let binding: expected '='");
        auto value = expression();
        return {joined(begin.span, value->span), name.text, mutableBinding, std::move(value)};
    }
    std::vector<StmtPtr> body() {
        expect(TokenKind::LeftBrace, "expected '{' to begin block");
        std::vector<StmtPtr> result;
        separators();
        while (!at(TokenKind::RightBrace)) {
            if (at(TokenKind::End)) fail("missing '}' delimiter in block");
            result.push_back(std::make_unique<Statement>(statement()));
            requireSeparator(true); separators();
        }
        take();
        return result;
    }
    Statement statement() {
        if (at(TokenKind::Let)) return Statement{letStmt()};
        if (at(TokenKind::Return)) {
            auto begin = take(); auto value = expression();
            return Statement{ReturnStmt{joined(begin.span, value->span), std::move(value)}};
        }
        if (at(TokenKind::If)) {
            auto begin=take(); auto condition=expression();
            IfStmt n; n.condition=std::move(condition); n.thenBody=body();
            auto end=tokens_[pos_-1].span;
            // A newline between a closed then-block and else belongs to the if statement.
            auto saved=pos_; softNewlines();
            if (match(TokenKind::Else)) { n.hasElse=true; softNewlines(); n.elseBody=body(); end=tokens_[pos_-1].span; }
            else pos_=saved;
            n.span=joined(begin.span,end); return Statement{std::move(n)};
        }
        if (at(TokenKind::For)) {
            auto begin=take();
            auto variable=expect(TokenKind::Identifier,"malformed for: expected loop variable");
            expect(TokenKind::In,"malformed for: expected 'in'");
            ForStmt n; n.variable=variable.text; n.start=expression();
            if (match(TokenKind::Colon)) {
                n.end=expression();
            } else { n.iterable=std::move(n.start); }
            n.body=body(); n.span=joined(begin.span,tokens_[pos_-1].span);
            return Statement{std::move(n)};
        }
        if (at(TokenKind::While)) {
            auto begin=take(); WhileStmt n; n.condition=expression(); n.body=body();
            n.span=joined(begin.span,tokens_[pos_-1].span); return Statement{std::move(n)};
        }
        if (at(TokenKind::Break)) {
            auto token=take(); return Statement{BreakStmt{token.span}};
        }
        if (at(TokenKind::Continue)) {
            auto token=take(); return Statement{ContinueStmt{token.span}};
        }
        if (at(TokenKind::Identifier) && peek(1).kind == TokenKind::Equal) {
            auto name = take(); take(); auto value = expression();
            return Statement{RebindStmt{joined(name.span, value->span), name.text, std::move(value)}};
        }
        fail("unexpected token in function body");
    }
    ImportDecl importDecl() {
        auto begin = take();
        auto path = expect(TokenKind::String, "malformed import: expected quoted path");
        expect(TokenKind::As, "malformed import: expected 'as'");
        auto alias = expect(TokenKind::Identifier, "malformed import: expected alias");
        return {joined(begin.span, alias.span), path.text.substr(1, path.text.size() - 2), alias.text};
    }
    FunctionDecl functionDecl() {
        bool exported = match(TokenKind::Export);
        auto begin = expect(TokenKind::Fn, "malformed function declaration: expected 'fn'");
        auto name = expect(TokenKind::Identifier, "malformed function signature: expected function name");
        expect(TokenKind::LeftParen, "malformed function signature: expected '('");
        FunctionDecl function; function.name = name.text; function.exported = exported;
        function.span.begin = exported ? tokens_[pos_ - 3].span.begin : begin.span.begin;
        softNewlines();
        if (!at(TokenKind::RightParen)) {
            do {
                auto paramName = expect(TokenKind::Identifier, "malformed function parameter: expected identifier");
                expect(TokenKind::Colon, "malformed function parameter: expected ':' and explicit type");
                auto type = typeSyntax(false);
                function.parameters.push_back({joined(paramName.span, type.span), paramName.text, std::move(type)});
                softNewlines();
            } while (match(TokenKind::Comma) && (softNewlines(), !at(TokenKind::RightParen)));
        }
        expect(TokenKind::RightParen, "malformed function signature: missing ')'");
        softNewlines();
        if (match(TokenKind::Arrow)) function.resultType = typeSyntax();
        else if (exported) fail("malformed exported function signature: result type required");
        softNewlines();
        if (!at(TokenKind::LeftBrace)) fail("malformed function signature: expected '{'");
        function.body=body();
        function.span.end = tokens_[pos_-1].span.end;
        return function;
    }
};
}

ParseResult parse(std::string_view source, std::string sourceName) {
    auto lexical = lex(source, sourceName);
    ParseResult result;
    result.diagnostics = std::move(lexical.diagnostics);
    if (!result.diagnostics.empty()) return result;
    try {
        result.module = ParserImpl(std::move(lexical.tokens), std::move(sourceName)).module();
    } catch (const ParseError& error) {
        result.diagnostics.push_back(error.diagnostic);
    } catch (const std::exception& error) {
        // Keep implementation exceptions out of the test/user-facing interface.
        auto lexicalEnd = lex(source, sourceName).tokens.back().span;
        result.diagnostics.push_back({sourceName, lexicalEnd, std::string("parser implementation error: ") + error.what()});
    }
    return result;
}
}
