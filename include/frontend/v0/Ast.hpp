#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include "frontend/v0/Token.hpp"

namespace thiran::v0 {
struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

struct IdentifierExpr { std::string name; };
struct IntegerLiteralExpr { std::string spelling; };
struct BooleanLiteralExpr { bool value; };
struct TensorLiteralExpr { std::vector<std::vector<ExprPtr>> rows; };
struct TupleExpr { std::vector<ExprPtr> elements; };
struct UnaryExpr { TokenKind op; ExprPtr operand; };
struct BinaryExpr { TokenKind op; ExprPtr left, right; };
struct CallExpr { ExprPtr callee; std::vector<ExprPtr> arguments; };
struct MemberExpr { ExprPtr object; std::string member; };
struct IndexSelector { SourceSpan span; ExprPtr value; };
struct SliceSelector { SourceSpan span; ExprPtr start, end, step; };
using AxisSelector = std::variant<IndexSelector, SliceSelector>;
struct IndexExpr { ExprPtr object; std::vector<AxisSelector> axes; };

struct Expr {
    SourceSpan span;
    std::variant<IdentifierExpr, IntegerLiteralExpr, BooleanLiteralExpr,
        TensorLiteralExpr, TupleExpr, UnaryExpr, BinaryExpr, CallExpr,
        MemberExpr, IndexExpr> node;
};

struct TypeSyntax {
    SourceSpan span;
    std::string name; // empty for tuple type syntax
    std::vector<TypeSyntax> elements; // tuple members or built-in generic type arguments
    std::optional<std::string> rank; // spelling of Tensor compile-time rank
};
struct Parameter { SourceSpan span; std::string name; TypeSyntax type; };
struct LetStmt { SourceSpan span; std::string name; bool mutableBinding; ExprPtr value; };
struct RebindStmt { SourceSpan span; std::string name; ExprPtr value; };
struct ReturnStmt { SourceSpan span; ExprPtr value; };
using Statement = std::variant<LetStmt, RebindStmt, ReturnStmt>;
struct ImportDecl { SourceSpan span; std::string path, alias; };
struct FunctionDecl {
    SourceSpan span;
    std::string name;
    bool exported;
    std::vector<Parameter> parameters;
    std::optional<TypeSyntax> resultType;
    std::vector<Statement> body;
};
using TopLevel = std::variant<ImportDecl, FunctionDecl, LetStmt>;
struct Module {
    SourceSpan span;
    std::string source;
    std::vector<TopLevel> items;
};

// Developer-only structural dump. Not a stable public serialization format.
std::string dump(const Module& module);
}
