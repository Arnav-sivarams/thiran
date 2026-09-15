#include "frontend/v0/Ast.hpp"
#include <sstream>
#include <type_traits>

namespace thiran::v0 {
namespace {
std::string op(TokenKind kind) {
    switch (kind) {
        case TokenKind::Plus: return "+";
        case TokenKind::Minus: return "-";
        case TokenKind::Star: return "*";
        case TokenKind::DotStar: return ".*";
        case TokenKind::Slash: return "/";
        case TokenKind::DotSlash: return "./";
        default: return "?";
    }
}
void expr(std::ostringstream& out, const Expr& value) {
    std::visit([&](const auto& n) {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, IdentifierExpr>) out << "id(" << n.name << ')';
        else if constexpr (std::is_same_v<T, IntegerLiteralExpr>) out << "int(" << n.spelling << ')';
        else if constexpr (std::is_same_v<T, BooleanLiteralExpr>) out << "bool(" << (n.value ? "true" : "false") << ')';
        else if constexpr (std::is_same_v<T, TensorLiteralExpr>) {
            out << "tensor(";
            for (std::size_t i = 0; i < n.rows.size(); ++i) {
                if (i) out << ';';
                out << '[';
                for (std::size_t j = 0; j < n.rows[i].size(); ++j) {
                    if (j) out << ',';
                    expr(out, *n.rows[i][j]);
                }
                out << ']';
            }
            out << ')';
        } else if constexpr (std::is_same_v<T, TupleExpr>) {
            out << "tuple(";
            for (std::size_t i = 0; i < n.elements.size(); ++i) {
                if (i) out << ',';
                expr(out, *n.elements[i]);
            }
            out << ')';
        } else if constexpr (std::is_same_v<T, UnaryExpr>) {
            out << "unary(" << op(n.op) << ','; expr(out, *n.operand); out << ')';
        } else if constexpr (std::is_same_v<T, BinaryExpr>) {
            out << "binary(" << op(n.op) << ','; expr(out, *n.left);
            out << ','; expr(out, *n.right); out << ')';
        } else if constexpr (std::is_same_v<T, CallExpr>) {
            out << "call("; expr(out, *n.callee);
            for (auto& arg : n.arguments) { out << ','; expr(out, *arg); }
            out << ')';
        } else if constexpr (std::is_same_v<T, MemberExpr>) {
            out << "member("; expr(out, *n.object); out << ',' << n.member << ')';
        } else if constexpr (std::is_same_v<T, IndexExpr>) {
            out << "index("; expr(out, *n.object);
            for (auto& axis : n.axes) {
                out << ',';
                std::visit([&](const auto& selector) {
                    using S = std::decay_t<decltype(selector)>;
                    if constexpr (std::is_same_v<S, IndexSelector>) {
                        out << "at("; expr(out, *selector.value); out << ')';
                    } else {
                        out << "slice(";
                        if (selector.start) expr(out, *selector.start); else out << '_';
                        out << ',';
                        if (selector.end) expr(out, *selector.end); else out << '_';
                        out << ',';
                        if (selector.step) expr(out, *selector.step); else out << '_';
                        out << ')';
                    }
                }, axis);
            }
            out << ')';
        }
    }, value.node);
}
void type(std::ostringstream& out, const TypeSyntax& t) {
    if (t.name.empty()) {
        out << '(';
        for (std::size_t i = 0; i < t.elements.size(); ++i) {
            if (i) out << ',';
            type(out, t.elements[i]);
        }
        out << ')';
    } else {
        out << t.name;
        if (!t.elements.empty()) {
            out << '<'; type(out, t.elements[0]);
            if (t.rank) out << ',' << *t.rank;
            out << '>';
        }
    }
}
void stmt(std::ostringstream& out, const Statement& s) {
    std::visit([&](const auto& n) {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, LetStmt>) {
            out << "let(" << (n.mutableBinding ? "mut," : "imm,") << n.name << ',';
            expr(out, *n.value); out << ')';
        } else if constexpr (std::is_same_v<T, RebindStmt>) {
            out << "rebind(" << n.name << ','; expr(out, *n.value); out << ')';
        } else {
            out << "return("; expr(out, *n.value); out << ')';
        }
    }, s);
}
}
std::string dump(const Module& module) {
    std::ostringstream out;
    out << "module(";
    for (std::size_t i = 0; i < module.items.size(); ++i) {
        if (i) out << ',';
        std::visit([&](const auto& n) {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, ImportDecl>) out << "import(" << n.path << " as " << n.alias << ')';
            else if constexpr (std::is_same_v<T, LetStmt>) {
                out << "let(" << (n.mutableBinding ? "mut," : "imm,") << n.name << ',';
                expr(out, *n.value); out << ')';
            }
            else {
                out << "fn(" << (n.exported ? "export," : "private,") << n.name << ",params(";
                for (std::size_t j = 0; j < n.parameters.size(); ++j) {
                    if (j) out << ',';
                    out << n.parameters[j].name << ':'; type(out, n.parameters[j].type);
                }
                out << "),result(";
                if (n.resultType) type(out, *n.resultType); else out << '_';
                out << "),body(";
                for (std::size_t j = 0; j < n.body.size(); ++j) {
                    if (j) out << ',';
                    stmt(out, n.body[j]);
                }
                out << "))";
            }
        }, module.items[i]);
    }
    out << ')';
    return out.str();
}
}
