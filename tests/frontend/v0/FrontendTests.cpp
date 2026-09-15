#include "frontend/v0/Parser.hpp"
#include "frontend/v0/Lexer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace thiran::v0;

namespace {
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void span(const SourceSpan& s) {
    check(s.begin.source == 0 && s.end.source == 0 &&
          s.begin.offset <= s.end.offset && s.begin.line >= 1 &&
          s.begin.column >= 1 && s.end.line >= 1 && s.end.column >= 1,
          "invalid AST source span");
}
void expressionSpans(const Expr& e) {
    span(e.span);
    std::visit([&](const auto& n) {
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, TensorLiteralExpr>) {
            for (const auto& row : n.rows) for (const auto& element : row) expressionSpans(*element);
        } else if constexpr (std::is_same_v<T, TupleExpr>) {
            for (const auto& element : n.elements) expressionSpans(*element);
        } else if constexpr (std::is_same_v<T, UnaryExpr>) expressionSpans(*n.operand);
        else if constexpr (std::is_same_v<T, BinaryExpr>) {
            expressionSpans(*n.left); expressionSpans(*n.right);
        } else if constexpr (std::is_same_v<T, CallExpr>) {
            expressionSpans(*n.callee);
            for (const auto& arg : n.arguments) expressionSpans(*arg);
        } else if constexpr (std::is_same_v<T, MemberExpr>) expressionSpans(*n.object);
        else if constexpr (std::is_same_v<T, IndexExpr>) {
            expressionSpans(*n.object);
            for (const auto& axis : n.axes) std::visit([&](const auto& selector) {
                span(selector.span);
                using S = std::decay_t<decltype(selector)>;
                if constexpr (std::is_same_v<S, IndexSelector>) expressionSpans(*selector.value);
                else {
                    if (selector.start) expressionSpans(*selector.start);
                    if (selector.end) expressionSpans(*selector.end);
                    if (selector.step) expressionSpans(*selector.step);
                }
            }, axis);
        }
    }, e.node);
}
void typeSpans(const TypeSyntax& t) {
    span(t.span);
    for (const auto& child : t.elements) typeSpans(child);
}
void statementSpans(const Statement& s) {
    std::visit([&](const auto& n) { span(n.span); expressionSpans(*n.value); }, s);
}
void moduleSpans(const Module& m) {
    span(m.span);
    for (const auto& item : m.items) std::visit([&](const auto& n) {
        span(n.span);
        using T = std::decay_t<decltype(n)>;
        if constexpr (std::is_same_v<T, LetStmt>) expressionSpans(*n.value);
        else if constexpr (std::is_same_v<T, FunctionDecl>) {
            for (const auto& p : n.parameters) { span(p.span); typeSpans(p.type); }
            if (n.resultType) typeSpans(*n.resultType);
            for (const auto& stmt : n.body) statementSpans(stmt);
        }
    }, item);
}
std::string valid(const std::string& id, const std::string& source) {
    auto result = parse(source, id + ".th");
    if (!result.diagnostics.empty()) throw std::runtime_error(id + ": " + result.diagnostics[0].format());
    check(result.module.has_value(), id + ": missing module");
    moduleSpans(*result.module);
    return dump(*result.module);
}
void invalid(const std::string& id, const std::string& source, const std::string& fragment) {
    auto result = parse(source, id + ".th");
    check(!result.module.has_value(), id + ": unexpectedly parsed");
    check(!result.diagnostics.empty(), id + ": missing diagnostic");
    auto diagnostic = result.diagnostics[0];
    check(diagnostic.source == id + ".th", id + ": source identity lost");
    span(diagnostic.span);
    check(diagnostic.format().find(id + ".th:") == 0, id + ": unlocated diagnostic");
    check(diagnostic.message.find(fragment) != std::string::npos, id + ": wrong diagnostic: " + diagnostic.format());
}
void lexerTests() {
    auto result = lex("A .* B ./ C . T - -> // ignored\nlet x = 12", "lex.th");
    check(result.diagnostics.empty(), "lexical diagnostic on valid input");
    const std::vector<TokenKind> expected = {
        TokenKind::Identifier, TokenKind::DotStar, TokenKind::Identifier,
        TokenKind::DotSlash, TokenKind::Identifier, TokenKind::Dot,
        TokenKind::Identifier, TokenKind::Minus, TokenKind::Arrow,
        TokenKind::Newline, TokenKind::Let, TokenKind::Identifier,
        TokenKind::Equal, TokenKind::Integer, TokenKind::End
    };
    check(result.tokens.size() == expected.size(), "wrong token count");
    for (std::size_t i = 0; i < expected.size(); ++i)
        check(result.tokens[i].kind == expected[i], "wrong maximal-munch token");
    check(result.tokens[1].text == ".*" && result.tokens[3].text == "./", "operator spelling lost");
    check(result.tokens[1].span.begin.line == 1 && result.tokens[1].span.begin.column == 3 &&
          result.tokens[1].span.end.column == 5, "operator span wrong");
    check(result.tokens[10].span.begin.line == 2 && result.tokens[10].span.begin.column == 1,
          "newline span tracking wrong");
    check(lex("import \"unterminated", "lex.th").diagnostics[0].message.find("unterminated string") != std::string::npos,
          "unterminated string not diagnosed");
    auto reserved = lex("if else for in while break continue struct true false", "lex.th");
    check(reserved.tokens[0].kind == TokenKind::If && reserved.tokens[7].kind == TokenKind::Struct &&
          reserved.tokens[8].kind == TokenKind::True, "reserved recognition wrong");
}
void validTests() {
    check(valid("V01", "let A = [1, 2; 3, 4]\nlet B = A\nlet C = A + B\n") ==
        "module(let(imm,A,tensor([int(1),int(2)];[int(3),int(4)])),let(imm,B,id(A)),let(imm,C,binary(+,id(A),id(B))))",
        "V01 dump/rows wrong");
    check(valid("V02", "let x = A[0, 1]") ==
        "module(let(imm,x,index(id(A),at(int(0)),at(int(1)))))", "V02 indexing wrong");
    check(valid("V03", "let row = A[0, :]\nlet block = A[0:2, 1:3]\nlet step = A[0:4:2, :]\n") ==
        "module(let(imm,row,index(id(A),at(int(0)),slice(_,_,_))),let(imm,block,index(id(A),slice(int(0),int(2),_),slice(int(1),int(3),_))),let(imm,step,index(id(A),slice(int(0),int(4),int(2)),slice(_,_,_))))",
        "V03 slice structure wrong");
    check(valid("V04", "let y = A + B * C .* D") ==
        "module(let(imm,y,binary(+,id(A),binary(.*,binary(*,id(B),id(C)),id(D)))))",
        "V04 precedence wrong");
    check(valid("V05", "fn add(\n a: Tensor<i64, 2>,\n b: Tensor<i64, 2>\n) -> Tensor<i64, 2> {\n return a + b\n}\n") ==
        "module(fn(private,add,params(a:Tensor<i64,2>,b:Tensor<i64,2>),result(Tensor<i64,2>),body(return(binary(+,id(a),id(b))))))",
        "V05 function wrong");
    check(valid("V06", "export fn identity(x: Tensor<i64, 2>) -> Tensor<i64, 2> {\n return x\n}") ==
        "module(fn(export,identity,params(x:Tensor<i64,2>),result(Tensor<i64,2>),body(return(id(x)))))",
        "V06 export wrong");
    check(valid("V07", "fn twice(x: Tensor<i64, 2>) -> Tensor<i64, 2> {\n return x + x\n}") ==
        "module(fn(private,twice,params(x:Tensor<i64,2>),result(Tensor<i64,2>),body(return(binary(+,id(x),id(x))))))",
        "V07 direct return wrong");
    check(valid("V08", "import \"layers.th\" as layers") ==
        "module(import(layers.th as layers))", "V08 import wrong");
    check(valid("V09", "let x = (A +\n B * C)\nlet M = [1,\n 2;\n 3, 4]\n") ==
        "module(let(imm,x,binary(+,id(A),binary(*,id(B),id(C)))),let(imm,M,tensor([int(1),int(2)];[int(3),int(4)])))",
        "V09 multiline wrong");
    check(valid("V10", "let pair = (x, state)") ==
        "module(let(imm,pair,tuple(id(x),id(state))))", "V10 tuple wrong");
    check(valid("EXTRA", "fn step(s: State) -> (Tensor<f32, 3>, State) { let mut x = 1; x = -f(A.T)[0:4:-1]; return (x, s) }") ==
        "module(fn(private,step,params(s:State),result((Tensor<f32,3>,State)),body(let(mut,x,int(1)),rebind(x,unary(-,index(call(id(f),member(id(A),T)),slice(int(0),int(4),unary(-,int(1)))))),return(tuple(id(x),id(s))))))",
        "extra type/tuple dump wrong");
    check(valid("BUFFER", "fn bytes(x: Buffer<u8>) -> Buffer<u8> { return x }").find("Buffer<u8>") != std::string::npos,
          "Buffer generic type missing");
    auto first = valid("DET", "let x = A * B .* C");
    check(first == valid("DET", "let x = A * B .* C"), "dump nondeterministic");
    check(first.find("binary(.*,binary(*,id(A),id(B)),id(C))") != std::string::npos,
          "left associativity wrong");
    check(valid("SEMI", "let A = [1, 2; 3, 4]; let x = A[0]").find("tensor([int(1),int(2)];[int(3),int(4)])") != std::string::npos,
          "semicolon context wrong");
}
void invalidTests() {
    invalid("I01", "let A = [1, 2; 3]", "ragged");
    invalid("I02", "let A = [1, 2", "missing ']' delimiter");
    invalid("I03", "let x = A[0:2:3:4]", "too many colons");
    invalid("I04", "fn f(x Tensor<i64, 2>) -> i64 { return x }", "function parameter");
    invalid("I05", "fn f(x: i64) -> i64 { return x", "missing '}' delimiter");
    invalid("I06", "fn f(x: Tensor<i64, 2) -> i64 { return x }", "generic delimiter");
    invalid("I07", "fn f(x: i64) -> i64 { return + }", "expected expression");
    invalid("I08", "let x = A .*/ B", "expected expression");
    invalid("I09", "let x =", "expected expression");
    invalid("I10", "let A = [1, 2;; 3, 4]", "missing row");
    invalid("STRING", "import \"broken", "unterminated string");
    invalid("POSTFIX", "let x = A.", "member name");
    invalid("CONTROL", "if true { }", "unexpected token");
}
void robustnessTests() {
    const std::vector<std::string> inputs = {
        "", "// comment\n\n", "let x = (((", "let x = []", "let x = [;]",
        "let x = [1,,2]", "let x = A[,:]", "let x = A[:::]",
        "let x = A[0:4:]", "let x = A[0:4:-1]", "let x = A[(1+2), :]",
        "let x = A..T", "let x = A./", "let x = A->B", "let x = A/./B",
        "fn f(x: Tensor<>) {", "fn f(,) -> i64 {}", "fn f(x: i64) -> i64 {{{",
        "let x = (a,b,)", "let x = [1;2;3]", "let x = {1,2}",
        "let x = - - 1", "let x = A[0:4:2:3]", "let x = (a +\n b)"
    };
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto result = parse(inputs[i], "robust.th");
        check(result.module.has_value() || !result.diagnostics.empty(), "robust input yielded neither AST nor diagnostic " + std::to_string(i));
        for (auto& diagnostic : result.diagnostics)
            check(diagnostic.span.begin.line >= 1 && diagnostic.span.begin.column >= 1,
                  "robust input yielded unlocated diagnostic");
    }
}
}

int main() {
    try {
        lexerTests(); validTests(); invalidTests(); robustnessTests();
        std::cout << "PASS V0 lexer/parser fixtures (10 valid, 10 invalid, 24 bounded robustness + extensions)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL V0 frontend: " << error.what() << '\n';
        return 1;
    }
}
