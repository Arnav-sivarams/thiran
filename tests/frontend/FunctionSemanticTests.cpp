#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "frontend/ModuleLinker.hpp"

namespace
{
int failures = 0;
#define CHECK(value) do { if(!(value)) { ++failures; std::cerr << "Failure at line " << __LINE__ << "\n"; } } while(false)
void write(const std::filesystem::path& path, const std::string& source)
{ std::filesystem::create_directories(path.parent_path()); std::ofstream output(path); output << source; }
bool has(const thiran::frontend::FrontendResult& result, const std::string& text)
{ for(const auto& diagnostic : result.diagnostics) if(diagnostic.find(text) != std::string::npos) return true; return false; }
}

int main()
{
    namespace fs = std::filesystem; using thiran::frontend::ModuleLinker;
    const fs::path root = fs::temp_directory_path() / "thiran-function-semantic-tests"; fs::remove_all(root);
    const auto check = [&](const std::string& name, const std::string& source, const std::string& expected)
    { write(root / name, source); const auto result = ModuleLinker::compile(root / name); if(!has(result, expected)) { std::cerr << "Missing '" << expected << "' in " << name << "\n"; for(const auto& item : result.diagnostics) std::cerr << item << "\n"; } CHECK(has(result, expected)); };

    write(root / "valid.th", "fn zero() {\n c = Constant(1)\n return c\n}\nfn forward(x) {\n y = later(x)\n return y\n}\nfn later(x) {\n y = ReLU(x)\n return y\n}\nX = Input()\nA = zero()\nY = forward(X)\nO = Output(Y)\n");
    CHECK(ModuleLinker::compile(root / "valid.th").succeeded());
    write(root / "lib.th", "export fn public(x) {\n y = ReLU(x)\n return y\n}\nfn private(x) {\n y = ReLU(x)\n return y\n}\n");
    write(root / "import.th", "import \"lib.th\" as lib\nX = Input()\nY = lib.public(X)\nO = Output(Y)\n");
    CHECK(ModuleLinker::compile(root / "import.th").succeeded());
    check("private.th", "import \"lib.th\" as lib\nX=Input()\nY=lib.private(X)\n", "is not exported");
    check("arity.th", "fn f(x,y) {\n z=Add(x,y)\n return z\n}\nX=Input()\nY=f(X)\n", "expects 2 arguments but received 1");
    check("missing.th", "fn f(x) {\n y=ReLU(x)\n}\nX=Input()\n", "requires exactly one return");
    check("duplicate-param.th", "fn f(x,x) {\n return x\n}\nX=Input()\n", "duplicate parameter");
    check("duplicate-fn.th", "fn f(x) {\n return x\n}\nfn f(x) {\n return x\n}\nX=Input()\n", "duplicate function");
    check("builtin.th", "fn ReLU(x) {\n return x\n}\nX=Input()\n", "collides with built-in operation");
    check("alias.th", "import \"lib.th\" as f\nfn f(x) {\n return x\n}\nX=Input()\n", "collides with import alias");
    check("value.th", "fn f(x) {\n return x\n}\nf=Input()\n", "duplicate declaration");
    check("shadow.th", "fn f(x) {\n x=ReLU(x)\n return x\n}\nX=Input()\n", "shadows parameter");
    check("before.th", "fn f(x) {\n a=ReLU(b)\n b=ReLU(x)\n return a\n}\nX=Input()\n", "used before its declaration");
    check("capture.th", "fn f(x) {\n y=Add(x,C)\n return y\n}\nC=Constant(1)\nX=Input()\n", "cannot capture module value");
    check("input.th", "fn f() {\n x=Input()\n return x\n}\nX=Input()\n", "Input is not allowed inside function");
    check("output.th", "fn f(x) {\n y=Output(x)\n return y\n}\nX=Input()\n", "Output is not allowed inside function");
    check("return-expression.th", "fn f(x) {\n return ReLU(x)\n}\nX=Input()\n", "return expressions are not allowed");
    check("outside.th", "return X\n", "return is only allowed inside a function");
    check("multiple.th", "fn f(x) {\n return x\n return x\n}\nX=Input()\n", "multiple return");
    check("direct.th", "fn f(x) {\n y=f(x)\n return y\n}\nX=Input()\n", "recursive function cycle");
    check("mutual.th", "fn a(x) {\n y=b(x)\n return y\n}\nfn b(x) {\n y=a(x)\n return y\n}\nX=Input()\n", "mutual.th.a -> mutual.th.b -> mutual.th.a");
    check("long.th", "fn a(x) {\n y=b(x)\n return y\n}\nfn b(x) {\n y=c(x)\n return y\n}\nfn c(x) {\n y=a(x)\n return y\n}\nX=Input()\n", "long.th.a -> long.th.b -> long.th.c -> long.th.a");
    check("reserved.th", "fn f(__thiran_call_x) {\n return __thiran_call_x\n}\nX=Input()\n", "reserved internal prefix");
    write(root / "context.th", "fn = Input()\nreturn = ReLU(fn)\nO = Output(return)\n"); { const auto context = ModuleLinker::compile(root / "context.th"); if(!context.succeeded()) for(const auto& item : context.diagnostics) std::cerr << item << "\n"; CHECK(context.succeeded()); }
    fs::remove_all(root);
    if(failures) return 1;
    std::cout << "All Function Semantic tests passed\n";
    return 0;
}
