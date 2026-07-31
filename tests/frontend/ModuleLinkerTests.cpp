#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "frontend/ModuleLinker.hpp"

namespace
{
int failures = 0;
#define CHECK(value) do { if(!(value)) { ++failures; std::cerr << "Failure at line " << __LINE__ << "\n"; } } while(false)
void write(const std::filesystem::path& path, const std::string& value)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << value;
}
bool has(const thiran::frontend::FrontendResult& result, const std::string& text)
{
    for(const auto& item : result.diagnostics) if(item.find(text) != std::string::npos) return true;
    return false;
}
}

int main()
{
    namespace fs = std::filesystem;
    using thiran::frontend::ModuleLinker;
    const fs::path source = THIRAN_SOURCE_DIR;
    auto legacy = ModuleLinker::compile(source / "examples/cnn.th");
    CHECK(legacy.succeeded() && legacy.program->modules.size() == 1);
    CHECK(legacy.graph->nodes.size() == 8);
    CHECK(legacy.graph->nodes.front()->name == "Image" && legacy.graph->nodes.back()->name == "O");
    CHECK(legacy.graph->nodes[2]->op == thiran::Operation::Conv2D);
    CHECK(legacy.graph->nodes[2]->inputs.size() == 2 && legacy.graph->nodes[2]->inputs[0]->name == "Image" && legacy.graph->nodes[2]->inputs[1]->name == "Filter");
    auto linked = ModuleLinker::compile(source / "tests/fixtures/modules/basic/main.th");
    CHECK(linked.succeeded());
    CHECK(linked.program->modules.size() == 2);
    CHECK(linked.graph->nodes.size() == 7);
    CHECK(linked.graph->nodes[0]->name == "__thiran_module_1_0_W");
    CHECK(linked.graph->nodes[3]->name == "X");
    CHECK(linked.graph->findNode("weights.W") == nullptr);
    for(int i = 0; i < 50; ++i)
    {
        auto repeat = ModuleLinker::compile(source / "tests/fixtures/modules/basic/main.th");
        CHECK(repeat.succeeded() && repeat.graph->nodes[0]->name == linked.graph->nodes[0]->name);
    }

    const auto root = fs::temp_directory_path() / "thiran-module-linker-tests";
    fs::remove_all(root);
    write(root / "leaf.th", "export C = Constant(1)\n");
    write(root / "left.th", "import \"leaf.th\" as leaf\nexport V = ReLU(leaf.C)\n");
    write(root / "right.th", "import \"leaf.th\" as leaf\nexport V = Sigmoid(leaf.C)\n");
    write(root / "diamond.th", "import \"left.th\" as left\nimport \"right.th\" as right\nX = Input()\nA = Add(X, left.V)\nB = Add(A, right.V)\nO = Output(B)\n");
    auto diamond = ModuleLinker::compile(root / "diamond.th");
    CHECK(diamond.succeeded() && diamond.program->modules.size() == 4);
    CHECK(diamond.graph->nodes.size() == 7); // the canonical leaf is lowered once
    CHECK(diamond.graph->nodes[0]->name.find("_C") != std::string::npos);
    CHECK(diamond.graph->nodes[1]->name != diamond.graph->nodes[2]->name);
    write(root / "self.th", "import \"self.th\" as self\nX = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "self.th"), "imports itself"));
    write(root / "a.th", "import \"b.th\" as b\nexport A = Constant(1)\n");
    write(root / "b.th", "import \"a.th\" as a\nexport B = Constant(1)\n");
    CHECK(has(ModuleLinker::compile(root / "a.th"), "import cycle"));
    auto cycle = ModuleLinker::compile(root / "a.th");
    CHECK(has(cycle, "imported from a.th:1:1") && has(cycle, "imported from b.th:1:1"));
    write(root / "private.th", "P = Constant(1)\n");
    write(root / "private-main.th", "import \"private.th\" as p\nX = Input()\nY = Add(X, p.P)\nO = Output(Y)\n");
    CHECK(has(ModuleLinker::compile(root / "private-main.th"), "not exported"));
    write(root / "input.th", "export X = Input()\n");
    write(root / "input-main.th", "import \"input.th\" as i\nX = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "input-main.th"), "Input is only allowed"));
    write(root / "output.th", "C = Constant(1)\nexport O = Output(C)\n");
    write(root / "output-main.th", "import \"output.th\" as o\nX = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "output-main.th"), "Output is only allowed"));
    write(root / "duplicate.th", "X = Input()\nX = ReLU(X)\n");
    CHECK(has(ModuleLinker::compile(root / "duplicate.th"), "duplicate declaration"));
    write(root / "reserved.th", "__thiran_module_bad = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "reserved.th"), "reserved internal prefix"));
    write(root / "late.th", "X = Input()\nimport \"private.th\" as p\n");
    CHECK(has(ModuleLinker::compile(root / "late.th"), "imports must precede"));
    write(root / "alias.th", "import \"private.th\" as p\nimport \"input.th\" as p\nX = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "alias.th"), "duplicate import alias"));
    write(root / "canonical-alias.th", "import \"private.th\" as p\nimport \"./private.th\" as q\nX = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "canonical-alias.th"), "already imported"));
    write(root / "reserved-alias.th", "import \"private.th\" as __thiran_module_bad\nX = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "reserved-alias.th"), "reserved internal prefix"));
    write(root / "missing-alias.th", "import \"private.th\"\nX = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "missing-alias.th"), "requires 'as"));
    write(root / "alias-collision.th", "import \"private.th\" as p\np = Input()\n");
    CHECK(has(ModuleLinker::compile(root / "alias-collision.th"), "collides with import alias"));
    write(root / "duplicate-export.th", "export X = Constant(1)\nexport X = Constant(2)\n");
    CHECK(has(ModuleLinker::compile(root / "duplicate-export.th"), "duplicate declaration"));
    write(root / "repeat.th", "import \"private.th\" as p\nimport \"./private.th\" as p\nX = Input()\n");
    CHECK(ModuleLinker::compile(root / "repeat.th").succeeded());
    write(root / "contextual.th", "as = Input()\nimport = ReLU(as)\nexport = Output(import)\n");
    CHECK(ModuleLinker::compile(root / "contextual.th").succeeded());
    write(root / "unresolved.th", "X = Input()\nY = ReLU(Missing)\nO = Output(Y)\n");
    CHECK(has(ModuleLinker::compile(root / "unresolved.th"), "unresolved value"));
    fs::remove_all(root);
    if(failures) return 1;
    std::cout << "All Module Linker tests passed\n";
    return 0;
}
