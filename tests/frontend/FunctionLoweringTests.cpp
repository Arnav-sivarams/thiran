#include <filesystem>
#include <iostream>
#include <string>

#include "frontend/ModuleLinker.hpp"
#include "optimizer/GraphVerifier.hpp"
#include "region/RegionPlan.hpp"

namespace
{
int failures = 0;
#define CHECK(value) do { if(!(value)) { ++failures; std::cerr << "Failure at line " << __LINE__ << "\n"; } } while(false)
std::string signature(const thiran::Graph& graph)
{
    std::string value;
    for(const auto& node : graph.nodes) { value += node->name + ":" + thiran::toString(node->op) + "("; for(const auto* input : node->inputs) value += input->name + ","; value += ")\n"; }
    return value;
}
}

int main()
{
    namespace fs = std::filesystem; using thiran::frontend::ModuleLinker;
    const fs::path root = THIRAN_SOURCE_DIR;
    auto local = ModuleLinker::compile(root / "tests/fixtures/functions/local.th"); CHECK(local.succeeded());
    CHECK(local.graph->nodes.size() == 4); CHECK(local.graph->nodes[1]->name.find("__thiran_call_0_0_0_first") == 0); CHECK(local.graph->nodes[2]->name == "Y");
    CHECK(local.provenance.count("Y") == 1); CHECK(local.provenance.at("Y").inlineChain.size() == 1);
    CHECK(thiran::GraphVerifier{}.verify(*local.graph));
    auto identity = ModuleLinker::compile(root / "tests/fixtures/functions/identity.th"); CHECK(identity.succeeded());
    CHECK(identity.graph->nodes.size() == 2); CHECK(identity.graph->nodes[1]->inputs[0] == identity.graph->nodes[0].get()); CHECK(identity.graph->findNode("Y") == nullptr);
    auto nested = ModuleLinker::compile(root / "tests/fixtures/functions/nested.th"); CHECK(nested.succeeded());
    CHECK(nested.provenance.at("Y").inlineChain.size() == 2); CHECK(thiran::GraphVerifier{}.verify(*nested.graph));
    auto cross = ModuleLinker::compile(root / "tests/fixtures/functions/cross_module.th"); CHECK(cross.succeeded()); CHECK(cross.graph->findNode("Y") != nullptr);
    CHECK(thiran::region::HybridRegionPlanner::build(*cross.graph).succeeded());
    const std::string expected = signature(*cross.graph);
    for(int i = 0; i < 50; ++i)
    {
        auto repeat = ModuleLinker::compile(root / "tests/fixtures/functions/cross_module.th");
        CHECK(repeat.succeeded()); CHECK(signature(*repeat.graph) == expected); CHECK(repeat.provenance.size() == cross.provenance.size());
    }
    auto legacy = ModuleLinker::compile(root / "examples/cnn.th"); CHECK(legacy.succeeded()); CHECK(legacy.graph->nodes.front()->name == "Image");
    if(failures) return 1;
    std::cout << "All Function Lowering tests passed\n";
    return 0;
}
