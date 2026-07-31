#include <algorithm>
#include <cstddef>
#include <deque>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ir/Graph.hpp"
#include "region/RegionFormation.hpp"
#include "region/RegionVerifier.hpp"
#include "utils/RegionPrinter.hpp"

namespace
{

using namespace thiran;
using namespace thiran::region;

int failures = 0;
std::string currentTest;

void fail(int line, const std::string& expected, const std::string& actual)
{
    ++failures;
    std::cerr << currentTest << ':' << line << ": expected " << expected
              << ", actual " << actual << '\n';
}

#define EXPECT_TRUE(value) do { if(!(value)) fail(__LINE__, #value, "false"); } while(false)
#define EXPECT_FALSE(value) do { if(value) fail(__LINE__, "false: " #value, "true"); } while(false)
#define EXPECT_EQ(expected, actual) do { if(!((expected) == (actual))) fail(__LINE__, #expected " == " #actual, "values differ"); } while(false)
#define EXPECT_PTR_EQ(expected, actual) do { if((expected) != (actual)) fail(__LINE__, #expected " == " #actual, "pointers differ"); } while(false)

std::size_t diagnosticCount(
    const RegionFormationResult& result,
    const std::string& code
)
{
    return static_cast<std::size_t>(std::count_if(
        result.diagnostics.begin(),
        result.diagnostics.end(),
        [&code](const auto& diagnostic) { return diagnostic.code == code; }
    ));
}

#define EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, code) EXPECT_TRUE(diagnosticCount((result), (code)) != 0)
#define EXPECT_DIAGNOSTIC_COUNT(result, code, count) EXPECT_EQ(std::size_t(count), diagnosticCount((result), (code)))

std::vector<NodeStrategyAssignment> assignAll(
    const Graph& graph,
    RegionStrategy strategy
)
{
    std::vector<NodeStrategyAssignment> result;
    for(const auto& node : graph.nodes)
    {
        result.push_back({node.get(), strategy});
    }
    return result;
}

std::string printed(const RegionGraph& graph)
{
    std::ostringstream output;
    RegionPrinter::print(graph, output);
    return output.str();
}

void expectWeaklyConnected(const Region& region)
{
    if(region.empty())
    {
        fail(__LINE__, "nonempty Region", "empty Region");
        return;
    }

    std::unordered_set<const Node*> members(
        region.nodes().begin(),
        region.nodes().end()
    );
    std::unordered_set<const Node*> visited;
    std::deque<const Node*> pending{region.nodes().front()};
    visited.insert(region.nodes().front());

    while(!pending.empty())
    {
        const Node* node = pending.front();
        pending.pop_front();
        const auto visit = [&](const std::vector<Node*>& neighbors)
        {
            for(const Node* neighbor : neighbors)
            {
                if(members.contains(neighbor) &&
                   visited.insert(neighbor).second)
                {
                    pending.push_back(neighbor);
                }
            }
        };
        visit(node->inputs);
        visit(node->outputs);
    }

    EXPECT_EQ(region.size(), visited.size());
}

void expectAllConnected(const RegionGraph& graph)
{
    for(const auto& region : graph.regions())
    {
        expectWeaklyConnected(*region);
    }
}

void expectRegion(
    const RegionGraph& graph,
    std::size_t id,
    RegionStrategy strategy,
    const std::vector<const Node*>& nodes
)
{
    const Region* region = graph.findRegion(static_cast<RegionId>(id));
    EXPECT_TRUE(region != nullptr);
    if(region != nullptr)
    {
        EXPECT_EQ(strategy, region->strategy());
        EXPECT_EQ(nodes, region->nodes());
    }
}

struct Diamond
{
    Graph graph;
    Node* input;
    Node* left;
    Node* right;
    Node* add;
    Node* output;

    Diamond()
    {
        input = graph.createNode("input", Operation::Input);
        left = graph.createNode("left", Operation::ReLU);
        right = graph.createNode("right", Operation::ReLU);
        add = graph.createNode("add", Operation::Add);
        output = graph.createNode("output", Operation::Output);
        graph.connect(input, left);
        graph.connect(input, right);
        graph.connect(left, add);
        graph.connect(right, add);
        graph.connect(add, output);
    }
};

void testResultSuccessContract()
{
    Graph graph;
    Node* node = graph.createNode("input", Operation::Input);
    auto good = TopologicalRegionFormer::form(
        graph, {{node, RegionStrategy::AheadOfTime}}
    );
    auto bad = TopologicalRegionFormer::form(graph, {});
    EXPECT_TRUE(good.succeeded());
    EXPECT_TRUE(good.regionGraph != nullptr);
    EXPECT_TRUE(good.diagnostics.empty());
    EXPECT_FALSE(bad.succeeded());
    EXPECT_PTR_EQ(nullptr, bad.regionGraph.get());
    EXPECT_FALSE(bad.diagnostics.empty());
}

void testEmptyGraph()
{
    Graph graph;
    auto result = TopologicalRegionFormer::form(graph, {});
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(result.regionGraph->finalized());
    EXPECT_EQ(std::size_t(0), result.regionGraph->size());
    EXPECT_EQ(std::size_t(0), result.regionGraph->dependencies().size());
    EXPECT_TRUE(RegionVerifier::verify(graph, *result.regionGraph).valid);
    EXPECT_TRUE(RegionVerifier::verify(
        graph, *result.regionGraph, RegionVerificationLevel::Executable
    ).valid);
    EXPECT_EQ(
        std::string("RegionGraph\n  Regions: 0\n  Dependencies: 0\n\n"
                    "Dependencies\n  <none>\n"),
        printed(*result.regionGraph)
    );
}

void testSingleNode()
{
    Graph graph;
    Node* node = graph.createNode("input", Operation::Input);
    auto result = TopologicalRegionFormer::form(
        graph, {{node, RegionStrategy::AheadOfTime}}
    );
    EXPECT_TRUE(result.succeeded());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {node});
    EXPECT_TRUE(result.regionGraph->findRegion(0)->inputs().empty());
    EXPECT_TRUE(result.regionGraph->findRegion(0)->outputs().empty());
    expectAllConnected(*result.regionGraph);
}

void testSameStrategyLinearChain()
{
    Graph graph;
    Node* a = graph.createNode("input", Operation::Input);
    Node* b = graph.createNode("relu", Operation::ReLU);
    Node* c = graph.createNode("output", Operation::Output);
    graph.connect(a, b); graph.connect(b, c);
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_TRUE(result.succeeded());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {a, b, c});
    EXPECT_EQ(std::vector<const Node*>({c}),
              result.regionGraph->findRegion(0)->outputs());
    expectAllConnected(*result.regionGraph);
}

void testAlternatingStrategyChain()
{
    Graph graph;
    Node* a = graph.createNode("A", Operation::Input);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* c = graph.createNode("C", Operation::ReLU);
    Node* d = graph.createNode("D", Operation::Output);
    graph.connect(a, b); graph.connect(b, c); graph.connect(c, d);
    auto result = TopologicalRegionFormer::form(graph, {
        {a, RegionStrategy::AheadOfTime},
        {b, RegionStrategy::JustInTime},
        {c, RegionStrategy::JustInTime},
        {d, RegionStrategy::Fallback}
    });
    EXPECT_TRUE(result.succeeded());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {a});
    expectRegion(*result.regionGraph, 1, RegionStrategy::JustInTime, {b, c});
    expectRegion(*result.regionGraph, 2, RegionStrategy::Fallback, {d});
    EXPECT_EQ(std::size_t(2), result.regionGraph->dependencies().size());
    expectAllConnected(*result.regionGraph);
}

void testReturnToEarlierStrategy()
{
    Graph graph;
    Node* a = graph.createNode("A", Operation::Input);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* c = graph.createNode("C", Operation::Output);
    graph.connect(a, b); graph.connect(b, c);
    auto result = TopologicalRegionFormer::form(graph, {
        {a, RegionStrategy::AheadOfTime},
        {b, RegionStrategy::JustInTime},
        {c, RegionStrategy::AheadOfTime}
    });
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(std::size_t(3), result.regionGraph->size());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {a});
    expectRegion(*result.regionGraph, 1, RegionStrategy::JustInTime, {b});
    expectRegion(*result.regionGraph, 2, RegionStrategy::AheadOfTime, {c});
    expectAllConnected(*result.regionGraph);
}

void testTwoDisconnectedSameStrategyNodes()
{
    Graph graph;
    Node* a = graph.createNode("inputA", Operation::Input);
    Node* b = graph.createNode("inputB", Operation::Input);
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_TRUE(result.succeeded());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {a});
    expectRegion(*result.regionGraph, 1, RegionStrategy::AheadOfTime, {b});
    EXPECT_TRUE(result.regionGraph->dependencies().empty());
    expectAllConnected(*result.regionGraph);
}

void testSameStrategyBranch()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* left = graph.createNode("left", Operation::ReLU);
    Node* right = graph.createNode("right", Operation::ReLU);
    graph.connect(input, left); graph.connect(input, right);
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::JustInTime)
    );
    EXPECT_TRUE(result.succeeded());
    expectRegion(
        *result.regionGraph, 0, RegionStrategy::JustInTime,
        {input, left, right}
    );
    expectAllConnected(*result.regionGraph);
}

void testSameStrategyJoin()
{
    Graph graph;
    Node* a = graph.createNode("inputA", Operation::Input);
    Node* b = graph.createNode("inputB", Operation::Input);
    Node* add = graph.createNode("add", Operation::Add);
    graph.connect(a, add); graph.connect(b, add);
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_TRUE(result.succeeded());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {a});
    expectRegion(*result.regionGraph, 1, RegionStrategy::AheadOfTime, {b, add});
    EXPECT_EQ(std::vector<const Node*>({a}),
              result.regionGraph->dependencies()[0].values);
    expectAllConnected(*result.regionGraph);
}

void testDiamondSameStrategy()
{
    Diamond d;
    auto result = TopologicalRegionFormer::form(
        d.graph, assignAll(d.graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_TRUE(result.succeeded());
    expectRegion(
        *result.regionGraph, 0, RegionStrategy::AheadOfTime,
        {d.input, d.left, d.right, d.add, d.output}
    );
    expectAllConnected(*result.regionGraph);
}

void testDiamondMixedStrategies()
{
    Diamond d;
    auto result = TopologicalRegionFormer::form(d.graph, {
        {d.input, RegionStrategy::AheadOfTime},
        {d.left, RegionStrategy::JustInTime},
        {d.right, RegionStrategy::JustInTime},
        {d.add, RegionStrategy::AheadOfTime},
        {d.output, RegionStrategy::AheadOfTime}
    });
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(std::size_t(4), result.regionGraph->size());
    EXPECT_EQ(std::size_t(4), result.regionGraph->dependencies().size());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {d.input});
    expectRegion(*result.regionGraph, 1, RegionStrategy::JustInTime, {d.left});
    expectRegion(*result.regionGraph, 2, RegionStrategy::JustInTime, {d.right});
    expectRegion(*result.regionGraph, 3, RegionStrategy::AheadOfTime,
                 {d.add, d.output});
    EXPECT_EQ(std::vector<const Node*>({d.input}),
              result.regionGraph->findRegion(1)->inputs());
    EXPECT_EQ(std::vector<const Node*>({d.input}),
              result.regionGraph->findRegion(2)->inputs());
    EXPECT_EQ(std::vector<const Node*>({d.left, d.right}),
              result.regionGraph->findRegion(3)->inputs());
    EXPECT_EQ(std::vector<const Node*>({d.output}),
              result.regionGraph->findRegion(3)->outputs());
    const auto& dependencies = result.regionGraph->dependencies();
    EXPECT_EQ(RegionId(0), dependencies[0].source);
    EXPECT_EQ(RegionId(1), dependencies[0].destination);
    EXPECT_EQ(RegionId(0), dependencies[1].source);
    EXPECT_EQ(RegionId(2), dependencies[1].destination);
    EXPECT_EQ(RegionId(1), dependencies[2].source);
    EXPECT_EQ(RegionId(3), dependencies[2].destination);
    EXPECT_EQ(RegionId(2), dependencies[3].source);
    EXPECT_EQ(RegionId(3), dependencies[3].destination);
    EXPECT_TRUE(RegionVerifier::verify(d.graph, *result.regionGraph).valid);
    EXPECT_TRUE(RegionVerifier::verify(
        d.graph, *result.regionGraph, RegionVerificationLevel::Executable
    ).valid);
    EXPECT_TRUE(printed(*result.regionGraph).find("Regions: 4") !=
                std::string::npos);
    expectAllConnected(*result.regionGraph);
}

void testStorageOrderIsNotTopological()
{
    Graph graph;
    Node* c = graph.createNode("C", Operation::Output);
    Node* a = graph.createNode("A", Operation::Input);
    Node* b = graph.createNode("B", Operation::ReLU);
    graph.connect(a, b); graph.connect(b, c);
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::JustInTime)
    );
    EXPECT_TRUE(result.succeeded());
    expectRegion(*result.regionGraph, 0, RegionStrategy::JustInTime, {a, b, c});
    expectAllConnected(*result.regionGraph);
}

void testReadyNodeStorageTieBreak()
{
    Graph graph;
    Node* second = graph.createNode("secondReady", Operation::Input);
    Node* first = graph.createNode("firstReady", Operation::Input);
    auto result = TopologicalRegionFormer::form(graph, {
        {first, RegionStrategy::Fallback},
        {second, RegionStrategy::AheadOfTime}
    });
    EXPECT_TRUE(result.succeeded());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {second});
    expectRegion(*result.regionGraph, 1, RegionStrategy::Fallback, {first});
    expectAllConnected(*result.regionGraph);
}

void testAssignmentOrderIndependence()
{
    Diamond d;
    std::vector<NodeStrategyAssignment> values{
        {d.input, RegionStrategy::AheadOfTime},
        {d.left, RegionStrategy::JustInTime},
        {d.right, RegionStrategy::JustInTime},
        {d.add, RegionStrategy::AheadOfTime},
        {d.output, RegionStrategy::Fallback}
    };
    auto first = TopologicalRegionFormer::form(d.graph, values);
    std::reverse(values.begin(), values.end());
    auto reverse = TopologicalRegionFormer::form(d.graph, values);
    std::rotate(values.begin(), values.begin() + 2, values.end());
    auto shuffled = TopologicalRegionFormer::form(d.graph, values);
    EXPECT_TRUE(first.succeeded() && reverse.succeeded() && shuffled.succeeded());
    EXPECT_EQ(printed(*first.regionGraph), printed(*reverse.regionGraph));
    EXPECT_EQ(printed(*first.regionGraph), printed(*shuffled.regionGraph));
    expectAllConnected(*first.regionGraph);
    expectAllConnected(*reverse.regionGraph);
    expectAllConnected(*shuffled.regionGraph);
}

void testRepeatability()
{
    std::string baseline;
    for(int iteration = 0; iteration < 50; ++iteration)
    {
        Diamond d;
        auto result = TopologicalRegionFormer::form(d.graph, {
            {d.right, RegionStrategy::JustInTime},
            {d.output, RegionStrategy::Fallback},
            {d.input, RegionStrategy::AheadOfTime},
            {d.add, RegionStrategy::AheadOfTime},
            {d.left, RegionStrategy::JustInTime}
        });
        EXPECT_TRUE(result.succeeded());
        EXPECT_EQ(std::size_t(5), result.regionGraph->size());
        EXPECT_EQ(std::size_t(5), result.regionGraph->dependencies().size());
        const std::string output = printed(*result.regionGraph);
        if(iteration == 0) baseline = output;
        EXPECT_EQ(baseline, output);
        for(std::size_t id = 0; id < result.regionGraph->size(); ++id)
        {
            EXPECT_EQ(static_cast<RegionId>(id),
                      result.regionGraph->findRegion(
                          static_cast<RegionId>(id))->id());
        }
        expectAllConnected(*result.regionGraph);
    }
}

void testNullAssignment()
{
    Graph graph;
    Node* node = graph.createNode("input", Operation::Input);
    auto result = TopologicalRegionFormer::form(graph, {
        {node, RegionStrategy::AheadOfTime},
        {nullptr, RegionStrategy::Fallback}
    });
    EXPECT_DIAGNOSTIC_COUNT(result, "RF003", 1);
    EXPECT_EQ(std::optional<std::size_t>(1),
              result.diagnostics[0].assignmentIndex);
    EXPECT_PTR_EQ(nullptr, result.regionGraph.get());
}

void testForeignAssignment()
{
    Graph graph;
    Graph other;
    Node* owned = graph.createNode("same", Operation::Input);
    Node* foreign = other.createNode("same", Operation::Input);
    EXPECT_EQ(owned->id, foreign->id);
    auto result = TopologicalRegionFormer::form(
        graph, {{foreign, RegionStrategy::AheadOfTime}}
    );
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RF004");
    const auto found = std::find_if(
        result.diagnostics.begin(), result.diagnostics.end(),
        [](const auto& diagnostic) { return diagnostic.code == "RF004"; }
    );
    EXPECT_FALSE(found->nodeName.has_value());
}

void testDuplicateAssignment()
{
    for(const RegionStrategy second :
        {RegionStrategy::AheadOfTime, RegionStrategy::Fallback})
    {
        Graph graph;
        Node* node = graph.createNode("input", Operation::Input);
        auto result = TopologicalRegionFormer::form(graph, {
            {node, RegionStrategy::AheadOfTime}, {node, second}
        });
        EXPECT_DIAGNOSTIC_COUNT(result, "RF005", 1);
        const auto found = std::find_if(
            result.diagnostics.begin(), result.diagnostics.end(),
            [](const auto& diagnostic) { return diagnostic.code == "RF005"; }
        );
        EXPECT_EQ(std::optional<std::size_t>(1), found->assignmentIndex);
    }
}

void testMissingAssignment()
{
    Graph graph;
    Node* a = graph.createNode("A", Operation::Input);
    graph.createNode("B", Operation::Input);
    graph.createNode("C", Operation::Input);
    auto result = TopologicalRegionFormer::form(
        graph, {{a, RegionStrategy::AheadOfTime}}
    );
    EXPECT_DIAGNOSTIC_COUNT(result, "RF006", 2);
    std::vector<std::string> names;
    for(const auto& diagnostic : result.diagnostics)
        if(diagnostic.code == "RF006") names.push_back(*diagnostic.nodeName);
    EXPECT_EQ(std::vector<std::string>({"B", "C"}), names);
}

void testUnresolvedStrategy()
{
    Graph graph;
    Node* node = graph.createNode("input", Operation::Input);
    auto result = TopologicalRegionFormer::form(
        graph, {{node, RegionStrategy::Unresolved}}
    );
    EXPECT_DIAGNOSTIC_COUNT(result, "RF007", 1);
    EXPECT_EQ(std::optional<std::string>("input"),
              result.diagnostics[0].nodeName);
    EXPECT_EQ(std::optional<std::size_t>(0),
              result.diagnostics[0].assignmentIndex);
}

void testCyclicGraph()
{
    Graph graph;
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* c = graph.createNode("C", Operation::ReLU);
    graph.connect(a, b); graph.connect(b, c); graph.connect(c, a);
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_DIAGNOSTIC_COUNT(result, "RF010", 1);
    EXPECT_EQ(std::size_t(1), result.diagnostics.size());
    EXPECT_PTR_EQ(nullptr, result.regionGraph.get());
}

void testMultipleValidStrategies()
{
    Graph graph;
    Node* a = graph.createNode("A", Operation::Input);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* c = graph.createNode("C", Operation::Output);
    graph.connect(a, b); graph.connect(b, c);
    auto result = TopologicalRegionFormer::form(graph, {
        {a, RegionStrategy::AheadOfTime},
        {b, RegionStrategy::JustInTime},
        {c, RegionStrategy::Fallback}
    });
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(RegionStrategy::AheadOfTime,
              result.regionGraph->findRegion(0)->strategy());
    EXPECT_EQ(RegionStrategy::JustInTime,
              result.regionGraph->findRegion(1)->strategy());
    EXPECT_EQ(RegionStrategy::Fallback,
              result.regionGraph->findRegion(2)->strategy());
    EXPECT_TRUE(RegionVerifier::verify(
        graph, *result.regionGraph, RegionVerificationLevel::Executable
    ).valid);
    expectAllConnected(*result.regionGraph);
}

void testNullInputAdjacencySafety()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* consumer = graph.createNode("consumer", Operation::ReLU);
    graph.connect(input, consumer);
    consumer->inputs.push_back(nullptr);
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_DIAGNOSTIC_COUNT(result, "RF008", 1);
    EXPECT_EQ(std::optional<std::string>("consumer"),
              result.diagnostics[0].nodeName);
    EXPECT_TRUE(result.diagnostics[0].message.find("operand index 1") !=
                std::string::npos);
}

void testForeignInputAdjacencySafety()
{
    Graph graph;
    Graph other;
    Node* input = graph.createNode("input", Operation::Input);
    Node* consumer = graph.createNode("consumer", Operation::ReLU);
    Node* foreign = other.createNode("secret", Operation::Input);
    graph.connect(input, consumer);
    consumer->inputs.push_back(foreign);
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_DIAGNOSTIC_COUNT(result, "RF009", 1);
    EXPECT_FALSE(result.diagnostics[0].message.find("secret") !=
                 std::string::npos);
}

void testAssignmentDiagnosticOrder()
{
    Graph graph;
    Graph other;
    Node* a = graph.createNode("A", Operation::Input);
    Node* b = graph.createNode("B", Operation::Input);
    graph.createNode("missing", Operation::Input);
    Node* foreign = other.createNode("foreign", Operation::Input);
    auto result = TopologicalRegionFormer::form(graph, {
        {nullptr, RegionStrategy::Fallback},
        {foreign, RegionStrategy::AheadOfTime},
        {a, RegionStrategy::AheadOfTime},
        {a, RegionStrategy::Fallback},
        {b, RegionStrategy::Unresolved}
    });
    EXPECT_EQ(std::size_t(5), result.diagnostics.size());
    const std::vector<std::string> expected{
        "RF003", "RF004", "RF005", "RF006", "RF007"
    };
    for(std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(expected[i], result.diagnostics[i].code);
}

void testFormationDoesNotMutateGraph()
{
    Diamond d;
    const std::size_t nodeCount = d.graph.nodes.size();
    const std::size_t edgeCount = d.graph.edges.size();
    std::vector<std::vector<Node*>> inputs;
    std::vector<std::vector<Node*>> outputs;
    std::vector<std::string> names;
    std::vector<Operation> operations;
    for(const auto& node : d.graph.nodes)
    {
        inputs.push_back(node->inputs);
        outputs.push_back(node->outputs);
        names.push_back(node->name);
        operations.push_back(node->op);
    }
    auto result = TopologicalRegionFormer::form(
        d.graph, assignAll(d.graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(nodeCount, d.graph.nodes.size());
    EXPECT_EQ(edgeCount, d.graph.edges.size());
    for(std::size_t i = 0; i < nodeCount; ++i)
    {
        EXPECT_EQ(inputs[i], d.graph.nodes[i]->inputs);
        EXPECT_EQ(outputs[i], d.graph.nodes[i]->outputs);
        EXPECT_EQ(names[i], d.graph.nodes[i]->name);
        EXPECT_EQ(operations[i], d.graph.nodes[i]->op);
    }
}

void testSharedExternalProducerDoesNotConnectSiblingRegions()
{
    Graph graph;
    Node* source = graph.createNode("source", Operation::Input);
    Node* first = graph.createNode("first", Operation::ReLU);
    Node* second = graph.createNode("second", Operation::ReLU);
    graph.connect(source, first); graph.connect(source, second);
    const std::vector<NodeStrategyAssignment> assignments{
        {source, RegionStrategy::AheadOfTime},
        {first, RegionStrategy::JustInTime},
        {second, RegionStrategy::JustInTime}
    };
    auto result = TopologicalRegionFormer::form(graph, assignments);
    auto repeated = TopologicalRegionFormer::form(graph, assignments);
    EXPECT_TRUE(result.succeeded() && repeated.succeeded());
    EXPECT_EQ(std::size_t(3), result.regionGraph->size());
    EXPECT_EQ(std::size_t(2), result.regionGraph->dependencies().size());
    expectRegion(*result.regionGraph, 0, RegionStrategy::AheadOfTime, {source});
    expectRegion(*result.regionGraph, 1, RegionStrategy::JustInTime, {first});
    expectRegion(*result.regionGraph, 2, RegionStrategy::JustInTime, {second});
    EXPECT_EQ(printed(*result.regionGraph), printed(*repeated.regionGraph));
    EXPECT_TRUE(RegionVerifier::verify(graph, *result.regionGraph).valid);
    EXPECT_TRUE(RegionVerifier::verify(
        graph, *result.regionGraph, RegionVerificationLevel::Executable
    ).valid);
    expectAllConnected(*result.regionGraph);
}

void testNullGraphHolder()
{
    Graph graph;
    graph.nodes.push_back(nullptr);
    auto result = TopologicalRegionFormer::form(graph, {});
    EXPECT_DIAGNOSTIC_COUNT(result, "RF001", 1);
}

void testDuplicateGraphHolder()
{
    Graph graph;
    graph.createNode("input", Operation::Input);
    graph.nodes.push_back(graph.nodes.front());
    auto result = TopologicalRegionFormer::form(graph, {});
    EXPECT_DIAGNOSTIC_COUNT(result, "RF002", 1);
    EXPECT_EQ(std::optional<std::string>("input"),
              result.diagnostics[0].nodeName);
}

void testGraphStorageDiagnosticOrder()
{
    Graph graph;
    graph.createNode("input", Operation::Input);
    graph.nodes.push_back(graph.nodes.front());
    graph.nodes.push_back(nullptr);
    auto result = TopologicalRegionFormer::form(graph, {});
    EXPECT_EQ(std::size_t(2), result.diagnostics.size());
    EXPECT_EQ(std::string("RF001"), result.diagnostics[0].code);
    EXPECT_EQ(std::string("RF002"), result.diagnostics[1].code);
}

void testEmptyGraphWithForeignAssignment()
{
    Graph graph;
    Graph other;
    Node* foreign = other.createNode("foreign", Operation::Input);
    auto result = TopologicalRegionFormer::form(
        graph, {{foreign, RegionStrategy::Fallback}}
    );
    EXPECT_DIAGNOSTIC_COUNT(result, "RF004", 1);
}

void testThousandNodeChain()
{
    Graph graph;
    std::vector<Node*> nodes;
    nodes.reserve(1000);
    for(int i = 0; i < 1000; ++i)
    {
        nodes.push_back(graph.createNode(
            "node" + std::to_string(i),
            i == 0 ? Operation::Input : Operation::ReLU
        ));
        if(i != 0) graph.connect(nodes[i - 1], nodes[i]);
    }
    auto result = TopologicalRegionFormer::form(
        graph, assignAll(graph, RegionStrategy::AheadOfTime)
    );
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(std::size_t(1), result.regionGraph->size());
    EXPECT_EQ(std::size_t(1000), result.regionGraph->findRegion(0)->size());
    EXPECT_EQ(std::vector<const Node*>(nodes.begin(), nodes.end()),
              result.regionGraph->findRegion(0)->nodes());
    EXPECT_TRUE(result.regionGraph->dependencies().empty());
    EXPECT_TRUE(RegionVerifier::verify(
        graph, *result.regionGraph, RegionVerificationLevel::Executable
    ).valid);
    expectAllConnected(*result.regionGraph);
}

template<typename Function>
void runTest(const std::string& name, Function function)
{
    currentTest = name;
    function();
}

}

int main()
{
    runTest("Result success contract", testResultSuccessContract);
    runTest("Empty graph", testEmptyGraph);
    runTest("Single node", testSingleNode);
    runTest("Same-strategy linear chain", testSameStrategyLinearChain);
    runTest("Alternating strategy chain", testAlternatingStrategyChain);
    runTest("Return to earlier strategy", testReturnToEarlierStrategy);
    runTest("Two disconnected same-strategy nodes", testTwoDisconnectedSameStrategyNodes);
    runTest("Same-strategy branch", testSameStrategyBranch);
    runTest("Same-strategy join", testSameStrategyJoin);
    runTest("Diamond same strategy", testDiamondSameStrategy);
    runTest("Diamond mixed strategies", testDiamondMixedStrategies);
    runTest("Storage order is not topological", testStorageOrderIsNotTopological);
    runTest("Ready-node storage tie-break", testReadyNodeStorageTieBreak);
    runTest("Assignment order independence", testAssignmentOrderIndependence);
    runTest("Repeatability", testRepeatability);
    runTest("Null assignment", testNullAssignment);
    runTest("Foreign assignment", testForeignAssignment);
    runTest("Duplicate assignment", testDuplicateAssignment);
    runTest("Missing assignment", testMissingAssignment);
    runTest("Unresolved strategy", testUnresolvedStrategy);
    runTest("Cyclic graph", testCyclicGraph);
    runTest("Multiple valid strategies", testMultipleValidStrategies);
    runTest("Null input adjacency safety", testNullInputAdjacencySafety);
    runTest("Foreign input adjacency safety", testForeignInputAdjacencySafety);
    runTest("Assignment diagnostic order", testAssignmentDiagnosticOrder);
    runTest("Formation does not mutate graph", testFormationDoesNotMutateGraph);
    runTest("Shared external producer does not connect sibling Regions", testSharedExternalProducerDoesNotConnectSiblingRegions);
    runTest("Null Graph holder", testNullGraphHolder);
    runTest("Duplicate Graph holder", testDuplicateGraphHolder);
    runTest("Graph storage diagnostic order", testGraphStorageDiagnosticOrder);
    runTest("Empty Graph with foreign assignment", testEmptyGraphWithForeignAssignment);
    runTest("Thousand Node chain", testThousandNodeChain);

    if(failures != 0)
    {
        std::cerr << failures << " Region Formation test failure(s)\n";
        return 1;
    }

    std::cout << "All Region Formation tests passed\n";
    return 0;
}
