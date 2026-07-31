#include <algorithm>
#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ir/Graph.hpp"
#include "region/RegionFormation.hpp"
#include "region/RegionVerifier.hpp"
#include "region/StrategyClassification.hpp"

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
    const StrategyClassificationResult& result,
    std::string_view code
)
{
    return static_cast<std::size_t>(std::count_if(
        result.diagnostics.begin(), result.diagnostics.end(),
        [code](const auto& diagnostic) { return diagnostic.code == code; }
    ));
}

#define EXPECT_DIAGNOSTIC_COUNT(result, code, count) EXPECT_EQ(std::size_t(count), diagnosticCount((result), (code)))
#define EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, code) EXPECT_TRUE(diagnosticCount((result), (code)) != 0)

void expectDecision(
    const StrategyClassificationResult& result,
    std::size_t index,
    const Node* node,
    RegionStrategy strategy,
    ShapeKnowledge knowledge,
    StrategyDecisionReason reason
)
{
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(index < result.decisions.size());
    if(index < result.decisions.size())
    {
        EXPECT_PTR_EQ(node, result.decisions[index].node);
        EXPECT_EQ(strategy, result.decisions[index].strategy);
        EXPECT_EQ(knowledge, result.decisions[index].shapeKnowledge);
        EXPECT_EQ(reason, result.decisions[index].reason);
    }
}

#define EXPECT_DECISION(result, index, node, strategy, knowledge, reason) \
    expectDecision((result), (index), (node), (strategy), (knowledge), (reason))

StrategyClassificationResult classifyOne(
    Operation operation,
    std::vector<int> shape
)
{
    Graph graph;
    Node* node = graph.createNode("node", operation);
    node->shape = std::move(shape);
    return BaselineStrategyClassifier::classify(graph);
}

void testEmptyGraphSuccess()
{
    Graph graph;
    auto result = BaselineStrategyClassifier::classify(graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(result.decisions.empty());
    EXPECT_TRUE(result.diagnostics.empty());
    EXPECT_TRUE(result.assignments().empty());
}

void testShapeKnowledgeStrings()
{
    EXPECT_EQ(std::string_view("STATIC"), toString(ShapeKnowledge::Static));
    EXPECT_EQ(std::string_view("RUNTIME_SPECIALIZABLE"),
              toString(ShapeKnowledge::RuntimeSpecializable));
    EXPECT_EQ(std::string_view("UNKNOWN_RANK"),
              toString(ShapeKnowledge::UnknownRank));
    EXPECT_EQ(std::string_view("UNKNOWN"),
              toString(static_cast<ShapeKnowledge>(255)));
}

void testReasonStrings()
{
    const std::vector<std::pair<StrategyDecisionReason, std::string_view>> values{
        {StrategyDecisionReason::InputBoundary, "INPUT_BOUNDARY"},
        {StrategyDecisionReason::OutputBoundary, "OUTPUT_BOUNDARY"},
        {StrategyDecisionReason::RuntimeTransfer, "RUNTIME_TRANSFER"},
        {StrategyDecisionReason::StaticConstant, "STATIC_CONSTANT"},
        {StrategyDecisionReason::StaticOptimizedOperation,
         "STATIC_OPTIMIZED_OPERATION"},
        {StrategyDecisionReason::DynamicOptimizedOperation,
         "DYNAMIC_OPTIMIZED_OPERATION"},
        {StrategyDecisionReason::UnknownRankFallback,
         "UNKNOWN_RANK_FALLBACK"},
        {StrategyDecisionReason::FallbackOnlyOperation,
         "FALLBACK_ONLY_OPERATION"}
    };
    for(const auto& [value, text] : values) EXPECT_EQ(text, toString(value));
    EXPECT_EQ(std::string_view("UNKNOWN"),
              toString(static_cast<StrategyDecisionReason>(255)));
}

void testInputStaticFallback()
{
    Graph graph; Node* n = graph.createNode("input", Operation::Input);
    n->shape = {1, 4}; auto r = BaselineStrategyClassifier::classify(graph);
    EXPECT_DECISION(r, 0, n, RegionStrategy::Fallback, ShapeKnowledge::Static,
                    StrategyDecisionReason::InputBoundary);
}

void testInputDynamicFallback()
{
    Graph graph; Node* n = graph.createNode("input", Operation::Input);
    n->shape = {-1, 4}; auto r = BaselineStrategyClassifier::classify(graph);
    EXPECT_DECISION(r, 0, n, RegionStrategy::Fallback,
                    ShapeKnowledge::RuntimeSpecializable,
                    StrategyDecisionReason::InputBoundary);
}

void testInputUnknownRankFallback()
{
    Graph graph; Node* n = graph.createNode("input", Operation::Input);
    auto r = BaselineStrategyClassifier::classify(graph);
    EXPECT_DECISION(r, 0, n, RegionStrategy::Fallback,
                    ShapeKnowledge::UnknownRank,
                    StrategyDecisionReason::InputBoundary);
}

void testOutputBoundaryFallback()
{
    Graph graph;
    Node* a = graph.createNode("a", Operation::Output); a->shape = {1};
    Node* b = graph.createNode("b", Operation::Output); b->shape = {-1};
    Node* c = graph.createNode("c", Operation::Output);
    auto r = BaselineStrategyClassifier::classify(graph);
    EXPECT_DECISION(r, 0, a, RegionStrategy::Fallback, ShapeKnowledge::Static,
                    StrategyDecisionReason::OutputBoundary);
    EXPECT_DECISION(r, 1, b, RegionStrategy::Fallback,
                    ShapeKnowledge::RuntimeSpecializable,
                    StrategyDecisionReason::OutputBoundary);
    EXPECT_DECISION(r, 2, c, RegionStrategy::Fallback,
                    ShapeKnowledge::UnknownRank,
                    StrategyDecisionReason::OutputBoundary);
}

void testTransferFallback()
{
    Graph graph;
    Node* a = graph.createNode("a", Operation::Transfer); a->shape = {1};
    Node* b = graph.createNode("b", Operation::Transfer); b->shape = {-1};
    auto r = BaselineStrategyClassifier::classify(graph);
    EXPECT_DECISION(r, 0, a, RegionStrategy::Fallback, ShapeKnowledge::Static,
                    StrategyDecisionReason::RuntimeTransfer);
    EXPECT_DECISION(r, 1, b, RegionStrategy::Fallback,
                    ShapeKnowledge::RuntimeSpecializable,
                    StrategyDecisionReason::RuntimeTransfer);
}

void testScalarConstantAot()
{
    Graph graph; Node* n = graph.createNode("c", Operation::Constant);
    auto r = BaselineStrategyClassifier::classify(graph);
    EXPECT_DECISION(r, 0, n, RegionStrategy::AheadOfTime,
                    ShapeKnowledge::Static,
                    StrategyDecisionReason::StaticConstant);
}

void testStaticTensorConstantAot()
{
    auto r = classifyOne(Operation::Constant, {2, 3});
    EXPECT_TRUE(r.succeeded());
    EXPECT_EQ(RegionStrategy::AheadOfTime, r.decisions[0].strategy);
}

void testDynamicConstantFailure()
{
    auto r = classifyOne(Operation::Constant, {-1, 3});
    EXPECT_FALSE(r.succeeded()); EXPECT_DIAGNOSTIC_COUNT(r, "SC005", 1);
    EXPECT_TRUE(r.decisions.empty()); EXPECT_TRUE(r.assignments().empty());
}

const std::vector<Operation> optimized{
    Operation::Add, Operation::Subtract, Operation::Multiply,
    Operation::Divide, Operation::MatMul, Operation::ReLU,
    Operation::Sigmoid, Operation::Tanh, Operation::Reshape,
    Operation::Transpose, Operation::FusedMatMulRelu
};
const std::vector<Operation> fallbackOnly{
    Operation::Conv2D, Operation::Softmax,
    Operation::MaxPool, Operation::AvgPool
};

void testOptimizedOperationStaticMatrix()
{
    for(Operation op : optimized)
    {
        auto r = classifyOne(op, {2, 4});
        EXPECT_TRUE(r.succeeded());
        EXPECT_EQ(RegionStrategy::AheadOfTime, r.decisions[0].strategy);
        EXPECT_EQ(StrategyDecisionReason::StaticOptimizedOperation,
                  r.decisions[0].reason);
    }
}

void testOptimizedOperationDynamicMatrix()
{
    for(Operation op : optimized)
    {
        auto r = classifyOne(op, {-1, 4});
        EXPECT_TRUE(r.succeeded());
        EXPECT_EQ(RegionStrategy::JustInTime, r.decisions[0].strategy);
        EXPECT_EQ(StrategyDecisionReason::DynamicOptimizedOperation,
                  r.decisions[0].reason);
    }
}

void testOptimizedOperationMultipleDynamicDimensions()
{
    for(Operation op : {Operation::MatMul, Operation::Add, Operation::Reshape})
        EXPECT_EQ(RegionStrategy::JustInTime,
                  classifyOne(op, {-1, -1}).decisions[0].strategy);
}

void testOptimizedOperationUnknownRank()
{
    for(Operation op : optimized)
    {
        auto r = classifyOne(op, {});
        EXPECT_EQ(RegionStrategy::Fallback, r.decisions[0].strategy);
        EXPECT_EQ(StrategyDecisionReason::UnknownRankFallback,
                  r.decisions[0].reason);
    }
}

void testFallbackOnlyStaticMatrix()
{
    for(Operation op : fallbackOnly)
    {
        auto r = classifyOne(op, {1, 4, 8, 8});
        EXPECT_EQ(RegionStrategy::Fallback, r.decisions[0].strategy);
        EXPECT_EQ(ShapeKnowledge::Static, r.decisions[0].shapeKnowledge);
    }
}

void testFallbackOnlyDynamicMatrix()
{
    for(Operation op : fallbackOnly)
    {
        auto r = classifyOne(op, {-1, 4, 8, 8});
        EXPECT_EQ(RegionStrategy::Fallback, r.decisions[0].strategy);
        EXPECT_EQ(ShapeKnowledge::RuntimeSpecializable,
                  r.decisions[0].shapeKnowledge);
    }
}

void testFallbackOnlyUnknownRank()
{
    for(Operation op : fallbackOnly)
    {
        auto r = classifyOne(op, {});
        EXPECT_EQ(RegionStrategy::Fallback, r.decisions[0].strategy);
        EXPECT_EQ(StrategyDecisionReason::FallbackOnlyOperation,
                  r.decisions[0].reason);
    }
}

void testZeroDimensionFailure()
{
    auto r = classifyOne(Operation::ReLU, {1, 0, 4});
    EXPECT_DIAGNOSTIC_COUNT(r, "SC004", 1);
    EXPECT_EQ(std::optional<std::size_t>(0), r.diagnostics[0].graphIndex);
    EXPECT_EQ(std::optional<std::size_t>(1), r.diagnostics[0].dimensionIndex);
    EXPECT_TRUE(r.diagnostics[0].message.find("0") != std::string::npos);
}

void testLessThanMinusOneFailure()
{
    auto r = classifyOne(Operation::ReLU, {1, -2, 4});
    EXPECT_DIAGNOSTIC_COUNT(r, "SC004", 1);
    EXPECT_EQ(std::optional<std::size_t>(1), r.diagnostics[0].dimensionIndex);
}

void testMultipleInvalidDimensions()
{
    Graph g; Node* a = g.createNode("a", Operation::ReLU);
    Node* b = g.createNode("b", Operation::Add);
    a->shape = {0, -2}; b->shape = {1, 0, -3};
    auto r = BaselineStrategyClassifier::classify(g);
    EXPECT_DIAGNOSTIC_COUNT(r, "SC004", 4);
    EXPECT_EQ(std::optional<std::size_t>(0), r.diagnostics[0].graphIndex);
    EXPECT_EQ(std::optional<std::size_t>(0), r.diagnostics[0].dimensionIndex);
    EXPECT_EQ(std::optional<std::size_t>(1), r.diagnostics[2].graphIndex);
    EXPECT_EQ(std::optional<std::size_t>(1), r.diagnostics[2].dimensionIndex);
}

void testUnknownOperationFailure()
{
    auto r = classifyOne(Operation::Unknown, {1});
    EXPECT_DIAGNOSTIC_COUNT(r, "SC003", 1);
    EXPECT_EQ(std::optional<std::string>("node"), r.diagnostics[0].nodeName);
    EXPECT_TRUE(r.decisions.empty());
}

void testUnknownOperationWithInvalidShapeOrder()
{
    auto r = classifyOne(Operation::Unknown, {0});
    EXPECT_EQ(std::size_t(2), r.diagnostics.size());
    EXPECT_EQ(std::string("SC003"), r.diagnostics[0].code);
    EXPECT_EQ(std::string("SC004"), r.diagnostics[1].code);
}

void testNullGraphHolder()
{
    Graph g; g.nodes.push_back(nullptr);
    auto r = BaselineStrategyClassifier::classify(g);
    EXPECT_DIAGNOSTIC_COUNT(r, "SC001", 1);
    EXPECT_EQ(std::optional<std::size_t>(0), r.diagnostics[0].graphIndex);
}

void testDuplicateGraphHolder()
{
    Graph g; g.createNode("same", Operation::Input);
    g.nodes.push_back(g.nodes.front());
    auto r = BaselineStrategyClassifier::classify(g);
    EXPECT_DIAGNOSTIC_COUNT(r, "SC002", 1);
    EXPECT_TRUE(r.diagnostics[0].message.find("0") != std::string::npos);
    EXPECT_TRUE(r.diagnostics[0].message.find("1") != std::string::npos);
}

void testStorageDiagnosticCategoryOrder()
{
    Graph g; g.createNode("same", Operation::Input);
    g.nodes.push_back(g.nodes.front()); g.nodes.push_back(nullptr);
    auto r = BaselineStrategyClassifier::classify(g);
    EXPECT_EQ(std::string("SC001"), r.diagnostics[0].code);
    EXPECT_EQ(std::string("SC002"), r.diagnostics[1].code);
}

void testDecisionGraphStorageOrder()
{
    Graph g;
    Node* z = g.createNode("z", Operation::Softmax); z->shape = {1};
    Node* a = g.createNode("a", Operation::Input); a->shape = {1};
    Node* m = g.createNode("m", Operation::ReLU); m->shape = {-1};
    auto r = BaselineStrategyClassifier::classify(g);
    EXPECT_PTR_EQ(z, r.decisions[0].node);
    EXPECT_PTR_EQ(a, r.decisions[1].node);
    EXPECT_PTR_EQ(m, r.decisions[2].node);
}

void testAssignmentsConversion()
{
    Graph g; Node* a = g.createNode("a", Operation::ReLU); a->shape = {1};
    Node* b = g.createNode("b", Operation::ReLU); b->shape = {-1};
    auto r = BaselineStrategyClassifier::classify(g); auto values = r.assignments();
    EXPECT_EQ(r.decisions.size(), values.size());
    for(std::size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_PTR_EQ(r.decisions[i].node, values[i].node);
        EXPECT_EQ(r.decisions[i].strategy, values[i].strategy);
    }
    EXPECT_TRUE(classifyOne(Operation::Unknown, {}).assignments().empty());
}

void testNoUnresolvedDecision()
{
    Graph g;
    const std::vector<Operation> ops{
        Operation::Input, Operation::Constant, Operation::ReLU,
        Operation::Add, Operation::MatMul, Operation::Softmax,
        Operation::Output, Operation::Transfer
    };
    for(std::size_t i = 0; i < ops.size(); ++i)
    {
        Node* n = g.createNode("n" + std::to_string(i), ops[i]);
        n->shape = (i == 3 ? std::vector<int>{-1} : std::vector<int>{1});
    }
    auto r = BaselineStrategyClassifier::classify(g);
    for(const auto& d : r.decisions)
        EXPECT_FALSE(d.strategy == RegionStrategy::Unresolved);
}

void testClassificationToFormationPipeline()
{
    Graph g; Node* a = g.createNode("a", Operation::Input); a->shape = {1};
    Node* b = g.createNode("b", Operation::ReLU); b->shape = {1};
    Node* c = g.createNode("c", Operation::Output); c->shape = {1};
    g.connect(a, b); g.connect(b, c);
    auto classification = BaselineStrategyClassifier::classify(g);
    auto formation = TopologicalRegionFormer::form(g, classification.assignments());
    EXPECT_TRUE(classification.succeeded()); EXPECT_TRUE(formation.succeeded());
    EXPECT_TRUE(RegionVerifier::verify(g, *formation.regionGraph).valid);
    EXPECT_TRUE(RegionVerifier::verify(
        g, *formation.regionGraph, RegionVerificationLevel::Executable).valid);
    for(const auto& d : classification.decisions)
        EXPECT_EQ(d.strategy, formation.regionGraph->regionFor(*d.node)->strategy());
}

void testAutomaticHybridChain()
{
    Graph g; Node* input = g.createNode("input", Operation::Input);
    Node* stat = g.createNode("staticRelu", Operation::ReLU);
    Node* dyn = g.createNode("dynamicRelu", Operation::ReLU);
    Node* soft = g.createNode("softmax", Operation::Softmax);
    Node* output = g.createNode("output", Operation::Output);
    input->shape = {1, 4}; stat->shape = {1, 4}; dyn->shape = {-1, 4};
    soft->shape = {-1, 4};
    g.connect(input, stat); g.connect(stat, dyn); g.connect(dyn, soft);
    g.connect(soft, output);
    auto c = BaselineStrategyClassifier::classify(g);
    auto f = TopologicalRegionFormer::form(g, c.assignments());
    EXPECT_TRUE(f.succeeded()); EXPECT_EQ(std::size_t(4), f.regionGraph->size());
    EXPECT_EQ(std::vector<const Node*>({input}), f.regionGraph->findRegion(0)->nodes());
    EXPECT_EQ(std::vector<const Node*>({stat}), f.regionGraph->findRegion(1)->nodes());
    EXPECT_EQ(std::vector<const Node*>({dyn}), f.regionGraph->findRegion(2)->nodes());
    EXPECT_EQ(std::vector<const Node*>({soft, output}),
              f.regionGraph->findRegion(3)->nodes());
    EXPECT_EQ(std::size_t(3), f.regionGraph->dependencies().size());
    for(std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(static_cast<RegionId>(i), f.regionGraph->dependencies()[i].source);
        EXPECT_EQ(static_cast<RegionId>(i + 1),
                  f.regionGraph->dependencies()[i].destination);
    }
}

void testClassificationRepeatability()
{
    for(int iteration = 0; iteration < 50; ++iteration)
    {
        Graph g; Node* a = g.createNode("a", Operation::Input); a->shape = {1};
        Node* b = g.createNode("b", Operation::ReLU); b->shape = {-1};
        auto first = BaselineStrategyClassifier::classify(g);
        auto second = BaselineStrategyClassifier::classify(g);
        EXPECT_EQ(first.decisions.size(), second.decisions.size());
        EXPECT_EQ(first.diagnostics.size(), second.diagnostics.size());
        for(std::size_t i = 0; i < first.decisions.size(); ++i)
        {
            EXPECT_EQ(first.decisions[i].node->name, second.decisions[i].node->name);
            EXPECT_EQ(first.decisions[i].strategy, second.decisions[i].strategy);
            EXPECT_EQ(first.decisions[i].shapeKnowledge,
                      second.decisions[i].shapeKnowledge);
            EXPECT_EQ(first.decisions[i].reason, second.decisions[i].reason);
        }
    }
}

void testAssignmentOrderIsGraphOrder()
{
    Graph g; Node* c = g.createNode("C", Operation::ReLU);
    Node* a = g.createNode("A", Operation::ReLU);
    Node* b = g.createNode("B", Operation::ReLU);
    c->shape = a->shape = b->shape = {1}; g.connect(a, b); g.connect(b, c);
    auto classification = BaselineStrategyClassifier::classify(g);
    auto assignments = classification.assignments();
    EXPECT_PTR_EQ(c, assignments[0].node); EXPECT_PTR_EQ(a, assignments[1].node);
    EXPECT_PTR_EQ(b, assignments[2].node);
    auto formation = TopologicalRegionFormer::form(g, assignments);
    EXPECT_EQ(std::vector<const Node*>({a, b, c}),
              formation.regionGraph->findRegion(0)->nodes());
}

void testGraphNonMutation()
{
    Graph g; Node* a = g.createNode("a", Operation::Constant);
    Node* b = g.createNode("b", Operation::ReLU);
    a->shape = {1}; b->shape = {-1}; g.connect(a, b);
    const auto holders = g.nodes; const auto edges = g.edges;
    const auto aShape = a->shape; const auto bShape = b->shape;
    const auto inputs = b->inputs; const auto outputs = a->outputs;
    const auto names = std::vector<std::string>{a->name, b->name};
    const auto ids = std::vector<int>{a->id, b->id};
    const auto attrs = std::vector{a->attributes, b->attributes};
    const auto devices = std::vector{a->device, b->device};
    const auto tensors = std::vector{a->tensor.shape, b->tensor.shape};
    auto r = BaselineStrategyClassifier::classify(g); EXPECT_TRUE(r.succeeded());
    EXPECT_EQ(holders, g.nodes); EXPECT_EQ(edges.size(), g.edges.size());
    EXPECT_EQ(aShape, a->shape); EXPECT_EQ(bShape, b->shape);
    EXPECT_EQ(inputs, b->inputs); EXPECT_EQ(outputs, a->outputs);
    EXPECT_EQ(names[0], a->name); EXPECT_EQ(names[1], b->name);
    EXPECT_EQ(ids[0], a->id); EXPECT_EQ(ids[1], b->id);
    EXPECT_EQ(attrs[0], a->attributes); EXPECT_EQ(attrs[1], b->attributes);
    EXPECT_EQ(devices[0], a->device); EXPECT_EQ(devices[1], b->device);
    EXPECT_EQ(tensors[0], a->tensor.shape); EXPECT_EQ(tensors[1], b->tensor.shape);
}

void testSameNameDistinctNodeIdentity()
{
    Graph g; Node* a = g.createNode("same", Operation::ReLU); a->shape = {1};
    Node* b = g.createNode("same", Operation::ReLU); b->shape = {1};
    auto r = BaselineStrategyClassifier::classify(g);
    EXPECT_EQ(std::size_t(2), r.decisions.size());
    EXPECT_PTR_EQ(a, r.decisions[0].node); EXPECT_PTR_EQ(b, r.decisions[1].node);
}

void testThousandNodeClassification()
{
    Graph g;
    for(int i = 0; i < 1000; ++i)
    {
        Operation op = i == 0 ? Operation::Input :
            (i == 999 ? Operation::Output :
             (i % 3 == 1 ? Operation::ReLU :
              (i % 3 == 2 ? Operation::ReLU : Operation::Softmax)));
        Node* n = g.createNode("n" + std::to_string(i), op);
        n->shape = i % 3 == 2 ? std::vector<int>{-1, 4}
                              : std::vector<int>{1, 4};
    }
    auto a = BaselineStrategyClassifier::classify(g);
    auto b = BaselineStrategyClassifier::classify(g);
    EXPECT_TRUE(a.succeeded()); EXPECT_EQ(std::size_t(1000), a.decisions.size());
    EXPECT_EQ(std::size_t(1000), a.assignments().size());
    for(std::size_t i = 0; i < a.decisions.size(); ++i)
    {
        EXPECT_PTR_EQ(g.nodes[i].get(), a.decisions[i].node);
        EXPECT_FALSE(a.decisions[i].strategy == RegionStrategy::Unresolved);
        EXPECT_EQ(a.decisions[i].strategy, b.decisions[i].strategy);
        EXPECT_EQ(a.decisions[i].reason, b.decisions[i].reason);
    }
}

void testDynamicConstantMultipleMinusOne()
{
    auto r = classifyOne(Operation::Constant, {-1, -1});
    EXPECT_DIAGNOSTIC_COUNT(r, "SC005", 1);
}

void testInvalidDynamicConstantSuppressesSc005()
{
    auto r = classifyOne(Operation::Constant, {-1, 0});
    EXPECT_DIAGNOSTIC_COUNT(r, "SC004", 1);
    EXPECT_DIAGNOSTIC_COUNT(r, "SC005", 0);
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
    runTest("Empty Graph success", testEmptyGraphSuccess);
    runTest("ShapeKnowledge strings", testShapeKnowledgeStrings);
    runTest("Reason strings", testReasonStrings);
    runTest("Input static fallback", testInputStaticFallback);
    runTest("Input dynamic fallback", testInputDynamicFallback);
    runTest("Input unknown-rank fallback", testInputUnknownRankFallback);
    runTest("Output boundary fallback", testOutputBoundaryFallback);
    runTest("Transfer fallback", testTransferFallback);
    runTest("Scalar Constant AOT", testScalarConstantAot);
    runTest("Static tensor Constant AOT", testStaticTensorConstantAot);
    runTest("Dynamic Constant failure", testDynamicConstantFailure);
    runTest("Optimized operation static matrix", testOptimizedOperationStaticMatrix);
    runTest("Optimized operation dynamic matrix", testOptimizedOperationDynamicMatrix);
    runTest("Optimized operation multiple dynamic dimensions", testOptimizedOperationMultipleDynamicDimensions);
    runTest("Optimized operation unknown rank", testOptimizedOperationUnknownRank);
    runTest("Fallback-only static matrix", testFallbackOnlyStaticMatrix);
    runTest("Fallback-only dynamic matrix", testFallbackOnlyDynamicMatrix);
    runTest("Fallback-only unknown rank", testFallbackOnlyUnknownRank);
    runTest("Zero dimension failure", testZeroDimensionFailure);
    runTest("Less-than-minus-one failure", testLessThanMinusOneFailure);
    runTest("Multiple invalid dimensions", testMultipleInvalidDimensions);
    runTest("Unknown operation failure", testUnknownOperationFailure);
    runTest("Unknown operation with invalid shape order", testUnknownOperationWithInvalidShapeOrder);
    runTest("Null Graph holder", testNullGraphHolder);
    runTest("Duplicate Graph holder", testDuplicateGraphHolder);
    runTest("Storage diagnostic category order", testStorageDiagnosticCategoryOrder);
    runTest("Decision Graph storage order", testDecisionGraphStorageOrder);
    runTest("Assignments conversion", testAssignmentsConversion);
    runTest("No unresolved decision", testNoUnresolvedDecision);
    runTest("Classification-to-formation pipeline", testClassificationToFormationPipeline);
    runTest("Automatic hybrid chain", testAutomaticHybridChain);
    runTest("Classification repeatability", testClassificationRepeatability);
    runTest("Assignment order is Graph order", testAssignmentOrderIsGraphOrder);
    runTest("Graph non-mutation", testGraphNonMutation);
    runTest("Same-name distinct Node identity", testSameNameDistinctNodeIdentity);
    runTest("Thousand-Node classification", testThousandNodeClassification);
    runTest("Dynamic Constant multiple minus one", testDynamicConstantMultipleMinusOne);
    runTest("Invalid dynamic Constant suppresses SC005", testInvalidDynamicConstantSuppressesSc005);

    if(failures != 0)
    {
        std::cerr << failures << " Strategy Classification test failure(s)\n";
        return 1;
    }
    std::cout << "All Strategy Classification tests passed\n";
    return 0;
}
