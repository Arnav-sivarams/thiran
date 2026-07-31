#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "region/RegionPlan.hpp"
#include "region/RegionPlanVerifier.hpp"
#include "region/RegionVerifier.hpp"
#include "utils/RegionPlanPrinter.hpp"

using namespace thiran;
using namespace thiran::region;

namespace
{
int failures = 0;
std::string currentTest;

void expect(bool condition, const char* expression, int line)
{
    if(condition) return;
    std::cerr << currentTest << ":" << line << ": expected " << expression
              << " to be true, actual false\n";
    ++failures;
}
#define EXPECT_TRUE(value) expect(static_cast<bool>(value), #value, __LINE__)
#define EXPECT_FALSE(value) expect(!(value), "!(" #value ")", __LINE__)
#define EXPECT_EQ(left, right) expect((left) == (right), #left " == " #right, __LINE__)
#define EXPECT_PTR_EQ(left, right) expect((left) == (right), #left " == " #right, __LINE__)

bool hasCode(const std::vector<RegionPlanDiagnostic>& diagnostics,
             const std::string& code)
{
    for(const auto& diagnostic : diagnostics)
        if(diagnostic.code == code) return true;
    return false;
}

std::string print(const RegionPlan& plan)
{
    std::ostringstream output;
    RegionPlanPrinter::print(plan, output);
    return output.str();
}

struct Hybrid {
    Graph graph;
    Node* input;
    Node* stat;
    Node* dynamic;
    Node* softmax;
    Node* output;
    Hybrid()
    {
        input = graph.createNode("input", Operation::Input);
        stat = graph.createNode("staticRelu", Operation::ReLU);
        dynamic = graph.createNode("dynamicRelu", Operation::ReLU);
        softmax = graph.createNode("softmax", Operation::Softmax);
        output = graph.createNode("output", Operation::Output);
        input->shape = {1, 4}; stat->shape = {1, 4};
        dynamic->shape = {-1, 4}; softmax->shape = {-1, 4};
        graph.connect(input, stat); graph.connect(stat, dynamic);
        graph.connect(dynamic, softmax); graph.connect(softmax, output);
    }
};

void testDiagnosticStageStrings()
{
    EXPECT_EQ(toString(RegionPlanDiagnosticStage::Classification), "CLASSIFICATION");
    EXPECT_EQ(toString(RegionPlanDiagnosticStage::Formation), "FORMATION");
    EXPECT_EQ(toString(RegionPlanDiagnosticStage::Verification), "VERIFICATION");
    EXPECT_EQ(toString(static_cast<RegionPlanDiagnosticStage>(255)), "UNKNOWN");
}
void testResultSuccessContract()
{
    Graph good; auto success = HybridRegionPlanner::build(good);
    EXPECT_TRUE(success.succeeded()); EXPECT_TRUE(success.plan != nullptr);
    Graph bad; Node* node = bad.createNode("bad", Operation::Unknown);
    node->shape = {1}; auto failure = HybridRegionPlanner::build(bad);
    EXPECT_FALSE(failure.succeeded()); EXPECT_TRUE(failure.plan == nullptr);
    EXPECT_FALSE(failure.diagnostics.empty());
}
void testEmptyGraphPlan()
{
    Graph graph; auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(result.succeeded()); EXPECT_TRUE(result.plan->empty());
    EXPECT_TRUE(result.plan->regionGraph().finalized());
    EXPECT_EQ(result.plan->nodeCount(), std::size_t(0));
    EXPECT_EQ(result.plan->regionCount(), std::size_t(0));
    EXPECT_TRUE(RegionPlanVerifier::verify(graph, *result.plan).valid);
    EXPECT_EQ(print(*result.plan),
        "HybridRegionPlan\n  Nodes: 0\n  Regions: 0\n"
        "  Dependencies: 0\n\nDecisions\n  <none>\n\nRegions\n"
        "  <none>\n\nDependencies\n  <none>\n");
}
void testSingleInputPlan()
{
    Graph graph; Node* input = graph.createNode("input", Operation::Input);
    input->shape = {1, 4}; auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(result.succeeded()); EXPECT_EQ(result.plan->nodeCount(), std::size_t(1));
    EXPECT_EQ(result.plan->decisions()[0].strategy, RegionStrategy::Fallback);
    EXPECT_EQ(result.plan->decisions()[0].reason, StrategyDecisionReason::InputBoundary);
    EXPECT_PTR_EQ(result.plan->decisionFor(*input), &result.plan->decisions()[0]);
    EXPECT_EQ(result.plan->regionFor(*input)->strategy(), RegionStrategy::Fallback);
}
void testStaticOptimizedChain()
{
    Graph graph; Node* input = graph.createNode("input", Operation::Input);
    Node* mm = graph.createNode("matmul", Operation::MatMul);
    Node* relu = graph.createNode("relu", Operation::ReLU);
    Node* output = graph.createNode("output", Operation::Output);
    for(Node* node : {input, mm, relu, output}) node->shape = {1, 4};
    graph.connect(input, mm); graph.connect(mm, relu); graph.connect(relu, output);
    auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(result.succeeded()); EXPECT_EQ(result.plan->regionCount(), std::size_t(3));
    EXPECT_EQ(result.plan->regionGraph().findRegion(1)->nodes(),
              std::vector<const Node*>({mm, relu}));
    EXPECT_EQ(result.plan->regionGraph().dependencies().size(), std::size_t(2));
}
void testAutomaticHybridChain()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    EXPECT_TRUE(result.succeeded()); EXPECT_EQ(result.plan->nodeCount(), std::size_t(5));
    EXPECT_EQ(result.plan->regionCount(), std::size_t(4));
    EXPECT_EQ(result.plan->regionGraph().dependencies().size(), std::size_t(3));
    EXPECT_TRUE(RegionVerifier::verify(value.graph, result.plan->regionGraph()).valid);
    EXPECT_TRUE(RegionVerifier::verify(value.graph, result.plan->regionGraph(),
        RegionVerificationLevel::Executable).valid);
}
void testExactPrinterSnapshot()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    const std::string expected =
"HybridRegionPlan\n"
"  Nodes: 5\n"
"  Regions: 4\n"
"  Dependencies: 3\n"
"\n"
"Decisions\n"
"  [0] input | Shape: STATIC | Strategy: FALLBACK | Reason: INPUT_BOUNDARY | Region: 0\n"
"  [1] staticRelu | Shape: STATIC | Strategy: AOT | Reason: STATIC_OPTIMIZED_OPERATION | Region: 1\n"
"  [2] dynamicRelu | Shape: RUNTIME_SPECIALIZABLE | Strategy: JIT | Reason: DYNAMIC_OPTIMIZED_OPERATION | Region: 2\n"
"  [3] softmax | Shape: RUNTIME_SPECIALIZABLE | Strategy: FALLBACK | Reason: FALLBACK_ONLY_OPERATION | Region: 3\n"
"  [4] output | Shape: UNKNOWN_RANK | Strategy: FALLBACK | Reason: OUTPUT_BOUNDARY | Region: 3\n"
"\n"
"Regions\n"
"  Region 0 [FALLBACK]\n"
"    Nodes: input\n"
"    Inputs: <none>\n"
"    Outputs: input\n"
"  Region 1 [AOT]\n"
"    Nodes: staticRelu\n"
"    Inputs: input\n"
"    Outputs: staticRelu\n"
"  Region 2 [JIT]\n"
"    Nodes: dynamicRelu\n"
"    Inputs: staticRelu\n"
"    Outputs: dynamicRelu\n"
"  Region 3 [FALLBACK]\n"
"    Nodes: softmax, output\n"
"    Inputs: dynamicRelu\n"
"    Outputs: output\n"
"\n"
"Dependencies\n"
"  0 -> 1: input\n"
"  1 -> 2: staticRelu\n"
"  2 -> 3: dynamicRelu\n";
    EXPECT_EQ(print(*result.plan), expected);
}
void testDecisionLookup()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    for(std::size_t i = 0; i < value.graph.nodes.size(); ++i)
        EXPECT_PTR_EQ(result.plan->decisionFor(*value.graph.nodes[i]),
                      &result.plan->decisions()[i]);
}
void testRegionLookup()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    for(const auto& decision : result.plan->decisions())
    {
        EXPECT_PTR_EQ(result.plan->regionFor(*decision.node),
            result.plan->regionGraph().regionFor(*decision.node));
        EXPECT_EQ(result.plan->regionFor(*decision.node)->strategy(), decision.strategy);
    }
}
void testForeignLookup()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    Graph foreign; Node* node = foreign.createNode("input", Operation::Input);
    node->shape = {1, 4};
    EXPECT_TRUE(result.plan->decisionFor(*node) == nullptr);
    EXPECT_TRUE(result.plan->regionFor(*node) == nullptr);
}
void testGraphStorageDecisionOrder()
{
    Graph graph; Node* c = graph.createNode("C", Operation::ReLU);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    c->shape = a->shape = b->shape = {1}; graph.connect(a, b); graph.connect(b, c);
    auto result = HybridRegionPlanner::build(graph);
    EXPECT_PTR_EQ(result.plan->decisions()[0].node, c);
    EXPECT_EQ(result.plan->regionGraph().findRegion(0)->nodes(),
              std::vector<const Node*>({a, b, c}));
}
void testPlanMoveConstruction()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    const std::string before = print(*result.plan);
    RegionPlan moved(std::move(*result.plan));
    EXPECT_EQ(print(moved), before);
    EXPECT_TRUE(RegionPlanVerifier::verify(value.graph, moved).valid);
}
void testPlanMoveAssignment()
{
    Hybrid first; Hybrid second;
    auto a = HybridRegionPlanner::build(first.graph);
    auto b = HybridRegionPlanner::build(second.graph);
    const std::string expected = print(*a.plan);
    *b.plan = std::move(*a.plan);
    EXPECT_PTR_EQ(&b.plan->sourceGraph(), &first.graph);
    EXPECT_EQ(print(*b.plan), expected);
}
void testTypeImmutabilityContracts()
{
    static_assert(!std::is_copy_constructible_v<RegionPlan>);
    static_assert(!std::is_copy_assignable_v<RegionPlan>);
    static_assert(std::is_move_constructible_v<RegionPlan>);
    static_assert(std::is_move_assignable_v<RegionPlan>);
    static_assert(std::is_same_v<decltype(std::declval<const RegionPlan&>().decisions()),
                                 const std::vector<NodeStrategyDecision>&>);
    static_assert(std::is_same_v<decltype(std::declval<const RegionPlan&>().regionGraph()),
                                 const RegionGraph&>);
    EXPECT_TRUE(true);
}
void testClassificationFailurePropagation()
{
    Graph graph; Node* node = graph.createNode("unknown", Operation::Unknown);
    node->shape = {1}; auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(hasCode(result.diagnostics, "SC003"));
    EXPECT_EQ(result.diagnostics[0].stage, RegionPlanDiagnosticStage::Classification);
}
void testShapeFailurePropagation()
{
    Graph graph; Node* node = graph.createNode("bad", Operation::ReLU);
    node->shape = {1, 0, 4}; auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(hasCode(result.diagnostics, "SC004"));
    EXPECT_EQ(result.diagnostics[0].dimensionIndex, std::optional<std::size_t>(1));
}
void testMultipleClassificationDiagnosticOrder()
{
    Graph graph; Node* node = graph.createNode("bad", Operation::Unknown);
    node->shape = {0}; auto result = HybridRegionPlanner::build(graph);
    EXPECT_EQ(result.diagnostics.size(), std::size_t(2));
    EXPECT_EQ(result.diagnostics[0].code, "SC003");
    EXPECT_EQ(result.diagnostics[1].code, "SC004");
}
void testNullInputFormationFailurePropagation()
{
    Graph graph; Node* node = graph.createNode("node", Operation::ReLU);
    node->shape = {1}; node->inputs.push_back(nullptr);
    auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(hasCode(result.diagnostics, "RF008"));
    EXPECT_EQ(result.diagnostics[0].stage, RegionPlanDiagnosticStage::Formation);
}
void testForeignInputFormationFailurePropagation()
{
    Graph graph; Node* node = graph.createNode("node", Operation::ReLU);
    Graph other; Node* foreign = other.createNode("secret", Operation::Input);
    node->shape = foreign->shape = {1}; node->inputs.push_back(foreign);
    auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(hasCode(result.diagnostics, "RF009"));
    EXPECT_TRUE(result.diagnostics[0].message.find("secret") == std::string::npos);
}
void testCyclicGraphFormationFailurePropagation()
{
    Graph graph; Node* a = graph.createNode("a", Operation::ReLU);
    Node* b = graph.createNode("b", Operation::ReLU);
    a->shape = b->shape = {1}; graph.connect(a, b); graph.connect(b, a);
    auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(hasCode(result.diagnostics, "RF010"));
}
void testRegionPlanVerifierSuccess()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    EXPECT_TRUE(RegionPlanVerifier::verify(value.graph, *result.plan).valid);
}
void testVerifierSourceGraphMismatch()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    Graph other; auto verification = RegionPlanVerifier::verify(other, *result.plan);
    EXPECT_FALSE(verification.valid); EXPECT_TRUE(hasCode(verification.diagnostics, "RP001"));
}
void testStaleShapeChange()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    value.stat->shape = {-1, 4};
    auto verification = RegionPlanVerifier::verify(value.graph, *result.plan);
    EXPECT_TRUE(hasCode(verification.diagnostics, "RP013"));
    EXPECT_EQ(result.plan->decisions()[1].strategy, RegionStrategy::AheadOfTime);
}
void testStaleOperationChange()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    value.stat->op = Operation::Softmax;
    EXPECT_TRUE(hasCode(RegionPlanVerifier::verify(value.graph, *result.plan).diagnostics,
                        "RP013"));
}
void testSourceBecomesUnclassifiable()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    value.stat->shape = {0};
    auto verification = RegionPlanVerifier::verify(value.graph, *result.plan);
    EXPECT_TRUE(hasCode(verification.diagnostics, "RP012"));
    bool mentions = false;
    for(const auto& d : verification.diagnostics)
        if(d.code == "RP012" && d.message.find("SC004") != std::string::npos)
            mentions = true;
    EXPECT_TRUE(mentions);
}
void testStaleAdjacencyChange()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    value.input->inputs.push_back(value.output);
    auto verification = RegionPlanVerifier::verify(value.graph, *result.plan);
    EXPECT_TRUE(hasCode(verification.diagnostics, "RP014"));
}
void testDecisionRegionConsistency()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    for(const auto& decision : result.plan->decisions())
    {
        EXPECT_FALSE(decision.strategy == RegionStrategy::Unresolved);
        EXPECT_EQ(decision.strategy, result.plan->regionFor(*decision.node)->strategy());
    }
}
void testClassificationAndFormationReplay()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    auto c = BaselineStrategyClassifier::classify(value.graph);
    auto f = TopologicalRegionFormer::form(value.graph, c.assignments());
    EXPECT_EQ(c.decisions.size(), result.plan->decisions().size());
    EXPECT_EQ(f.regionGraph->size(), result.plan->regionCount());
    EXPECT_TRUE(RegionPlanVerifier::verify(value.graph, *result.plan).valid);
}
void testRepeatability()
{
    Hybrid value; const std::string expected =
        print(*HybridRegionPlanner::build(value.graph).plan);
    for(int i = 0; i < 50; ++i)
        EXPECT_EQ(print(*HybridRegionPlanner::build(value.graph).plan), expected);
}
void testGraphNonMutation()
{
    Hybrid value; const auto holders = value.graph.nodes;
    const auto edges = value.graph.edges; const auto inputs = value.stat->inputs;
    auto result = HybridRegionPlanner::build(value.graph);
    EXPECT_TRUE(result.succeeded()); EXPECT_EQ(value.graph.nodes, holders);
    EXPECT_EQ(value.graph.edges.size(), edges.size()); EXPECT_EQ(value.stat->inputs, inputs);
}
void testDisconnectedGraph()
{
    Graph graph; Node* a = graph.createNode("a", Operation::ReLU);
    Node* b = graph.createNode("b", Operation::ReLU);
    a->shape = b->shape = {1}; auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(result.succeeded()); EXPECT_EQ(result.plan->nodeCount(), std::size_t(2));
    EXPECT_TRUE(RegionPlanVerifier::verify(graph, *result.plan).valid);
}
void testMixedDiamond()
{
    Graph graph; Node* source = graph.createNode("source", Operation::Input);
    Node* left = graph.createNode("left", Operation::ReLU);
    Node* right = graph.createNode("right", Operation::ReLU);
    Node* join = graph.createNode("join", Operation::Add);
    source->shape = {1}; left->shape = right->shape = {-1}; join->shape = {1};
    graph.connect(source, left); graph.connect(source, right);
    graph.connect(left, join); graph.connect(right, join);
    auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(result.plan->regionFor(*left) != result.plan->regionFor(*right));
}
void testThousandNodePlan()
{
    Graph graph; Node* previous = nullptr;
    for(int i = 0; i < 1000; ++i)
    {
        Node* node = graph.createNode("n" + std::to_string(i), Operation::ReLU);
        node->shape = (i % 2 == 0) ? std::vector<int>{1} : std::vector<int>{-1};
        if(previous != nullptr) graph.connect(previous, node);
        previous = node;
    }
    auto result = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(result.succeeded()); EXPECT_EQ(result.plan->nodeCount(), std::size_t(1000));
    EXPECT_TRUE(RegionPlanVerifier::verify(graph, *result.plan).valid);
}
void testEmptyAndNonemptyResultInvariants() { testResultSuccessContract(); testEmptyGraphPlan(); }
void testPrinterNeverEmitsPointers()
{
    Hybrid value; auto result = HybridRegionPlanner::build(value.graph);
    EXPECT_TRUE(print(*result.plan).find("0x") == std::string::npos);
}
void testPrinterReasonCoverage()
{
    Graph graph;
    const std::vector<std::pair<Operation, std::vector<int>>> cases{
        {Operation::Input, {1}}, {Operation::Output, {1}},
        {Operation::Transfer, {1}}, {Operation::Constant, {1}},
        {Operation::ReLU, {1}}, {Operation::ReLU, {-1}},
        {Operation::Add, {}}, {Operation::Softmax, {1}}};
    for(std::size_t i = 0; i < cases.size(); ++i)
    {
        Node* node = graph.createNode("reason" + std::to_string(i), cases[i].first);
        node->shape = cases[i].second;
    }
    auto result = HybridRegionPlanner::build(graph); const std::string text = print(*result.plan);
    for(const char* reason : {"INPUT_BOUNDARY", "OUTPUT_BOUNDARY", "RUNTIME_TRANSFER",
         "STATIC_CONSTANT", "STATIC_OPTIMIZED_OPERATION", "DYNAMIC_OPTIMIZED_OPERATION",
         "UNKNOWN_RANK_FALLBACK", "FALLBACK_ONLY_OPERATION"})
        EXPECT_TRUE(text.find(reason) != std::string::npos);
}

template<typename Function>
void runTest(const std::string& name, Function function)
{
    currentTest = name; function();
}
}

int main()
{
    runTest("Diagnostic stage strings", testDiagnosticStageStrings);
    runTest("Result success contract", testResultSuccessContract);
    runTest("Empty Graph plan", testEmptyGraphPlan);
    runTest("Single Input plan", testSingleInputPlan);
    runTest("Static optimized chain", testStaticOptimizedChain);
    runTest("Automatic hybrid chain", testAutomaticHybridChain);
    runTest("Exact printer snapshot", testExactPrinterSnapshot);
    runTest("Decision lookup", testDecisionLookup);
    runTest("Region lookup", testRegionLookup);
    runTest("Foreign lookup", testForeignLookup);
    runTest("Graph storage decision order", testGraphStorageDecisionOrder);
    runTest("Plan move construction", testPlanMoveConstruction);
    runTest("Plan move assignment", testPlanMoveAssignment);
    runTest("Type immutability contracts", testTypeImmutabilityContracts);
    runTest("Classification failure propagation", testClassificationFailurePropagation);
    runTest("Shape failure propagation", testShapeFailurePropagation);
    runTest("Multiple classification diagnostic order", testMultipleClassificationDiagnosticOrder);
    runTest("Null input formation failure propagation", testNullInputFormationFailurePropagation);
    runTest("Foreign input formation failure propagation", testForeignInputFormationFailurePropagation);
    runTest("Cyclic Graph formation failure propagation", testCyclicGraphFormationFailurePropagation);
    runTest("RegionPlan verifier success", testRegionPlanVerifierSuccess);
    runTest("Verifier source Graph mismatch", testVerifierSourceGraphMismatch);
    runTest("Stale shape change", testStaleShapeChange);
    runTest("Stale operation change", testStaleOperationChange);
    runTest("Source becomes unclassifiable", testSourceBecomesUnclassifiable);
    runTest("Stale adjacency change", testStaleAdjacencyChange);
    runTest("Decision-Region consistency", testDecisionRegionConsistency);
    runTest("Classification and formation replay", testClassificationAndFormationReplay);
    runTest("Repeatability", testRepeatability);
    runTest("Graph non-mutation", testGraphNonMutation);
    runTest("Disconnected Graph", testDisconnectedGraph);
    runTest("Mixed diamond", testMixedDiamond);
    runTest("Thousand-Node plan", testThousandNodePlan);
    runTest("Empty and nonempty result invariants", testEmptyAndNonemptyResultInvariants);
    runTest("Printer never emits pointers", testPrinterNeverEmitsPointers);
    runTest("Printer reason coverage", testPrinterReasonCoverage);
    if(failures != 0)
    {
        std::cerr << failures << " Region Plan test failure(s)\n";
        return 1;
    }
    std::cout << "All Region Plan tests passed\n";
    return 0;
}
