#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "region/RegionPlan.hpp"
#include "region/RegionPythonEmitter.hpp"
#include "utils/RegionExecutionDiagnosticPrinter.hpp"

using namespace thiran;
using namespace thiran::region;

namespace
{

int failures = 0;
std::string currentTest;

void expect(bool condition, const char* expression, int line)
{
    if(condition) return;
    ++failures;
    std::cerr << currentTest << ':' << line << ": expected "
              << expression << " to be true, actual false\n";
}

#define EXPECT_TRUE(value) expect(static_cast<bool>(value), #value, __LINE__)
#define EXPECT_FALSE(value) expect(!(value), "!(" #value ")", __LINE__)
#define EXPECT_EQ(left, right) expect((left) == (right), #left " == " #right, __LINE__)
#define EXPECT_CONTAINS(text, token) expect((text).find(token) != std::string::npos, #text " contains " #token, __LINE__)

struct Chain
{
    Graph graph;
    Node* input;
    Node* relu;
    Node* output;

    explicit Chain(std::vector<int> shape = {1, 4})
    {
        input = graph.createNode("input", Operation::Input);
        relu = graph.createNode("relu", Operation::ReLU);
        output = graph.createNode("output", Operation::Output);
        input->shape = shape;
        relu->shape = shape;
        output->shape = shape;
        graph.connect(input, relu);
        graph.connect(relu, output);
    }
};

RegionPythonEmissionResult emit(Graph& graph)
{
    auto plan = HybridRegionPlanner::build(graph);
    if(!plan.succeeded())
    {
        return {};
    }
    return RegionPythonEmitter::emit(graph, *plan.plan);
}

bool hasCode(
    const std::vector<RegionExecutionDiagnostic>& diagnostics,
    const std::string& code
)
{
    for(const auto& diagnostic : diagnostics)
        if(diagnostic.code == code) return true;
    return false;
}

std::string printed(
    const std::vector<RegionExecutionDiagnostic>& diagnostics
)
{
    std::ostringstream output;
    RegionExecutionDiagnosticPrinter::print(diagnostics, output);
    return output.str();
}

void testResultSuccessContract()
{
    Graph graph;
    auto plan = HybridRegionPlanner::build(graph);
    auto success = RegionPythonEmitter::emit(graph, *plan.plan);
    EXPECT_TRUE(success.succeeded());
    EXPECT_TRUE(success.artifact.has_value());
    EXPECT_TRUE(success.diagnostics.empty());
    Graph other;
    auto failure = RegionPythonEmitter::emit(other, *plan.plan);
    EXPECT_FALSE(failure.succeeded());
    EXPECT_FALSE(failure.artifact.has_value());
    EXPECT_FALSE(failure.diagnostics.empty());
}

void testEmptyGraphEmission()
{
    Graph graph;
    auto result = emit(graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_CONTAINS(result.artifact->source, "runtime_device = torch.device('cpu')");
    EXPECT_FALSE(result.artifact->source.find("def _region_") != std::string::npos);
}

void testSingleInputToOutput()
{
    Chain chain;
    auto result = emit(chain.graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_CONTAINS(result.artifact->source, "v_2 = v_1");
    EXPECT_CONTAINS(result.artifact->source, "outputs['output'] = values[2]");
}

void testStaticOptimizedChain()
{
    Chain chain;
    auto result = emit(chain.graph);
    EXPECT_CONTAINS(result.artifact->source, "v_1 = torch.relu(v_0)");
}

void testAutomaticHybridChain()
{
    Chain chain({-1, 4});
    auto result = emit(chain.graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_CONTAINS(result.artifact->source, "strategy: JIT");
}

void testOneFunctionPerRegion()
{
    Chain chain;
    auto plan = HybridRegionPlanner::build(chain.graph);
    auto result = RegionPythonEmitter::emit(chain.graph, *plan.plan);
    for(const auto& region : plan.plan->regionGraph().regions())
        EXPECT_CONTAINS(result.artifact->source,
            "def _region_" + std::to_string(region->id()));
}

void testStrategyComments()
{
    Chain chain;
    auto result = emit(chain.graph);
    EXPECT_CONTAINS(result.artifact->source, "# Region 0 strategy: FALLBACK");
    EXPECT_CONTAINS(result.artifact->source, "strategy: AOT");
}

void testRegionDefinitionsFollowStorageOrder()
{
    Chain chain;
    auto result = emit(chain.graph);
    const auto first = result.artifact->source.find("def _region_0");
    const auto second = result.artifact->source.find("def _region_1");
    EXPECT_TRUE(first < second);
}

void testOrchestratorFollowsDependencyOrder()
{
    Chain chain;
    auto result = emit(chain.graph);
    const auto body = result.artifact->source.find("def run_region_plan");
    const auto first = result.artifact->source.find("_region_0(", body);
    const auto second = result.artifact->source.find("_region_1(", body);
    EXPECT_TRUE(first < second);
}

void testRegionNodesFollowStoredExecutionOrder()
{
    Chain chain;
    auto result = emit(chain.graph);
    EXPECT_TRUE(result.artifact->source.find("v_1 = torch.relu(v_0)") <
                result.artifact->source.find("v_2 = v_1"));
}

void testExternalArgumentsFollowRegionInputOrder()
{
    Graph graph;
    Node* a = graph.createNode("a", Operation::Input);
    Node* b = graph.createNode("b", Operation::Input);
    Node* add = graph.createNode("add", Operation::Add);
    a->shape = b->shape = add->shape = {1};
    graph.connect(a, add); graph.connect(b, add);
    auto result = emit(graph);
    EXPECT_CONTAINS(result.artifact->source,
        "(inputs, runtime_device, v_0, v_1)");
}

void testRegionReturnsFollowOutputOrder()
{
    Chain chain;
    auto result = emit(chain.graph);
    EXPECT_CONTAINS(result.artifact->source, "return (v_0,)");
}

void testMultipleCrossingValues()
{
    Graph graph;
    Node* a = graph.createNode("a", Operation::Input);
    Node* b = graph.createNode("b", Operation::Input);
    Node* add = graph.createNode("add", Operation::Add);
    for(Node* node : {a, b, add}) node->shape = {1};
    graph.connect(a, add); graph.connect(b, add);
    auto result = emit(graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_CONTAINS(result.artifact->source, "values[");
}

void testSharedCrossingValuePassedToTwoRegions()
{
    Graph graph;
    Node* source = graph.createNode("source", Operation::Input);
    Node* left = graph.createNode("left", Operation::ReLU);
    Node* right = graph.createNode("right", Operation::ReLU);
    source->shape = {1}; left->shape = right->shape = {-1};
    graph.connect(source, left); graph.connect(source, right);
    auto result = emit(graph);
    EXPECT_TRUE(result.succeeded());
    const auto first = result.artifact->source.find("values[0]");
    EXPECT_TRUE(first != std::string::npos);
    EXPECT_TRUE(result.artifact->source.find("values[0]", first + 1) !=
                std::string::npos);
}

void testDisconnectedComponents()
{
    Graph graph;
    Node* a = graph.createNode("a", Operation::Input);
    Node* b = graph.createNode("b", Operation::Input);
    a->shape = b->shape = {1};
    auto result = emit(graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_CONTAINS(result.artifact->source, "def _region_1");
}

void testScalarConstant()
{
    Graph graph;
    Node* value = graph.createNode("constant", Operation::Constant);
    value->constantValue = 1.25f;
    auto result = emit(graph);
    EXPECT_CONTAINS(result.artifact->source,
        "torch.tensor(1.25, dtype=torch.float32, device=runtime_device)");
}

void testShapedConstant()
{
    Graph graph;
    Node* value = graph.createNode("constant", Operation::Constant);
    value->constantValue = 2.0f; value->shape = {2, 3};
    auto result = emit(graph);
    EXPECT_CONTAINS(result.artifact->source,
        "torch.full((2, 3), 2.0, dtype=torch.float32");
}

void testMultipleSemanticOutputs()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* one = graph.createNode("one", Operation::Output);
    Node* two = graph.createNode("two", Operation::Output);
    input->shape = one->shape = two->shape = {1};
    graph.connect(input, one); graph.connect(input, two);
    auto result = emit(graph);
    EXPECT_CONTAINS(result.artifact->source, "outputs['one']");
    EXPECT_CONTAINS(result.artifact->source, "outputs['two']");
}

void testOutputAliasesOperandValue()
{
    Chain chain;
    auto result = emit(chain.graph);
    EXPECT_CONTAINS(result.artifact->source, "v_2 = v_1");
}

void testInputAndFallbackOperationInOneRegion()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* softmax = graph.createNode("softmax", Operation::Softmax);
    input->shape = softmax->shape = {1};
    graph.connect(input, softmax);
    auto result = emit(graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_CONTAINS(result.artifact->source, "torch.softmax(v_0, dim=-1)");
}

void testOutputAndFallbackOperationInOneRegion()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* softmax = graph.createNode("softmax", Operation::Softmax);
    Node* output = graph.createNode("output", Operation::Output);
    input->shape = softmax->shape = output->shape = {1};
    graph.connect(input, softmax); graph.connect(softmax, output);
    auto result = emit(graph);
    EXPECT_CONTAINS(result.artifact->source, "v_2 = v_1");
}

void testCompleteOperationLoweringMatrix()
{
    const std::vector<Operation> unary{
        Operation::ReLU, Operation::Sigmoid, Operation::Tanh,
        Operation::Softmax, Operation::MaxPool, Operation::AvgPool,
        Operation::Reshape, Operation::Transpose, Operation::Transfer
    };
    for(const Operation operation : unary)
    {
        Graph graph;
        Node* input = graph.createNode("input", Operation::Input);
        Node* node = graph.createNode("node", operation);
        input->shape = node->shape = {2, 2};
        graph.connect(input, node);
        EXPECT_TRUE(emit(graph).succeeded());
    }
    const std::vector<Operation> binary{
        Operation::Add, Operation::Subtract, Operation::Multiply,
        Operation::Divide, Operation::MatMul, Operation::Conv2D
    };
    for(const Operation operation : binary)
    {
        Graph graph;
        Node* a = graph.createNode("a", Operation::Input);
        Node* b = graph.createNode("b", Operation::Input);
        Node* node = graph.createNode("node", operation);
        a->shape = b->shape = node->shape = {2, 2};
        graph.connect(a, node); graph.connect(b, node);
        EXPECT_TRUE(emit(graph).succeeded());
    }
}

void testFusedMatMulReluLowering()
{
    Graph graph;
    Node* a = graph.createNode("a", Operation::Input);
    Node* b = graph.createNode("b", Operation::Input);
    Node* node = graph.createNode("fused", Operation::FusedMatMulRelu);
    a->shape = b->shape = node->shape = {2, 2};
    graph.connect(a, node); graph.connect(b, node);
    auto result = emit(graph);
    EXPECT_CONTAINS(result.artifact->source,
        "torch.relu(torch.matmul(v_0, v_1))");
}

void testReshapeOneDynamicDimension()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* reshape = graph.createNode("reshape", Operation::Reshape);
    input->shape = {2, 2}; reshape->shape = {-1, 2};
    graph.connect(input, reshape);
    auto result = emit(graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_CONTAINS(result.artifact->source, "(-1, 2)");
}

void testReshapeMultipleDynamicDimensionsRejected()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* reshape = graph.createNode("reshape", Operation::Reshape);
    input->shape = {2, 2}; reshape->shape = {-1, -1};
    graph.connect(input, reshape);
    auto plan = HybridRegionPlanner::build(graph);
    if(plan.succeeded())
    {
        auto result = RegionPythonEmitter::emit(graph, *plan.plan);
        EXPECT_TRUE(hasCode(result.diagnostics, "RPE004"));
    }
    else
    {
        EXPECT_TRUE(!plan.diagnostics.empty());
    }
}

void testInvalidOperationArityRejected()
{
    Graph graph;
    Node* a = graph.createNode("a", Operation::Input);
    Node* add = graph.createNode("add", Operation::Add);
    a->shape = add->shape = {1};
    graph.connect(a, add);
    auto plan = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(plan.succeeded());
    if(plan.succeeded())
    {
        auto result = RegionPythonEmitter::emit(graph, *plan.plan);
        EXPECT_TRUE(hasCode(result.diagnostics, "RPE003"));
    }
}

void testStalePlanRejectedThroughRPE001()
{
    Chain chain;
    auto plan = HybridRegionPlanner::build(chain.graph);
    chain.relu->shape = {-1, 4};
    auto result = RegionPythonEmitter::emit(chain.graph, *plan.plan);
    EXPECT_TRUE(hasCode(result.diagnostics, "RPE001"));
}

void testWrongSourceGraphRejectedThroughRPE001()
{
    Graph graph;
    auto plan = HybridRegionPlanner::build(graph);
    Graph other;
    auto result = RegionPythonEmitter::emit(other, *plan.plan);
    EXPECT_TRUE(hasCode(result.diagnostics, "RPE001"));
}

void testNoArtifactReturnedOnFailure()
{
    Graph graph;
    auto plan = HybridRegionPlanner::build(graph);
    Graph other;
    auto result = RegionPythonEmitter::emit(other, *plan.plan);
    EXPECT_FALSE(result.artifact.has_value());
}

void testDeterministicEmissionFiftyTimes()
{
    Chain chain;
    auto plan = HybridRegionPlanner::build(chain.graph);
    const auto first = RegionPythonEmitter::emit(
        chain.graph, *plan.plan).artifact->source;
    for(int iteration = 0; iteration < 50; ++iteration)
        EXPECT_EQ(first, RegionPythonEmitter::emit(
            chain.graph, *plan.plan).artifact->source);
}

void testGraphIndexVariableSymbols()
{
    Chain chain;
    chain.input->name = "same-name";
    chain.relu->name = "same name";
    auto result = emit(chain.graph);
    EXPECT_CONTAINS(result.artifact->source, "v_0");
    EXPECT_CONTAINS(result.artifact->source, "v_1");
}

void testArbitraryNodeNamePythonEscaping()
{
    Graph graph;
    Node* input = graph.createNode("a'b\\c\n\t", Operation::Input);
    input->shape = {1};
    auto result = emit(graph);
    EXPECT_CONTAINS(result.artifact->source, "'a\\'b\\\\c\\n\\t'");
}

void testNoPointerAddresses()
{
    Chain chain;
    EXPECT_TRUE(emit(chain.graph).artifact->source.find("0x") ==
                std::string::npos);
}

void testNoRandomInputGeneration()
{
    Chain chain;
    EXPECT_TRUE(emit(chain.graph).artifact->source.find("torch.randn") ==
                std::string::npos);
}

void testNoTritonContent()
{
    Chain chain;
    EXPECT_TRUE(emit(chain.graph).artifact->source.find("triton") ==
                std::string::npos);
}

void testGraphAndPlanNonMutation()
{
    Chain chain;
    auto plan = HybridRegionPlanner::build(chain.graph);
    const auto nodeCount = chain.graph.nodes.size();
    const auto regionCount = plan.plan->regionCount();
    RegionPythonEmitter::emit(chain.graph, *plan.plan);
    EXPECT_EQ(nodeCount, chain.graph.nodes.size());
    EXPECT_EQ(regionCount, plan.plan->regionCount());
}

void testDiagnosticPrinterEmptySnapshot()
{
    EXPECT_EQ(printed({}), "Region executor diagnostics\n  <none>\n");
}

void testDiagnosticPrinterMetadataSnapshot()
{
    RegionExecutionDiagnostic diagnostic{
        "RPE003", "expected 2 inputs but found 1", 1, "add"};
    EXPECT_EQ(
        printed({diagnostic}),
        "Region executor diagnostics\n"
        "  [RPE003] region=1 node=add: expected 2 inputs but found 1\n"
    );
}

void testDiagnosticPrinterOrderPreservation()
{
    const std::string output = printed({
        {"RPE001", "first", std::nullopt, std::nullopt},
        {"RPE005", "second", std::nullopt, std::nullopt}
    });
    EXPECT_TRUE(output.find("RPE001") < output.find("RPE005"));
}

void testOneThousandNodeEmissionScale()
{
    Graph graph;
    Node* previous = graph.createNode("input", Operation::Input);
    previous->shape = {1};
    for(int index = 1; index < 1000; ++index)
    {
        Node* next = graph.createNode(
            "node" + std::to_string(index),
            Operation::ReLU
        );
        next->shape = {1};
        graph.connect(previous, next);
        previous = next;
    }
    auto result = emit(graph);
    EXPECT_TRUE(result.succeeded());
    EXPECT_CONTAINS(result.artifact->source, "v_999 = torch.relu(v_998)");
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
    runTest("testResultSuccessContract", testResultSuccessContract);
    runTest("testEmptyGraphEmission", testEmptyGraphEmission);
    runTest("testSingleInputToOutput", testSingleInputToOutput);
    runTest("testStaticOptimizedChain", testStaticOptimizedChain);
    runTest("testAutomaticHybridChain", testAutomaticHybridChain);
    runTest("testOneFunctionPerRegion", testOneFunctionPerRegion);
    runTest("testStrategyComments", testStrategyComments);
    runTest("testRegionDefinitionsFollowStorageOrder", testRegionDefinitionsFollowStorageOrder);
    runTest("testOrchestratorFollowsDependencyOrder", testOrchestratorFollowsDependencyOrder);
    runTest("testRegionNodesFollowStoredExecutionOrder", testRegionNodesFollowStoredExecutionOrder);
    runTest("testExternalArgumentsFollowRegionInputOrder", testExternalArgumentsFollowRegionInputOrder);
    runTest("testRegionReturnsFollowOutputOrder", testRegionReturnsFollowOutputOrder);
    runTest("testMultipleCrossingValues", testMultipleCrossingValues);
    runTest("testSharedCrossingValuePassedToTwoRegions", testSharedCrossingValuePassedToTwoRegions);
    runTest("testDisconnectedComponents", testDisconnectedComponents);
    runTest("testScalarConstant", testScalarConstant);
    runTest("testShapedConstant", testShapedConstant);
    runTest("testMultipleSemanticOutputs", testMultipleSemanticOutputs);
    runTest("testOutputAliasesOperandValue", testOutputAliasesOperandValue);
    runTest("testInputAndFallbackOperationInOneRegion", testInputAndFallbackOperationInOneRegion);
    runTest("testOutputAndFallbackOperationInOneRegion", testOutputAndFallbackOperationInOneRegion);
    runTest("testCompleteOperationLoweringMatrix", testCompleteOperationLoweringMatrix);
    runTest("testFusedMatMulReluLowering", testFusedMatMulReluLowering);
    runTest("testReshapeOneDynamicDimension", testReshapeOneDynamicDimension);
    runTest("testReshapeMultipleDynamicDimensionsRejected", testReshapeMultipleDynamicDimensionsRejected);
    runTest("testInvalidOperationArityRejected", testInvalidOperationArityRejected);
    runTest("testStalePlanRejectedThroughRPE001", testStalePlanRejectedThroughRPE001);
    runTest("testWrongSourceGraphRejectedThroughRPE001", testWrongSourceGraphRejectedThroughRPE001);
    runTest("testNoArtifactReturnedOnFailure", testNoArtifactReturnedOnFailure);
    runTest("testDeterministicEmissionFiftyTimes", testDeterministicEmissionFiftyTimes);
    runTest("testGraphIndexVariableSymbols", testGraphIndexVariableSymbols);
    runTest("testArbitraryNodeNamePythonEscaping", testArbitraryNodeNamePythonEscaping);
    runTest("testNoPointerAddresses", testNoPointerAddresses);
    runTest("testNoRandomInputGeneration", testNoRandomInputGeneration);
    runTest("testNoTritonContent", testNoTritonContent);
    runTest("testGraphAndPlanNonMutation", testGraphAndPlanNonMutation);
    runTest("testDiagnosticPrinterEmptySnapshot", testDiagnosticPrinterEmptySnapshot);
    runTest("testDiagnosticPrinterMetadataSnapshot", testDiagnosticPrinterMetadataSnapshot);
    runTest("testDiagnosticPrinterOrderPreservation", testDiagnosticPrinterOrderPreservation);
    runTest("testOneThousandNodeEmissionScale", testOneThousandNodeEmissionScale);

    if(failures != 0)
    {
        std::cerr << failures << " Region Python Emitter test(s) failed\n";
        return 1;
    }
    std::cout << "All Region Python Emitter tests passed\n";
    return 0;
}
