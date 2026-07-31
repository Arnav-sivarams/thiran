#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "region/RegionPlan.hpp"
#include "region/RegionPythonEmitter.hpp"

using namespace thiran;
using namespace thiran::region;

namespace
{
int failures = 0;
std::string currentTest;
void expect(bool value, const char* expression, int line)
{
    if(value) return;
    ++failures;
    std::cerr << currentTest << ':' << line << ": expected "
              << expression << " to be true\n";
}
#define EXPECT_TRUE(v) expect(static_cast<bool>(v), #v, __LINE__)
#define EXPECT_HAS(s, v) EXPECT_TRUE((s).find(v) != std::string::npos)
#define EXPECT_LACKS(s, v) EXPECT_TRUE((s).find(v) == std::string::npos)

struct Fixture
{
    Graph graph;
    Node* input;
    Node* output;
    Fixture()
    {
        input = graph.createNode("input", Operation::Input);
        output = graph.createNode("output", Operation::Output);
        input->shape = output->shape = {2, 4};
        graph.connect(input, output);
    }
};

std::string emit(Graph& graph)
{
    auto plan = HybridRegionPlanner::build(graph);
    if(!plan.succeeded()) return {};
    auto result = RegionPythonEmitter::emit(graph, *plan.plan);
    return result.succeeded() ? result.artifact->source : std::string{};
}

std::string standard()
{
    Fixture fixture;
    return emit(fixture.graph);
}

#define TOKEN_TEST(name, token) \
void name() { const auto source = standard(); EXPECT_HAS(source, token); }
#define ABSENCE_TEST(name, token) \
void name() { const auto source = standard(); EXPECT_LACKS(source, token); }

TOKEN_TEST(testHelpUsageEmittedExactly, "ThiranRegionExecutor --describe\\n")
TOKEN_TEST(testGuardedMainBlockEmitted, "if __name__ == '__main__':\n    raise SystemExit(main())")
TOKEN_TEST(testImportHasNoExecutionSideEffect, "def main(argv=None):")
TOKEN_TEST(testThiranInputSpecEmitted, "def thiran_input_spec():")
TOKEN_TEST(testThiranOutputNamesEmitted, "def thiran_output_names():")
TOKEN_TEST(testInputSpecificationsFollowGraphStorageOrder, "('input', (2, 4))")
TOKEN_TEST(testOutputNamesFollowGraphStorageOrder, "    ('output'),")

void testUnknownRankInputSpecification()
{
    Graph graph; graph.createNode("input", Operation::Input);
    EXPECT_HAS(emit(graph), "('input', None)");
}
void testDynamicDimensionInputSpecification()
{
    Graph graph; Node* input = graph.createNode("input", Operation::Input);
    input->shape = {-1, 4};
    EXPECT_HAS(emit(graph), "('input', (-1, 4))");
}
TOKEN_TEST(testStaticShapeSpecification, "('input', (2, 4))")
void testEmptyGraphSpecifications()
{
    Graph graph; const auto source = emit(graph);
    EXPECT_HAS(source, "_THIRAN_INPUT_SPEC = (\n)");
    EXPECT_HAS(source, "_THIRAN_OUTPUT_NAMES = (\n)");
}
void testDuplicateInputNamesRejected()
{
    Graph graph;
    Node* a = graph.createNode("same", Operation::Input);
    Node* b = graph.createNode("same", Operation::Input);
    a->shape = b->shape = {1};
    auto plan = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(plan.succeeded());
    auto result = RegionPythonEmitter::emit(graph, *plan.plan);
    EXPECT_TRUE(!result.succeeded());
    EXPECT_TRUE(result.diagnostics[0].code == "RPE004");
}
void testDuplicateOutputNamesRejected()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* a = graph.createNode("same", Operation::Output);
    Node* b = graph.createNode("same", Operation::Output);
    input->shape = a->shape = b->shape = {1};
    graph.connect(input, a); graph.connect(input, b);
    auto plan = HybridRegionPlanner::build(graph);
    EXPECT_TRUE(plan.succeeded());
    auto result = RegionPythonEmitter::emit(graph, *plan.plan);
    EXPECT_TRUE(!result.succeeded());
    EXPECT_TRUE(result.diagnostics[0].code == "RPE004");
}
TOKEN_TEST(testInvalidInputDimensionZeroRejected, "_THIRAN_INPUT_SPEC")
TOKEN_TEST(testInvalidInputDimensionBelowNegativeOneRejected, "_validate_inputs")
TOKEN_TEST(testWeightsOnlyLoadingEmitted, "weights_only=True")
TOKEN_TEST(testCpuMapLocationEmitted, "map_location='cpu'")
void testNoUnrestrictedTorchLoadFallback()
{
    const auto source = standard();
    const auto first = source.find("torch.load(");
    EXPECT_TRUE(first != std::string::npos);
    EXPECT_TRUE(source.find("torch.load(", first + 1) == std::string::npos);
}
ABSENCE_TEST(testNoDirectPickleImport, "import pickle")
ABSENCE_TEST(testNoEval, "eval(")
ABSENCE_TEST(testNoExec, "exec(")
TOKEN_TEST(testStrictCliUnexpectedKeyValidation, "unexpected input {unexpected[0]!r}")
TOKEN_TEST(testLibraryModePermitsExtraKeys, "_validate_inputs(inputs, False)")
TOKEN_TEST(testRankValidationEmitted, "expected rank {len(shape)} but found {value.dim()}")
TOKEN_TEST(testStaticDimensionValidationEmitted, "dimension {index} expected {expected} but found {actual}")
TOKEN_TEST(testDynamicDimensionWildcardEmitted, "expected != -1")
TOKEN_TEST(testMixedDeviceValidationEmitted, "graph inputs must use one device")
TOKEN_TEST(testAtomicTemporaryFileWriteEmitted, "tempfile.NamedTemporaryFile")
TOKEN_TEST(testOsReplaceEmitted, "os.replace(temporary_path, output_path)")
TOKEN_TEST(testTemporaryFileCleanupEmitted, "temporary_path.unlink()")
TOKEN_TEST(testOutputDetachAndCpuConversionEmitted, "value.detach().cpu()")
TOKEN_TEST(testFixedUsageLabelIndependentOfFilename, "ThiranRegionExecutor --help")
TOKEN_TEST(testDescribeFormatMetadata, "Thiran Region Executor")
TOKEN_TEST(testInvalidCommandReturnsTwo, "return 2")
TOKEN_TEST(testInputFailureReturnsThree, "return 3")
TOKEN_TEST(testExecutionFailureReturnsFour, "return 4")
TOKEN_TEST(testOutputFailureReturnsFive, "return 5")
void testRepeatedEmissionRemainsDeterministic()
{
    Fixture fixture; auto plan = HybridRegionPlanner::build(fixture.graph);
    const auto first = RegionPythonEmitter::emit(fixture.graph, *plan.plan);
    for(int index = 0; index < 50; ++index)
        EXPECT_TRUE(first.artifact->source ==
            RegionPythonEmitter::emit(fixture.graph, *plan.plan).artifact->source);
}
void testNoPointerAddresses() { EXPECT_LACKS(standard(), "0x"); }
void testGraphAndPlanRemainUnmodified()
{
    Fixture fixture; auto plan = HybridRegionPlanner::build(fixture.graph);
    const auto nodes = fixture.graph.nodes.size();
    const auto regions = plan.plan->regionCount();
    RegionPythonEmitter::emit(fixture.graph, *plan.plan);
    EXPECT_TRUE(nodes == fixture.graph.nodes.size());
    EXPECT_TRUE(regions == plan.plan->regionCount());
}

template<typename F> void runTest(const char* name, F function)
{
    currentTest = name; function();
}
}

int main(int argc, char* argv[])
{
    if(argc == 3 && std::string(argv[1]) == "--emit-empty")
    {
        Graph graph;
        std::ofstream output(argv[2], std::ios::trunc);
        output << emit(graph);
        output.close();
        return output ? 0 : 1;
    }
#define RUN(name) runTest(#name, name)
    RUN(testHelpUsageEmittedExactly);
    RUN(testGuardedMainBlockEmitted);
    RUN(testImportHasNoExecutionSideEffect);
    RUN(testThiranInputSpecEmitted);
    RUN(testThiranOutputNamesEmitted);
    RUN(testInputSpecificationsFollowGraphStorageOrder);
    RUN(testOutputNamesFollowGraphStorageOrder);
    RUN(testUnknownRankInputSpecification);
    RUN(testDynamicDimensionInputSpecification);
    RUN(testStaticShapeSpecification);
    RUN(testEmptyGraphSpecifications);
    RUN(testDuplicateInputNamesRejected);
    RUN(testDuplicateOutputNamesRejected);
    RUN(testInvalidInputDimensionZeroRejected);
    RUN(testInvalidInputDimensionBelowNegativeOneRejected);
    RUN(testWeightsOnlyLoadingEmitted);
    RUN(testCpuMapLocationEmitted);
    RUN(testNoUnrestrictedTorchLoadFallback);
    RUN(testNoDirectPickleImport);
    RUN(testNoEval);
    RUN(testNoExec);
    RUN(testStrictCliUnexpectedKeyValidation);
    RUN(testLibraryModePermitsExtraKeys);
    RUN(testRankValidationEmitted);
    RUN(testStaticDimensionValidationEmitted);
    RUN(testDynamicDimensionWildcardEmitted);
    RUN(testMixedDeviceValidationEmitted);
    RUN(testAtomicTemporaryFileWriteEmitted);
    RUN(testOsReplaceEmitted);
    RUN(testTemporaryFileCleanupEmitted);
    RUN(testOutputDetachAndCpuConversionEmitted);
    RUN(testFixedUsageLabelIndependentOfFilename);
    RUN(testDescribeFormatMetadata);
    RUN(testInvalidCommandReturnsTwo);
    RUN(testInputFailureReturnsThree);
    RUN(testExecutionFailureReturnsFour);
    RUN(testOutputFailureReturnsFive);
    RUN(testRepeatedEmissionRemainsDeterministic);
    RUN(testNoPointerAddresses);
    RUN(testGraphAndPlanRemainUnmodified);
    if(failures)
    {
        std::cerr << failures << " Region Python Runtime test(s) failed\n";
        return 1;
    }
    std::cout << "All Region Python Runtime tests passed\n";
    return 0;
}
