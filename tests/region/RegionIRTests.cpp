#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ir/Graph.hpp"
#include "region/RegionGraph.hpp"
#include "region/RegionVerifier.hpp"
#include "utils/RegionPrinter.hpp"

namespace
{

using thiran::Graph;
using thiran::Node;
using thiran::Operation;
using thiran::RegionPrinter;
using thiran::region::RegionGraph;
using thiran::region::RegionId;
using thiran::region::RegionStrategy;
using thiran::region::RegionVerificationLevel;
using thiran::region::RegionVerificationResult;
using thiran::region::RegionVerifier;

int failures = 0;
std::string currentTest;

template<typename Expected, typename Actual>
void expectEqual(
    const Expected& expected,
    const Actual& actual,
    const char* expectedExpression,
    const char* actualExpression,
    int line
)
{
    if(expected == actual)
    {
        return;
    }

    ++failures;
    std::cerr << currentTest << ":" << line
              << ": expected " << expectedExpression << " = " << expected
              << ", actual " << actualExpression << " = " << actual
              << "\n";
}

void expectTrue(
    bool actual,
    const char* expression,
    int line
)
{
    if(actual)
    {
        return;
    }

    ++failures;
    std::cerr << currentTest << ":" << line
              << ": expected " << expression << " = true, actual false\n";
}

void expectFalse(
    bool actual,
    const char* expression,
    int line
)
{
    if(!actual)
    {
        return;
    }

    ++failures;
    std::cerr << currentTest << ":" << line
              << ": expected " << expression << " = false, actual true\n";
}

void expectPointerEqual(
    const void* expected,
    const void* actual,
    const char* expectedExpression,
    const char* actualExpression,
    int line
)
{
    if(expected == actual)
    {
        return;
    }

    ++failures;
    std::cerr << currentTest << ":" << line
              << ": expected " << expectedExpression << " = " << expected
              << ", actual " << actualExpression << " = " << actual
              << "\n";
}

std::size_t diagnosticCount(
    const RegionVerificationResult& result,
    std::string_view code
)
{
    std::size_t count = 0;
    for(const auto& diagnostic : result.diagnostics)
    {
        if(diagnostic.code == code)
        {
            ++count;
        }
    }
    return count;
}

std::size_t diagnosticIndex(
    const RegionVerificationResult& result,
    std::string_view code
)
{
    for(std::size_t i = 0; i < result.diagnostics.size(); ++i)
    {
        if(result.diagnostics[i].code == code)
        {
            return i;
        }
    }
    return result.diagnostics.size();
}

void expectContainsDiagnosticCode(
    const RegionVerificationResult& result,
    std::string_view code,
    int line
)
{
    if(diagnosticCount(result, code) != 0)
    {
        return;
    }

    ++failures;
    std::cerr << currentTest << ":" << line
              << ": expected diagnostic code " << code
              << ", actual diagnostics did not contain it\n";
}

#define EXPECT_TRUE(expression) \
    expectTrue((expression), #expression, __LINE__)
#define EXPECT_FALSE(expression) \
    expectFalse((expression), #expression, __LINE__)
#define EXPECT_EQ(expected, actual) \
    expectEqual((expected), (actual), #expected, #actual, __LINE__)
#define EXPECT_PTR_EQ(expected, actual) \
    expectPointerEqual((expected), (actual), #expected, #actual, __LINE__)
#define EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, code) \
    expectContainsDiagnosticCode((result), (code), __LINE__)

void testRegionStrategyStrings()
{
    EXPECT_EQ(
        std::string_view("UNRESOLVED"),
        thiran::region::toString(RegionStrategy::Unresolved)
    );
    EXPECT_EQ(
        std::string_view("AOT"),
        thiran::region::toString(RegionStrategy::AheadOfTime)
    );
    EXPECT_EQ(
        std::string_view("JIT"),
        thiran::region::toString(RegionStrategy::JustInTime)
    );
    EXPECT_EQ(
        std::string_view("FALLBACK"),
        thiran::region::toString(RegionStrategy::Fallback)
    );
    EXPECT_EQ(
        std::string_view("UNKNOWN"),
        thiran::region::toString(static_cast<RegionStrategy>(255))
    );
}

void testEmptyConstruction()
{
    Graph graph;
    graph.createNode("input", Operation::Input);
    RegionGraph regionGraph(graph);

    EXPECT_EQ(std::size_t(0), regionGraph.size());
    EXPECT_TRUE(regionGraph.empty());
    EXPECT_FALSE(regionGraph.finalized());
    EXPECT_TRUE(regionGraph.dependencies().empty());

    const auto result = RegionVerifier::verify(graph, regionGraph);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG002");
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG003");

    Graph otherGraph;
    const auto mismatch = RegionVerifier::verify(otherGraph, regionGraph);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(mismatch, "RG001");
}

void testDeterministicRegionIds()
{
    Graph graph;
    RegionGraph regionGraph(graph);
    auto& first = regionGraph.createRegion();
    auto& second = regionGraph.createRegion();
    auto& third = regionGraph.createRegion();

    EXPECT_EQ(RegionId(0), first.id());
    EXPECT_EQ(RegionId(1), second.id());
    EXPECT_EQ(RegionId(2), third.id());
    EXPECT_PTR_EQ(&first, regionGraph.findRegion(0));
    EXPECT_PTR_EQ(&second, regionGraph.findRegion(1));
    EXPECT_PTR_EQ(&third, regionGraph.findRegion(2));
    EXPECT_PTR_EQ(nullptr, regionGraph.findRegion(99));
}

void testNodeAssignmentRules()
{
    Graph graph;
    Node* first = graph.createNode("first", Operation::Input);
    Node* second = graph.createNode("second", Operation::Input);
    Graph foreignGraph;
    Node* foreign = foreignGraph.createNode("first", Operation::Input);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();

    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *first));
    EXPECT_FALSE(regionGraph.appendNode(region0.id(), *first));
    EXPECT_FALSE(regionGraph.appendNode(region1.id(), *first));
    EXPECT_EQ(std::size_t(1), region0.size());
    EXPECT_TRUE(region1.empty());
    EXPECT_FALSE(regionGraph.appendNode(99, *second));
    EXPECT_EQ(std::size_t(1), region0.size());
    EXPECT_TRUE(region1.empty());
    EXPECT_FALSE(regionGraph.appendNode(region1.id(), *foreign));
    EXPECT_EQ(std::size_t(1), region0.size());
    EXPECT_TRUE(region1.empty());
    EXPECT_TRUE(region0.contains(*first));
    EXPECT_FALSE(region0.contains(*foreign));
    EXPECT_EQ(std::size_t(1), region0.size());
    EXPECT_FALSE(region0.empty());
    EXPECT_TRUE(region1.empty());
    EXPECT_PTR_EQ(&region0, regionGraph.regionFor(*first));
    EXPECT_PTR_EQ(nullptr, regionGraph.regionFor(*second));
}

void testSingleRegionStructuralGraph()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* relu = graph.createNode("relu", Operation::ReLU);
    Node* output = graph.createNode("output", Operation::Output);
    graph.connect(input, relu);
    graph.connect(relu, output);

    RegionGraph regionGraph(graph);
    auto& region = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *input));
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *relu));
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *output));
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_EQ(std::size_t(3), region.nodes().size());
    EXPECT_PTR_EQ(input, region.nodes()[0]);
    EXPECT_PTR_EQ(relu, region.nodes()[1]);
    EXPECT_PTR_EQ(output, region.nodes()[2]);
    EXPECT_TRUE(region.inputs().empty());
    EXPECT_EQ(std::size_t(1), region.outputs().size());
    EXPECT_PTR_EQ(output, region.outputs()[0]);
    EXPECT_TRUE(regionGraph.dependencies().empty());
    EXPECT_TRUE(RegionVerifier::verify(graph, regionGraph).valid);

    const auto unresolved = RegionVerifier::verify(
        graph,
        regionGraph,
        RegionVerificationLevel::Executable
    );
    EXPECT_FALSE(unresolved.valid);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(unresolved, "RG022");

    EXPECT_TRUE(regionGraph.setStrategy(
        region.id(),
        RegionStrategy::AheadOfTime
    ));
    EXPECT_TRUE(RegionVerifier::verify(
        graph,
        regionGraph,
        RegionVerificationLevel::Executable
    ).valid);
}

void testTwoRegionLinearBoundary()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* relu = graph.createNode("relu", Operation::ReLU);
    Node* output = graph.createNode("output", Operation::Output);
    graph.connect(input, relu);
    graph.connect(relu, output);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *input));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *relu));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *output));
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_TRUE(region0.inputs().empty());
    EXPECT_EQ(std::size_t(1), region0.outputs().size());
    EXPECT_PTR_EQ(input, region0.outputs()[0]);
    EXPECT_EQ(std::size_t(1), region1.inputs().size());
    EXPECT_PTR_EQ(input, region1.inputs()[0]);
    EXPECT_EQ(std::size_t(1), region1.outputs().size());
    EXPECT_PTR_EQ(output, region1.outputs()[0]);

    EXPECT_EQ(std::size_t(1), regionGraph.dependencies().size());
    const auto& dependency = regionGraph.dependencies()[0];
    EXPECT_EQ(RegionId(0), dependency.source);
    EXPECT_EQ(RegionId(1), dependency.destination);
    EXPECT_EQ(std::size_t(1), dependency.values.size());
    EXPECT_PTR_EQ(input, dependency.values[0]);
    EXPECT_TRUE(RegionVerifier::verify(graph, regionGraph).valid);
}

void testInputDeduplication()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* reluA = graph.createNode("reluA", Operation::ReLU);
    Node* reluB = graph.createNode("reluB", Operation::ReLU);
    graph.connect(input, reluA);
    graph.connect(input, reluB);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *input));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *reluA));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *reluB));
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_EQ(std::size_t(1), region1.inputs().size());
    EXPECT_PTR_EQ(input, region1.inputs()[0]);
    EXPECT_EQ(std::size_t(1), regionGraph.dependencies().size());
    EXPECT_EQ(
        std::size_t(1),
        regionGraph.dependencies()[0].values.size()
    );
    EXPECT_PTR_EQ(input, regionGraph.dependencies()[0].values[0]);
    EXPECT_EQ(std::size_t(1), region0.outputs().size());
    EXPECT_PTR_EQ(input, region0.outputs()[0]);
}

std::string multipleCrossingSnapshot()
{
    Graph graph;
    Node* producerB = graph.createNode("producerB", Operation::Input);
    Node* producerA = graph.createNode("producerA", Operation::Input);
    Node* reluA = graph.createNode("reluA", Operation::ReLU);
    Node* reluB = graph.createNode("reluB", Operation::ReLU);
    graph.connect(producerA, reluA);
    graph.connect(producerB, reluB);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    regionGraph.appendNode(region0.id(), *producerA);
    regionGraph.appendNode(region0.id(), *producerB);
    regionGraph.appendNode(region1.id(), *reluA);
    regionGraph.appendNode(region1.id(), *reluB);
    regionGraph.finalize();

    const auto& dependency = regionGraph.dependencies()[0];
    std::ostringstream output;
    output << dependency.source << "->" << dependency.destination;
    for(const Node* value : dependency.values)
    {
        output << ":" << value->name;
    }
    return output.str();
}

void testMultipleCrossingValues()
{
    Graph graph;
    Node* producerB = graph.createNode("producerB", Operation::Input);
    Node* producerA = graph.createNode("producerA", Operation::Input);
    Node* reluA = graph.createNode("reluA", Operation::ReLU);
    Node* reluB = graph.createNode("reluB", Operation::ReLU);
    graph.connect(producerA, reluA);
    graph.connect(producerB, reluB);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *producerA));
    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *producerB));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *reluA));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *reluB));
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_EQ(std::size_t(1), regionGraph.dependencies().size());
    const auto& dependency = regionGraph.dependencies()[0];
    EXPECT_EQ(RegionId(0), dependency.source);
    EXPECT_EQ(RegionId(1), dependency.destination);
    EXPECT_EQ(std::size_t(2), dependency.values.size());
    EXPECT_PTR_EQ(producerB, dependency.values[0]);
    EXPECT_PTR_EQ(producerA, dependency.values[1]);
    EXPECT_EQ(multipleCrossingSnapshot(), multipleCrossingSnapshot());
}

void testFinalizationFreezesStructure()
{
    Graph graph;
    Node* first = graph.createNode("first", Operation::Input);
    Node* second = graph.createNode("second", Operation::Input);
    RegionGraph regionGraph(graph);
    auto& region = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *first));
    EXPECT_TRUE(regionGraph.finalize());

    const std::size_t regionCount = regionGraph.size();
    const std::size_t memberCount = region.size();
    EXPECT_FALSE(regionGraph.appendNode(region.id(), *second));
    EXPECT_FALSE(regionGraph.finalize());
    EXPECT_EQ(regionCount, regionGraph.size());
    EXPECT_EQ(memberCount, region.size());
    EXPECT_PTR_EQ(nullptr, regionGraph.regionFor(*second));
    EXPECT_TRUE(regionGraph.setStrategy(
        region.id(),
        RegionStrategy::Fallback
    ));
}

void testEmptyRegionDiagnostic()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *input));
    EXPECT_TRUE(regionGraph.finalize());

    const auto result = RegionVerifier::verify(graph, regionGraph);
    EXPECT_FALSE(result.valid);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG005");
}

void testMissingGraphNodeDiagnostic()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    graph.createNode("missing", Operation::Input);
    RegionGraph regionGraph(graph);
    auto& region = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *input));
    EXPECT_TRUE(regionGraph.finalize());

    const auto result = RegionVerifier::verify(graph, regionGraph);
    EXPECT_FALSE(result.valid);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG010");
}

void testNonTopologicalRegionOrder()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* relu = graph.createNode("relu", Operation::ReLU);
    graph.connect(input, relu);

    RegionGraph regionGraph(graph);
    auto& region = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *relu));
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *input));
    EXPECT_TRUE(regionGraph.finalize());

    const auto result = RegionVerifier::verify(graph, regionGraph);
    EXPECT_FALSE(result.valid);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG011");
}

void testCyclicRegionGraphFromNonConvexPartition()
{
    Graph graph;
    Node* a = graph.createNode("A", Operation::Input);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* c = graph.createNode("C", Operation::ReLU);
    graph.connect(a, b);
    graph.connect(b, c);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *a));
    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *c));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *b));
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_EQ(std::size_t(2), regionGraph.dependencies().size());
    const auto result = RegionVerifier::verify(graph, regionGraph);
    EXPECT_FALSE(result.valid);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG021");
}

void testMultipleUnresolvedStrategies()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* output = graph.createNode("output", Operation::Output);
    graph.connect(input, output);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    regionGraph.appendNode(region0.id(), *input);
    regionGraph.appendNode(region1.id(), *output);
    regionGraph.finalize();

    const auto unresolved = RegionVerifier::verify(
        graph,
        regionGraph,
        RegionVerificationLevel::Executable
    );
    EXPECT_EQ(std::size_t(2), diagnosticCount(unresolved, "RG022"));

    std::vector<RegionId> unresolvedIds;
    for(const auto& diagnostic : unresolved.diagnostics)
    {
        if(diagnostic.code == "RG022" && diagnostic.regionId.has_value())
        {
            unresolvedIds.push_back(*diagnostic.regionId);
        }
    }
    EXPECT_EQ(std::size_t(2), unresolvedIds.size());
    EXPECT_EQ(RegionId(0), unresolvedIds[0]);
    EXPECT_EQ(RegionId(1), unresolvedIds[1]);

    EXPECT_TRUE(regionGraph.setStrategy(
        region0.id(),
        RegionStrategy::AheadOfTime
    ));
    EXPECT_TRUE(regionGraph.setStrategy(
        region1.id(),
        RegionStrategy::Fallback
    ));
    EXPECT_TRUE(RegionVerifier::verify(
        graph,
        regionGraph,
        RegionVerificationLevel::Executable
    ).valid);
}

std::string linearRegionPrint()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* relu = graph.createNode("relu", Operation::ReLU);
    Node* output = graph.createNode("output", Operation::Output);
    graph.connect(input, relu);
    graph.connect(relu, output);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    regionGraph.appendNode(region0.id(), *input);
    regionGraph.appendNode(region1.id(), *relu);
    regionGraph.appendNode(region1.id(), *output);
    regionGraph.setStrategy(region0.id(), RegionStrategy::AheadOfTime);
    regionGraph.setStrategy(region1.id(), RegionStrategy::JustInTime);
    regionGraph.finalize();

    std::ostringstream outputStream;
    RegionPrinter::print(regionGraph, outputStream);
    return outputStream.str();
}

void testPrinterSnapshot()
{
    const std::string expected =
        "RegionGraph\n"
        "  Regions: 2\n"
        "  Dependencies: 1\n"
        "\n"
        "Region 0 [AOT]\n"
        "  Nodes: input\n"
        "  Inputs: <none>\n"
        "  Outputs: input\n"
        "\n"
        "Region 1 [JIT]\n"
        "  Nodes: relu, output\n"
        "  Inputs: input\n"
        "  Outputs: output\n"
        "\n"
        "Dependencies\n"
        "  0 -> 1: input\n";

    EXPECT_EQ(expected, linearRegionPrint());
}

void testRepeatability()
{
    const std::string expected = linearRegionPrint();
    for(int iteration = 0; iteration < 20; ++iteration)
    {
        EXPECT_EQ(expected, linearRegionPrint());
    }

    const std::string dependencyExpected = multipleCrossingSnapshot();
    for(int iteration = 0; iteration < 20; ++iteration)
    {
        EXPECT_EQ(dependencyExpected, multipleCrossingSnapshot());
    }
}

void testSourceGraphMismatch()
{
    Graph graphA;
    Node* inputA = graphA.createNode("input", Operation::Input);
    RegionGraph regionGraph(graphA);
    auto& region = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *inputA));
    EXPECT_TRUE(regionGraph.finalize());

    Graph graphB;
    graphB.createNode("input", Operation::Input);
    const auto result = RegionVerifier::verify(graphB, regionGraph);

    EXPECT_FALSE(result.valid);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG001");
    EXPECT_EQ(result.diagnostics.empty(), result.valid);
}

void testNotFinalized()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    RegionGraph regionGraph(graph);
    auto& region = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *input));

    const auto result = RegionVerifier::verify(graph, regionGraph);
    EXPECT_FALSE(result.valid);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG002");
    EXPECT_EQ(result.diagnostics.empty(), result.valid);
}

void testNonemptyGraphWithZeroRegions()
{
    Graph graph;
    graph.createNode("first", Operation::Input);
    graph.createNode("second", Operation::Input);
    RegionGraph regionGraph(graph);
    EXPECT_TRUE(regionGraph.finalize());

    const auto result = RegionVerifier::verify(graph, regionGraph);
    EXPECT_FALSE(result.valid);
    EXPECT_CONTAINS_DIAGNOSTIC_CODE(result, "RG003");
    EXPECT_EQ(std::size_t(2), diagnosticCount(result, "RG010"));
    EXPECT_TRUE(
        diagnosticIndex(result, "RG003") <
        diagnosticIndex(result, "RG010")
    );
    EXPECT_EQ(result.diagnostics.empty(), result.valid);
}

void testEmptyGraphWithZeroRegions()
{
    Graph graph;
    RegionGraph regionGraph(graph);
    EXPECT_TRUE(regionGraph.finalize());

    const auto result = RegionVerifier::verify(graph, regionGraph);
    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.diagnostics.empty());
    EXPECT_TRUE(regionGraph.dependencies().empty());

    std::ostringstream output;
    RegionPrinter::print(regionGraph, output);
    EXPECT_EQ(
        std::string(
            "RegionGraph\n"
            "  Regions: 0\n"
            "  Dependencies: 0\n"
            "\n"
            "Dependencies\n"
            "  <none>\n"
        ),
        output.str()
    );
}

void testUnknownStrategyRegionId()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    RegionGraph regionGraph(graph);
    auto& region = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region.id(), *input));

    EXPECT_FALSE(regionGraph.setStrategy(99, RegionStrategy::Fallback));
    EXPECT_EQ(
        static_cast<int>(RegionStrategy::Unresolved),
        static_cast<int>(region.strategy())
    );
    EXPECT_EQ(std::size_t(1), regionGraph.size());
    EXPECT_PTR_EQ(&region, regionGraph.regionFor(*input));

    EXPECT_TRUE(regionGraph.setStrategy(
        region.id(),
        RegionStrategy::AheadOfTime
    ));
    EXPECT_EQ(
        static_cast<int>(RegionStrategy::AheadOfTime),
        static_cast<int>(region.strategy())
    );
}

void testRegionLookup()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* relu = graph.createNode("relu", Operation::ReLU);
    Node* output = graph.createNode("output", Operation::Output);
    graph.connect(input, relu);
    graph.connect(relu, output);

    Graph foreignGraph;
    Node* foreign = foreignGraph.createNode("input", Operation::Input);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    EXPECT_TRUE(regionGraph.appendNode(region0.id(), *input));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *relu));
    EXPECT_TRUE(regionGraph.appendNode(region1.id(), *output));
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_PTR_EQ(&region0, regionGraph.regionFor(*input));
    EXPECT_PTR_EQ(&region1, regionGraph.regionFor(*relu));
    EXPECT_PTR_EQ(&region1, regionGraph.regionFor(*output));
    EXPECT_PTR_EQ(&region0, regionGraph.findRegion(0));
    EXPECT_PTR_EQ(&region1, regionGraph.findRegion(1));
    EXPECT_PTR_EQ(nullptr, regionGraph.findRegion(2));
    EXPECT_PTR_EQ(nullptr, regionGraph.regionFor(*foreign));
}

void testOperandOrderVersusDependencyValueOrder()
{
    Graph graph;
    Node* a = graph.createNode("A", Operation::Input);
    Node* b = graph.createNode("B", Operation::Input);
    Node* destination = graph.createNode("destination", Operation::Add);
    graph.connect(b, destination);
    graph.connect(a, destination);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    regionGraph.appendNode(region0.id(), *a);
    regionGraph.appendNode(region0.id(), *b);
    regionGraph.appendNode(region1.id(), *destination);
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_EQ(std::size_t(2), region1.inputs().size());
    EXPECT_PTR_EQ(b, region1.inputs()[0]);
    EXPECT_PTR_EQ(a, region1.inputs()[1]);
    EXPECT_EQ(std::size_t(1), regionGraph.dependencies().size());
    EXPECT_EQ(
        std::size_t(2),
        regionGraph.dependencies()[0].values.size()
    );
    EXPECT_PTR_EQ(a, regionGraph.dependencies()[0].values[0]);
    EXPECT_PTR_EQ(b, regionGraph.dependencies()[0].values[1]);
}

void testDependencyLexicographicOrder()
{
    Graph graph;
    Node* a = graph.createNode("A", Operation::Input);
    Node* b = graph.createNode("B", Operation::Input);
    Node* middle = graph.createNode("middle", Operation::ReLU);
    Node* final = graph.createNode("final", Operation::Add);
    graph.connect(a, middle);
    graph.connect(middle, final);
    graph.connect(b, final);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    auto& region2 = regionGraph.createRegion();
    regionGraph.appendNode(region0.id(), *a);
    regionGraph.appendNode(region0.id(), *b);
    regionGraph.appendNode(region1.id(), *middle);
    regionGraph.appendNode(region2.id(), *final);
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_EQ(std::size_t(3), regionGraph.dependencies().size());
    EXPECT_EQ(RegionId(0), regionGraph.dependencies()[0].source);
    EXPECT_EQ(RegionId(1), regionGraph.dependencies()[0].destination);
    EXPECT_EQ(RegionId(0), regionGraph.dependencies()[1].source);
    EXPECT_EQ(RegionId(2), regionGraph.dependencies()[1].destination);
    EXPECT_EQ(RegionId(1), regionGraph.dependencies()[2].source);
    EXPECT_EQ(RegionId(2), regionGraph.dependencies()[2].destination);

    std::ostringstream output;
    RegionPrinter::print(regionGraph, output);
    const std::string printed = output.str();
    const auto first = printed.find("  0 -> 1: A\n");
    const auto second = printed.find("  0 -> 2: B\n");
    const auto third = printed.find("  1 -> 2: middle\n");
    EXPECT_TRUE(first != std::string::npos);
    EXPECT_TRUE(second != std::string::npos);
    EXPECT_TRUE(third != std::string::npos);
    EXPECT_TRUE(first < second);
    EXPECT_TRUE(second < third);
}

void testMovedRegionGraphPreservesValidity()
{
    Graph graph;
    Node* input = graph.createNode("input", Operation::Input);
    Node* output = graph.createNode("output", Operation::Output);
    graph.connect(input, output);

    RegionGraph original(graph);
    auto& region0 = original.createRegion();
    auto& region1 = original.createRegion();
    original.appendNode(region0.id(), *input);
    original.appendNode(region1.id(), *output);
    original.setStrategy(region0.id(), RegionStrategy::AheadOfTime);
    original.setStrategy(region1.id(), RegionStrategy::Fallback);
    EXPECT_TRUE(original.finalize());

    std::ostringstream before;
    RegionPrinter::print(original, before);
    const std::size_t originalRegionCount = original.size();
    const std::size_t originalDependencyCount =
        original.dependencies().size();

    RegionGraph moved(std::move(original));

    EXPECT_PTR_EQ(&graph, &moved.sourceGraph());
    EXPECT_TRUE(moved.finalized());
    EXPECT_EQ(originalRegionCount, moved.size());
    EXPECT_EQ(originalDependencyCount, moved.dependencies().size());
    EXPECT_PTR_EQ(moved.findRegion(0), moved.regionFor(*input));
    EXPECT_PTR_EQ(moved.findRegion(1), moved.regionFor(*output));
    EXPECT_TRUE(RegionVerifier::verify(graph, moved).valid);
    EXPECT_TRUE(RegionVerifier::verify(
        graph,
        moved,
        RegionVerificationLevel::Executable
    ).valid);

    std::ostringstream after;
    RegionPrinter::print(moved, after);
    EXPECT_EQ(before.str(), after.str());
}

void testMultipleSemanticOutputs()
{
    Graph graph;
    Node* input0 = graph.createNode("input0", Operation::Input);
    Node* input1 = graph.createNode("input1", Operation::Input);
    Node* output0 = graph.createNode("output0", Operation::Output);
    Node* output1 = graph.createNode("output1", Operation::Output);
    graph.connect(input0, output0);
    graph.connect(input1, output1);

    RegionGraph regionGraph(graph);
    auto& region0 = regionGraph.createRegion();
    auto& region1 = regionGraph.createRegion();
    regionGraph.appendNode(region0.id(), *input0);
    regionGraph.appendNode(region0.id(), *output0);
    regionGraph.appendNode(region1.id(), *input1);
    regionGraph.appendNode(region1.id(), *output1);
    EXPECT_TRUE(regionGraph.finalize());

    EXPECT_EQ(std::size_t(1), region0.outputs().size());
    EXPECT_PTR_EQ(output0, region0.outputs()[0]);
    EXPECT_EQ(std::size_t(1), region1.outputs().size());
    EXPECT_PTR_EQ(output1, region1.outputs()[0]);
    EXPECT_TRUE(RegionVerifier::verify(graph, regionGraph).valid);
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
    runTest("Region strategy strings", testRegionStrategyStrings);
    runTest("Empty construction", testEmptyConstruction);
    runTest("Deterministic Region IDs", testDeterministicRegionIds);
    runTest("Node assignment rules", testNodeAssignmentRules);
    runTest(
        "Single Region structural graph",
        testSingleRegionStructuralGraph
    );
    runTest(
        "Two Region linear boundary",
        testTwoRegionLinearBoundary
    );
    runTest("Input deduplication", testInputDeduplication);
    runTest("Multiple crossing values", testMultipleCrossingValues);
    runTest(
        "Finalization freezes structure",
        testFinalizationFreezesStructure
    );
    runTest("Empty Region diagnostic", testEmptyRegionDiagnostic);
    runTest(
        "Missing Graph Node diagnostic",
        testMissingGraphNodeDiagnostic
    );
    runTest(
        "Non-topological Region order",
        testNonTopologicalRegionOrder
    );
    runTest(
        "Cyclic RegionGraph from non-convex partition",
        testCyclicRegionGraphFromNonConvexPartition
    );
    runTest(
        "Multiple unresolved strategies",
        testMultipleUnresolvedStrategies
    );
    runTest("Printer snapshot", testPrinterSnapshot);
    runTest("Repeatability", testRepeatability);
    runTest("Source Graph mismatch", testSourceGraphMismatch);
    runTest("Not finalized", testNotFinalized);
    runTest(
        "Nonempty Graph with zero Regions",
        testNonemptyGraphWithZeroRegions
    );
    runTest(
        "Empty Graph with zero Regions",
        testEmptyGraphWithZeroRegions
    );
    runTest(
        "Unknown strategy Region ID",
        testUnknownStrategyRegionId
    );
    runTest("Region lookup", testRegionLookup);
    runTest(
        "Operand order versus dependency value order",
        testOperandOrderVersusDependencyValueOrder
    );
    runTest(
        "Dependency lexicographic order",
        testDependencyLexicographicOrder
    );
    runTest(
        "Moved RegionGraph preserves validity",
        testMovedRegionGraphPreservesValidity
    );
    runTest(
        "Multiple semantic Outputs",
        testMultipleSemanticOutputs
    );

    if(failures != 0)
    {
        std::cerr << failures << " Region IR test failure(s)\n";
        return 1;
    }

    std::cout << "All Region IR tests passed\n";
    return 0;
}
