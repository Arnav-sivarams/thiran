#include <cstddef>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "region/RegionPlan.hpp"
#include "utils/RegionPlanDiagnosticPrinter.hpp"

namespace
{

using thiran::RegionPlanDiagnosticPrinter;
using thiran::region::RegionPlanDiagnostic;
using thiran::region::RegionPlanDiagnosticStage;

int failures = 0;
std::string currentTest;

void expectEqual(
    const std::string& expected,
    const std::string& actual,
    int line
)
{
    if(expected != actual)
    {
        ++failures;
        std::cerr << currentTest << ':' << line
                  << ": expected '" << expected
                  << "', actual '" << actual << "'\n";
    }
}

void expectTrue(bool value, const std::string& expression, int line)
{
    if(!value)
    {
        ++failures;
        std::cerr << currentTest << ':' << line
                  << ": expected " << expression << " to be true\n";
    }
}

#define EXPECT_EQ(expected, actual) expectEqual((expected), (actual), __LINE__)
#define EXPECT_TRUE(value) expectTrue((value), #value, __LINE__)

RegionPlanDiagnostic diagnostic(
    RegionPlanDiagnosticStage stage,
    std::string code,
    std::string message
)
{
    return {
        stage,
        std::move(code),
        std::move(message),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt
    };
}

std::string print(const std::vector<RegionPlanDiagnostic>& diagnostics)
{
    std::ostringstream output;
    RegionPlanDiagnosticPrinter::print(diagnostics, output);
    return output.str();
}

void testEmptyDiagnosticList()
{
    EXPECT_EQ(
        "Region plan diagnostics\n  <none>\n",
        print({})
    );
}

void testClassificationDiagnostic()
{
    EXPECT_EQ(
        "Region plan diagnostics\n"
        "  [CLASSIFICATION][SC001]: bad shape\n",
        print({diagnostic(
            RegionPlanDiagnosticStage::Classification,
            "SC001",
            "bad shape"
        )})
    );
}

void testFormationDiagnostic()
{
    auto value = diagnostic(
        RegionPlanDiagnosticStage::Formation,
        "RF008",
        "null producer"
    );
    value.nodeName = "consumer";
    EXPECT_EQ(
        "Region plan diagnostics\n"
        "  [FORMATION][RF008] node=consumer: null producer\n",
        print({value})
    );
}

void testVerificationDiagnostic()
{
    auto value = diagnostic(
        RegionPlanDiagnosticStage::Verification,
        "RP011",
        "strategy mismatch"
    );
    value.nodeName = "relu";
    value.regionId = 2;
    EXPECT_EQ(
        "Region plan diagnostics\n"
        "  [VERIFICATION][RP011] node=relu region=2: strategy mismatch\n",
        print({value})
    );
}

void testAllMetadataFields()
{
    auto value = diagnostic(
        RegionPlanDiagnosticStage::Classification,
        "ALL",
        "message"
    );
    value.nodeName = "node";
    value.graphIndex = 1;
    value.assignmentIndex = 2;
    value.dimensionIndex = 3;
    value.regionId = 4;
    EXPECT_EQ(
        "Region plan diagnostics\n"
        "  [CLASSIFICATION][ALL] node=node graph-index=1 "
        "assignment-index=2 dimension-index=3 region=4: message\n",
        print({value})
    );
}

void testOmittedMetadata()
{
    EXPECT_EQ(
        "Region plan diagnostics\n"
        "  [VERIFICATION][RP001]: message\n",
        print({diagnostic(
            RegionPlanDiagnosticStage::Verification,
            "RP001",
            "message"
        )})
    );
}

void testDiagnosticOrderPreservation()
{
    const std::string output = print({
        diagnostic(RegionPlanDiagnosticStage::Formation, "FIRST", "one"),
        diagnostic(RegionPlanDiagnosticStage::Verification, "SECOND", "two")
    });
    EXPECT_TRUE(output.find("FIRST") < output.find("SECOND"));
}

void testExactFinalNewline()
{
    const std::string output = print({
        diagnostic(RegionPlanDiagnosticStage::Formation, "RF", "message")
    });
    EXPECT_TRUE(!output.empty());
    EXPECT_TRUE(output.back() == '\n');
    EXPECT_TRUE(output.size() == 1 || output[output.size() - 2] != '\n');
}

void testNoBlankLines()
{
    EXPECT_TRUE(print({
        diagnostic(RegionPlanDiagnosticStage::Formation, "RF", "message")
    }).find("\n\n") == std::string::npos);
}

void testNoPointerAddresses()
{
    EXPECT_TRUE(print({
        diagnostic(RegionPlanDiagnosticStage::Formation, "RF", "message")
    }).find("0x") == std::string::npos);
}

void testInvalidDiagnosticStageString()
{
    EXPECT_EQ(
        "Region plan diagnostics\n  [UNKNOWN][X]: message\n",
        print({diagnostic(
            static_cast<RegionPlanDiagnosticStage>(255),
            "X",
            "message"
        )})
    );
}

void testMultiDiagnosticExactSnapshot()
{
    auto classification = diagnostic(
        RegionPlanDiagnosticStage::Classification,
        "SC004",
        "invalid shape"
    );
    classification.nodeName = "bad";
    classification.graphIndex = 0;
    classification.dimensionIndex = 1;
    auto formation = diagnostic(
        RegionPlanDiagnosticStage::Formation,
        "RF008",
        "null producer"
    );
    formation.nodeName = "consumer";

    EXPECT_EQ(
        "Region plan diagnostics\n"
        "  [CLASSIFICATION][SC004] node=bad graph-index=0 "
        "dimension-index=1: invalid shape\n"
        "  [FORMATION][RF008] node=consumer: null producer\n",
        print({classification, formation})
    );
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
    runTest("Empty diagnostic list", testEmptyDiagnosticList);
    runTest("Classification diagnostic", testClassificationDiagnostic);
    runTest("Formation diagnostic", testFormationDiagnostic);
    runTest("Verification diagnostic", testVerificationDiagnostic);
    runTest("All metadata fields", testAllMetadataFields);
    runTest("Omitted metadata", testOmittedMetadata);
    runTest("Diagnostic order preservation", testDiagnosticOrderPreservation);
    runTest("Exact final newline", testExactFinalNewline);
    runTest("No blank lines", testNoBlankLines);
    runTest("No pointer addresses", testNoPointerAddresses);
    runTest("Invalid diagnostic stage string", testInvalidDiagnosticStageString);
    runTest("Multi-diagnostic exact snapshot", testMultiDiagnosticExactSnapshot);

    if(failures != 0)
    {
        std::cerr << failures
                  << " Region Plan CLI Support test failure(s)\n";
        return 1;
    }

    std::cout << "All Region Plan CLI Support tests passed\n";
    return 0;
}
