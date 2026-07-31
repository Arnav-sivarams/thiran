#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "analysis/CostModel.hpp"
#include "analysis/DependencyAnalysis.hpp"
#include "analysis/GraphStatistics.hpp"
#include "analysis/ResourceAnalysis.hpp"
#include "analysis/ShapeInference.hpp"
#include "backend/BackendBuilder.hpp"
#include "backend/FileEmitter.hpp"
#include "backend/TritonCodeGenerator.hpp"
#include "distributed/CommunicationPlanner.hpp"
#include "distributed/PartitionPlanner.hpp"
#include "driver/Doctor.hpp"
#include "frontend/ModuleLinker.hpp"
#include "optimizer/CanonicalizationPass.hpp"
#include "optimizer/ConstantFold.hpp"
#include "optimizer/DCEPass.hpp"
#include "optimizer/FusionRule.hpp"
#include "optimizer/GraphVerifier.hpp"
#include "optimizer/RemoveDoubleReluRule.hpp"
#include "optimizer/RewriteEngine.hpp"
#include "region/RegionPlan.hpp"
#include "region/RegionPythonEmitter.hpp"
#include "scheduler/DevicePlanner.hpp"
#include "scheduler/ExecutionPlanner.hpp"
#include "utils/BackendPrinter.hpp"
#include "utils/CodePrinter.hpp"
#include "utils/CommunicationPrinter.hpp"
#include "utils/ExecutionPrinter.hpp"
#include "utils/FusionPrinter.hpp"
#include "utils/GraphPrinter.hpp"
#include "utils/GraphvizEmitter.hpp"
#include "utils/PartitionPrinter.hpp"
#include "utils/RegionPlanDiagnosticPrinter.hpp"
#include "utils/RegionExecutionDiagnosticPrinter.hpp"
#include "utils/RegionPlanPrinter.hpp"
#include "utils/ResourcePrinter.hpp"
#include "utils/SchedulePrinter.hpp"
#include "utils/ShapePrinter.hpp"
#include "thiran/Version.hpp"

namespace
{

constexpr const char* usage =
    "Usage:\n"
    "  Thiran <source-file>\n"
    "  Thiran --plan <source-file>\n"
    "  Thiran --emit-plan <source-file> <output-file>\n"
    "  Thiran --emit-region-executor <source-file> <output-file>\n"
    "  Thiran --version\n"
    "  Thiran --help\n";

constexpr const char* doctorUsage = "Usage:\n  thiran doctor\n";

enum class DriverMode
{
    Normal,
    Plan,
    EmitPlan,
    EmitRegionExecutor
};

struct CommandLine final
{
    DriverMode mode;
    std::string sourcePath;
    std::string outputPath;
};

struct PreparedGraph final
{
    thiran::Graph graph;
    thiran::ShapeInference inference;
    std::size_t parsedNodeCount = 0;
    std::size_t fusionCount = 0;
    std::chrono::steady_clock::time_point optimizationStart;
    std::chrono::steady_clock::time_point optimizationEnd;
    bool succeeded = false;
    int failureExitCode = 1;
};

class ScopedOutputSilencer final
{
public:
    explicit ScopedOutputSilencer(bool silence)
        : original_(nullptr)
    {
        if(silence)
        {
            original_ = std::cout.rdbuf(sink_.rdbuf());
        }
    }

    ~ScopedOutputSilencer()
    {
        if(original_ != nullptr)
        {
            std::cout.rdbuf(original_);
        }
    }

    ScopedOutputSilencer(const ScopedOutputSilencer&) = delete;
    ScopedOutputSilencer& operator=(const ScopedOutputSilencer&) = delete;

private:
    std::ostringstream sink_;
    std::streambuf* original_;
};

bool parseCommandLine(int argc, char* argv[], CommandLine& command)
{
    if(argc == 2 &&
       (std::string(argv[1]) == "--help" ||
        std::string(argv[1]) == "-h"))
    {
        std::cout << usage;
        return false;
    }

    if(argc == 2 && argv[1][0] != '-')
    {
        command = {DriverMode::Normal, argv[1], {}};
        return true;
    }
    if(argc == 3 && std::string(argv[1]) == "--plan" &&
       argv[2][0] != '-')
    {
        command = {DriverMode::Plan, argv[2], {}};
        return true;
    }
    if(argc == 4 && std::string(argv[1]) == "--emit-plan" &&
       argv[2][0] != '-')
    {
        command = {DriverMode::EmitPlan, argv[2], argv[3]};
        return true;
    }
    if(argc == 4 &&
       std::string(argv[1]) == "--emit-region-executor" &&
       argv[2][0] != '-')
    {
        command = {
            DriverMode::EmitRegionExecutor,
            argv[2],
            argv[3]
        };
        return true;
    }

    std::cerr << "Invalid command line\n" << usage;
    return false;
}

PreparedGraph prepareGraph(
    const std::string& sourcePath,
    bool normalOutput
)
{
    PreparedGraph prepared;
    auto frontend = thiran::frontend::ModuleLinker::compile(sourcePath);
    if(!frontend.succeeded())
    {
        if(frontend.diagnostics.size() == 1 &&
           frontend.diagnostics.front().rfind("could not open source file", 0) == 0)
        {
            std::cout << "Could not open file.\n";
            return prepared;
        }
        std::cout << "Parsing failed:\n";
        for(const auto& diagnostic : frontend.diagnostics)
        {
            std::cout << "  error: " << diagnostic << "\n";
        }
        return prepared;
    }
    prepared.graph = std::move(*frontend.graph);

    thiran::GraphVerifier verifier;
    {
        ScopedOutputSilencer silence(!normalOutput);
        if(!verifier.verify(prepared.graph))
        {
            prepared.failureExitCode = 0;
            return prepared;
        }
    }

    prepared.parsedNodeCount = prepared.graph.nodes.size();
    if(normalOutput)
    {
        std::cout << "\n=== Optimization ===\n";
    }
    prepared.optimizationStart = std::chrono::steady_clock::now();

    {
        ScopedOutputSilencer silence(!normalOutput);
        thiran::CanonicalizationPass pass;
        pass.run(prepared.graph);
        thiran::ConstantFold constantFold;
        constantFold.run(prepared.graph);
        thiran::RewriteEngine engine;
        engine.addRule(
            std::make_shared<thiran::RemoveDoubleReluRule>()
        );
        engine.run(prepared.graph);
    }

    thiran::FusionRule fusion;
    const std::size_t beforeFusion = prepared.graph.nodes.size();
    fusion.run(prepared.graph);
    prepared.fusionCount = beforeFusion - prepared.graph.nodes.size();
    thiran::DCEPass dce;
    dce.run(prepared.graph);
    prepared.optimizationEnd = std::chrono::steady_clock::now();

    {
        ScopedOutputSilencer silence(!normalOutput);
        if(!verifier.verify(prepared.graph))
        {
            prepared.failureExitCode = 1;
            return prepared;
        }
    }

    if(normalOutput)
    {
        thiran::FusionPrinter::print(prepared.graph);
        thiran::GraphPrinter::print(prepared.graph);
        if(thiran::GraphvizEmitter::write(prepared.graph, "graph.dot"))
        {
            std::cout << "Graphviz output: graph.dot\n";
        }
    }

    prepared.inference.infer(prepared.graph);
    if(prepared.inference.hasErrors())
    {
        std::cout << "Shape inference failed:\n";
        for(const auto& diagnostic : prepared.inference.diagnostics())
        {
            std::cout << "  error: " << diagnostic << "\n";
        }
        return prepared;
    }

    prepared.succeeded = true;
    return prepared;
}

bool samePath(
    const std::string& sourcePath,
    const std::string& outputPath
)
{
    std::error_code sourceError;
    std::error_code outputError;
    const auto source = std::filesystem::absolute(
        sourcePath,
        sourceError
    ).lexically_normal();
    const auto output = std::filesystem::absolute(
        outputPath,
        outputError
    ).lexically_normal();

    if(!sourceError && !outputError && source == output)
    {
        return true;
    }

    std::error_code equivalentError;
    return std::filesystem::equivalent(
        sourcePath,
        outputPath,
        equivalentError
    ) && !equivalentError;
}

int runPlanning(
    const CommandLine& command,
    thiran::Graph& graph
)
{
    auto result = thiran::region::HybridRegionPlanner::build(graph);
    if(!result.succeeded())
    {
        std::cerr << "Hybrid region planning failed\n";
        thiran::RegionPlanDiagnosticPrinter::print(
            result.diagnostics,
            std::cerr
        );
        return 3;
    }

    if(command.mode == DriverMode::Plan)
    {
        thiran::RegionPlanPrinter::print(*result.plan, std::cout);
        return 0;
    }

    if(command.mode == DriverMode::EmitRegionExecutor)
    {
        auto emission = thiran::region::RegionPythonEmitter::emit(
            graph,
            *result.plan
        );
        if(!emission.succeeded())
        {
            std::cerr << "Region executor emission failed\n";
            thiran::RegionExecutionDiagnosticPrinter::print(
                emission.diagnostics,
                std::cerr
            );
            return 5;
        }

        if(samePath(command.sourcePath, command.outputPath))
        {
            std::cerr
                << "Refusing to overwrite source file with Region executor: "
                << command.outputPath << "\n";
            return 4;
        }

        std::ofstream output(command.outputPath, std::ios::trunc);
        if(!output.is_open())
        {
            std::cerr << "Failed to open Region executor output: "
                      << command.outputPath << "\n";
            return 4;
        }

        output << emission.artifact->source;
        output.flush();
        if(!output)
        {
            output.close();
            std::error_code cleanupError;
            std::filesystem::remove(command.outputPath, cleanupError);
            std::cerr << "Failed to write Region executor output: "
                      << command.outputPath << "\n";
            return 4;
        }
        output.close();
        if(!output)
        {
            std::error_code cleanupError;
            std::filesystem::remove(command.outputPath, cleanupError);
            std::cerr << "Failed to write Region executor output: "
                      << command.outputPath << "\n";
            return 4;
        }

        std::cout << "Wrote Region-controlled Python executor: "
                  << command.outputPath << "\n";
        return 0;
    }

    if(samePath(command.sourcePath, command.outputPath))
    {
        std::cerr
            << "Refusing to overwrite source file with RegionPlan: "
            << command.outputPath << "\n";
        return 4;
    }

    std::ostringstream planText;
    thiran::RegionPlanPrinter::print(*result.plan, planText);
    std::ofstream output(command.outputPath, std::ios::trunc);
    if(!output.is_open())
    {
        std::cerr << "Failed to open RegionPlan output: "
                  << command.outputPath << "\n";
        return 4;
    }

    output << planText.str();
    output.flush();
    if(!output)
    {
        output.close();
        std::error_code cleanupError;
        std::filesystem::remove(command.outputPath, cleanupError);
        std::cerr << "Failed to write RegionPlan output: "
                  << command.outputPath << "\n";
        return 4;
    }

    output.close();
    if(!output)
    {
        std::error_code cleanupError;
        std::filesystem::remove(command.outputPath, cleanupError);
        std::cerr << "Failed to write RegionPlan output: "
                  << command.outputPath << "\n";
        return 4;
    }

    std::cout << "Wrote hybrid region plan: "
              << command.outputPath << "\n";
    return 0;
}

int runNormalCompilation(
    PreparedGraph& prepared,
    std::chrono::steady_clock::time_point compilationStart
)
{
    thiran::Graph& graph = prepared.graph;
    thiran::ShapeInference& inference = prepared.inference;
    thiran::ResourceAnalysis resources;
    resources.analyze(graph, inference);
    thiran::GraphStatistics stats;

    std::cout << "\n";
    std::cout << "Nodes   : " << stats.nodeCount(graph) << "\n";
    std::cout << "Edges   : " << stats.edgeCount(graph) << "\n";
    std::cout << "Inputs  : " << stats.inputCount(graph) << "\n";
    std::cout << "Outputs : " << stats.outputCount(graph) << "\n";
    std::cout << "Depth   : " << stats.maxDepth(graph) << "\n";

    thiran::CostModel cost;
    std::cout << "Graph Cost : " << cost.graphCost(graph) << "\n";
    thiran::DependencyAnalysis dependency;
    auto levels = dependency.executionLevels(graph);
    thiran::SchedulePrinter::print(levels);
    thiran::ExecutionPlanner executionPlanner;
    auto execution = executionPlanner.build(graph);
    thiran::DevicePlanner devices;
    devices.assign(execution);
    thiran::PartitionPlanner partitionPlanner;
    auto partitions = partitionPlanner.partition(execution);
    thiran::PartitionPrinter::print(partitions);
    thiran::CommunicationPlanner communicationPlanner;
    auto communication = communicationPlanner.build(partitions);
    thiran::CommunicationPrinter::print(communication);
    thiran::BackendBuilder backendBuilder;
    auto backend = backendBuilder.build(execution);
    thiran::BackendPrinter::print(backend);
    thiran::TritonCodeGenerator generator;
    auto code = generator.generate(backend);

    if(!thiran::FileEmitter::write("generated.py", code))
    {
        std::cout << "Could not write generated.py\n";
        return 1;
    }

    thiran::CodePrinter::print(code);
    const auto compilationEnd = std::chrono::steady_clock::now();

    std::cout << "\n===== COMPILATION SUMMARY =====\n\n";
    std::cout << "Nodes                : " << stats.nodeCount(graph) << "\n";
    std::cout << "Edges                : " << stats.edgeCount(graph) << "\n";
    std::cout << "Depth                : " << stats.maxDepth(graph) << "\n";
    std::cout << "Peak Memory          : "
              << resources.peakMemoryBytes() / (1024.0 * 1024.0)
              << " MB\n";
    std::cout << "Estimated FLOPs      : "
              << resources.totalFlops() << "\n";
    std::cout << "Estimated Runtime    : "
              << resources.estimatedRuntime() << " s\n";
    std::cout << "Kernel Count         : " << backend.kernels.size() << "\n";
    std::cout << "Partition Count      : "
              << partitions.partitions.size() << "\n";
    std::cout << "Fusion Count         : "
              << prepared.fusionCount << "\n";
    std::cout << "Optimization Count   : "
              << prepared.parsedNodeCount - graph.nodes.size() << "\n";
    std::cout << "Optimization Time    : "
              << std::chrono::duration<double, std::milli>(
                     prepared.optimizationEnd -
                     prepared.optimizationStart
                 ).count()
              << " ms\n";
    std::cout << "Compile Time         : "
              << std::chrono::duration<double, std::milli>(
                     compilationEnd - compilationStart
                 ).count()
              << " ms\n";

    thiran::ExecutionPrinter::print(execution);
    thiran::ResourcePrinter::print(graph, resources);
    thiran::ShapePrinter::print(graph, inference);
    return 0;
}

}

int main(int argc, char* argv[])
{
    if(argc == 2 && std::string(argv[1]) == "--version")
    {
        std::cout << "Thiran " << thiran::version::string << "\n";
        return 0;
    }
    if(argc >= 2 && std::string(argv[1]) == "doctor")
    {
        if(argc == 3 && (std::string(argv[2]) == "--help" ||
                         std::string(argv[2]) == "-h"))
        {
            std::cout << doctorUsage;
            return 0;
        }
        if(argc != 2)
        {
            std::cerr << "Invalid doctor command\n" << doctorUsage;
            return 2;
        }
        const auto result = thiran::driver::Doctor::probe();
        std::cout << thiran::driver::Doctor::format(result);
        return thiran::driver::Doctor::exitCode(result);
    }
    const auto compilationStart = std::chrono::steady_clock::now();
    CommandLine command{DriverMode::Normal, {}, {}};
    if(!parseCommandLine(argc, argv, command))
    {
        if(argc == 2 &&
           (std::string(argv[1]) == "--help" ||
            std::string(argv[1]) == "-h"))
        {
            return 0;
        }
        return 2;
    }

    const bool normal = command.mode == DriverMode::Normal;
    PreparedGraph prepared = prepareGraph(command.sourcePath, normal);
    if(!prepared.succeeded)
    {
        return prepared.failureExitCode;
    }

    if(!normal)
    {
        return runPlanning(command, prepared.graph);
    }

    return runNormalCompilation(prepared, compilationStart);
}
