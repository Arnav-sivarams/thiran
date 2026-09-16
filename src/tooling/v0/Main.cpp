#include "tooling/v0/Driver.hpp"
#include "tooling/v0/BuildConfig.hpp"

#include <filesystem>
#include <iostream>
#include <string>

#ifndef TH009_CXX
#define TH009_CXX "c++"
#endif
#ifndef TH009_INCLUDE
#define TH009_INCLUDE "include"
#endif
#ifndef TH009_BUILD
#define TH009_BUILD "."
#endif

namespace {
using namespace thiran::v0::tooling;
void help() {
    std::cout <<
        "thiran-v0 - experimental developer compiler for the bounded V0 path\n\n"
        "Usage:\n"
        "  thiran-v0 check <file.th>\n"
        "  thiran-v0 emit-region <file.th> [--entry <name>]\n"
        "  thiran-v0 emit-cpp <file.th> [--entry <name>]\n"
        "  thiran-v0 build <file.th> [--entry <name>] -o <artifact>\n"
        "  thiran-v0 run <file.th> [--entry <name>]\n\n"
        "This is a non-production Linux/WSL developer tool. build/run use\n"
        "STRICT_NATIVE and a configured host C++20 compiler; no evaluator fallback.\n";
}
int exitCode(CompilerStage stage) {
    switch (stage) {
    case CompilerStage::Input: case CompilerStage::Syntax: return 2;
    case CompilerStage::Semantic: case CompilerStage::SemanticVerifier: return 3;
    case CompilerStage::OwnershipEffect: return 4;
    case CompilerStage::Backend: return 5;
    case CompilerStage::HostCompiler: return 6;
    case CompilerStage::ArtifactExecution: return 7;
    default: return 8;
    }
}
int fail(const DriverResult& result) {
    for (const auto& item : result.diagnostics) std::cerr << item.format() << '\n';
    if (!result.coverage.empty()) std::cerr << result.coverage;
    return exitCode(result.stage);
}
HostToolchainConfig config() {
    const std::filesystem::path build = TH009_BUILD;
    return {TH009_CXX, configuredHostCompilerArguments(), {TH009_INCLUDE},
            {build / "libthiran_v0_storage.a", build / "libthiran_v0_analysis.a",
             build / "libthiran_v0_semantic.a", build / "libthiran_v0_frontend.a"}};
}
}

int main(int argc, char** argv) {
    if (argc == 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        help(); return 0;
    }
    const std::string command = argv[1];
    if (command != "check" && command != "emit-region" && command != "emit-cpp" &&
        command != "build" && command != "run") {
        std::cerr << "unknown command: " << command << '\n'; help(); return 2;
    }
    std::string sourcePath, entry = "main", output;
    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--entry" && i + 1 < argc) entry = argv[++i];
        else if (argument == "-o" && i + 1 < argc) output = argv[++i];
        else if (!argument.empty() && argument[0] == '-') {
            std::cerr << "unknown or incomplete option: " << argument << '\n'; return 2;
        } else if (sourcePath.empty()) sourcePath = argument;
        else { std::cerr << "unexpected argument: " << argument << '\n'; return 2; }
    }
    if (sourcePath.empty() || std::filesystem::path(sourcePath).extension() != ".th") {
        std::cerr << "input: expected one .th source file\n"; return 2;
    }
    if (command == "build" && output.empty()) { std::cerr << "build: -o <artifact> is required\n"; return 2; }
    DriverDiagnostic inputFailure;
    auto source = readSourceFile(sourcePath, inputFailure);
    if (!source) { std::cerr << inputFailure.format() << '\n'; return 2; }
    CompilerDriver driver(config());
    if (command == "check") {
        auto result = driver.checkSource(*source);
        if (!result.success) return fail(result);
        std::cout << "CHECK PASS\n"; return 0;
    }
    if (command == "emit-region") {
        auto result = driver.emitRegion(*source, entry);
        if (!result.success) return fail(result);
        std::cout << *result.generatedSource; return 0;
    }
    if (command == "emit-cpp") {
        auto result = driver.emitNativeCpp(*source, entry);
        if (!result.success) return fail(result);
        std::cout << *result.generatedSource; return 0;
    }
    if (command == "build") {
        BuildOptions options; options.entry = entry; options.output = output;
        auto result = driver.buildNative(*source, options);
        if (!result.success) return fail(result);
        std::cout << "BUILD PASS " << result.artifactPath->string() << '\n'; return 0;
    }
    auto result = driver.runNative(*source, entry);
    if (!result.success) {
        if (result.subprocess) {
            std::cout << result.subprocess->standardOutput;
            std::cerr << result.subprocess->standardError;
        }
        return fail(result);
    }
    std::cout << result.subprocess->standardOutput;
    std::cerr << result.subprocess->standardError;
    return result.subprocess->exitStatus;
}
