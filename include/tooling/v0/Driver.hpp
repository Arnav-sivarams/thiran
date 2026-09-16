#pragma once

#include "backend/v0/NativeCpu.hpp"
#include "interop/v0/NativeLibrary.hpp"
#include "tooling/v0/Process.hpp"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace thiran::v0::tooling {

enum class CompilerStage {
    None, Input, Syntax, Semantic, SemanticVerifier, OwnershipEffect,
    Backend, HostCompiler, ArtifactExecution, InternalTool
};
std::string stageName(CompilerStage stage);

struct DriverDiagnostic {
    CompilerStage stage = CompilerStage::InternalTool;
    std::string category;
    std::string message;
    std::string source;
    std::optional<SourceSpan> span;
    std::string format() const;
};

struct SourceInput { std::string identity; std::string bytes; };
struct HostToolchainConfig {
    std::string compilerExecutable;
    std::vector<std::string> compilerArguments;
    std::vector<std::filesystem::path> includePaths;
    std::vector<std::filesystem::path> v0StaticLibraries;
};
struct BuildOptions {
    std::string entry = "main";
    std::filesystem::path output;
    std::vector<interop::NativeLibraryContract> nativeContracts;
};
struct DevelopmentBuildRecord {
    CompilerStage stage = CompilerStage::None;
    std::string entry;
    std::string coverage;
    std::optional<std::filesystem::path> emittedSourcePath;
    std::optional<std::filesystem::path> outputArtifact;
    std::optional<int> hostCompilerExitStatus;
};
struct CheckedProgram {
    semantic::Module module;
    analysis::OwnershipAnalysisResult ownership;
};
struct ParseSourceResult {
    std::optional<thiran::v0::Module> module;
    std::vector<DriverDiagnostic> diagnostics;
    bool ok() const { return module.has_value() && diagnostics.empty(); }
};
struct DriverResult {
    bool success = false;
    CompilerStage stage = CompilerStage::None;
    std::vector<DriverDiagnostic> diagnostics;
    std::string coverage;
    std::optional<std::string> generatedSource;
    std::optional<std::filesystem::path> artifactPath;
    std::optional<ProcessResult> subprocess;
    std::optional<DevelopmentBuildRecord> buildRecord;
    std::shared_ptr<CheckedProgram> checked;
    std::optional<backend::TensorRegion> region;
};

class CompilerDriver {
public:
    explicit CompilerDriver(HostToolchainConfig config);
    ParseSourceResult parseSource(const SourceInput& source) const;
    DriverResult checkSource(const SourceInput& source) const;
    DriverResult extractNative(const SourceInput& source, std::string_view entry) const;
    DriverResult emitRegion(const SourceInput& source, std::string_view entry) const;
    DriverResult emitNativeCpp(const SourceInput& source, std::string_view entry) const;
    DriverResult buildNative(const SourceInput& source, const BuildOptions& options) const;
    DriverResult runNative(const SourceInput& source, std::string_view entry) const;

    // Internal integration proof: compiles a C++ stub through the same verified
    // structured link plan. It does not expose a Thiran source FFI operation.
    DriverResult buildIntegrationStubForTesting(
        std::string_view cpp, const std::filesystem::path& output,
        const std::vector<interop::NativeLibraryContract>& contracts) const;
private:
    HostToolchainConfig config_;
    DriverResult compileCpp(std::string_view cpp, const std::filesystem::path& output,
                            std::string_view entry, std::string coverage,
                            const std::vector<interop::NativeLibraryContract>& contracts) const;
};

std::optional<SourceInput> readSourceFile(const std::filesystem::path& path,
                                          DriverDiagnostic& diagnostic);

} // namespace thiran::v0::tooling
