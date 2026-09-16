#include "tooling/v0/Driver.hpp"

#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Verifier.hpp"
#include <fstream>
#include <sstream>
#include <system_error>
#include <unistd.h>

namespace thiran::v0::tooling {
namespace {
class TemporaryDirectory {
public:
    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "thiran-v0-XXXXXX").string();
        std::vector<char> storage(pattern.begin(), pattern.end()); storage.push_back('\0');
        if (auto* made = ::mkdtemp(storage.data())) path_ = made;
    }
    ~TemporaryDirectory() {
        if (!path_.empty()) { std::error_code ignored; std::filesystem::remove_all(path_, ignored); }
    }
    bool valid() const { return !path_.empty(); }
    const std::filesystem::path& path() const { return path_; }
private:
    std::filesystem::path path_;
};
DriverDiagnostic diagnostic(CompilerStage stage, std::string category, std::string message,
                            std::string source = {}, std::optional<SourceSpan> span = std::nullopt) {
    return {stage, std::move(category), std::move(message), std::move(source), span};
}
std::string joinErrors(const std::vector<std::string>& errors) {
    std::ostringstream out;
    for (std::size_t i = 0; i < errors.size(); ++i) { if (i) out << "; "; out << errors[i]; }
    return out.str();
}
}

std::string stageName(CompilerStage stage) {
    switch (stage) {
    case CompilerStage::None: return "success";
    case CompilerStage::Input: return "input";
    case CompilerStage::Syntax: return "syntax";
    case CompilerStage::Semantic: return "semantic";
    case CompilerStage::SemanticVerifier: return "semantic-verifier";
    case CompilerStage::OwnershipEffect: return "ownership-effect";
    case CompilerStage::Backend: return "backend";
    case CompilerStage::HostCompiler: return "host-compiler";
    case CompilerStage::ArtifactExecution: return "artifact-execution";
    case CompilerStage::InternalTool: return "internal-tool";
    }
    return "internal-tool";
}
std::string DriverDiagnostic::format() const {
    std::ostringstream out;
    if (!source.empty()) {
        out << source;
        if (span) out << ':' << span->begin.line << ':' << span->begin.column;
        out << ": ";
    }
    out << stageName(stage) << '[' << category << "]: " << message;
    return out.str();
}

std::optional<SourceInput> readSourceFile(const std::filesystem::path& path,
                                          DriverDiagnostic& failure) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        failure = diagnostic(CompilerStage::Input, "INPUT-UNREADABLE", "cannot read source file", path.string());
        return std::nullopt;
    }
    std::ostringstream bytes; bytes << input.rdbuf();
    if (!input.good() && !input.eof()) {
        failure = diagnostic(CompilerStage::Input, "INPUT-READ", "failed while reading exact source bytes", path.string());
        return std::nullopt;
    }
    return SourceInput{path.string(), bytes.str()};
}

CompilerDriver::CompilerDriver(HostToolchainConfig config) : config_(std::move(config)) {}

ParseSourceResult CompilerDriver::parseSource(const SourceInput& source) const {
    ParseSourceResult out;
    auto parsed = thiran::v0::parse(source.bytes, source.identity);
    for (const auto& item : parsed.diagnostics)
        out.diagnostics.push_back(diagnostic(CompilerStage::Syntax, "SYNTAX", item.message, item.source, item.span));
    if (parsed.module) out.module = std::move(*parsed.module);
    return out;
}

DriverResult CompilerDriver::checkSource(const SourceInput& source) const {
    DriverResult out;
    auto parsed = parseSource(source);
    if (!parsed.ok()) {
        out.stage = CompilerStage::Syntax; out.diagnostics = std::move(parsed.diagnostics); return out;
    }
    auto analyzed = semantic::analyze(*parsed.module);
    if (!analyzed.diagnostics.empty() || !analyzed.module) {
        out.stage = CompilerStage::Semantic;
        for (const auto& item : analyzed.diagnostics)
            out.diagnostics.push_back(diagnostic(CompilerStage::Semantic, item.category, item.message,
                                                 item.source, item.span));
        return out;
    }
    auto verification = semantic::verify(*analyzed.module);
    if (!verification.ok) {
        out.stage = CompilerStage::SemanticVerifier;
        for (const auto& error : verification.errors)
            out.diagnostics.push_back(diagnostic(out.stage, "SEMANTIC-IR-INVALID", error, source.identity));
        return out;
    }
    auto ownership = analysis::analyze(*analyzed.module);
    if (!ownership.ok()) {
        out.stage = CompilerStage::OwnershipEffect;
        for (const auto& item : ownership.diagnostics)
            out.diagnostics.push_back(diagnostic(out.stage, item.category, item.message, item.source, item.span));
        return out;
    }
    auto audit = analysis::auditFacts(*analyzed.module, ownership);
    if (!audit.empty()) {
        out.stage = CompilerStage::OwnershipEffect;
        for (const auto& error : audit)
            out.diagnostics.push_back(diagnostic(out.stage, "OWNERSHIP-FACT-AUDIT", error, source.identity));
        return out;
    }
    out.checked = std::make_shared<CheckedProgram>(CheckedProgram{std::move(*analyzed.module), std::move(ownership)});
    out.success = true; out.stage = CompilerStage::None;
    return out;
}

DriverResult CompilerDriver::extractNative(const SourceInput& source, std::string_view entry) const {
    auto out = checkSource(source);
    if (!out.success) return out;
    auto native = backend::extractStrictNative(out.checked->module, out.checked->ownership,
                                               std::string(entry), true);
    out.coverage = native.coverage;
    if (!native.ok()) {
        out.success = false; out.stage = CompilerStage::Backend;
        out.diagnostics.push_back(diagnostic(out.stage, "BACKEND-UNSUPPORTED", native.diagnostic,
                                             source.identity));
        return out;
    }
    out.region = std::move(*native.region);
    return out;
}

DriverResult CompilerDriver::emitRegion(const SourceInput& source, std::string_view entry) const {
    auto out = extractNative(source, entry);
    if (out.success) out.generatedSource = out.region->dump();
    return out;
}
DriverResult CompilerDriver::emitNativeCpp(const SourceInput& source, std::string_view entry) const {
    auto out = extractNative(source, entry);
    if (out.success) out.generatedSource = backend::emitCpp20(*out.region, true);
    return out;
}

DriverResult CompilerDriver::compileCpp(
    std::string_view cpp, const std::filesystem::path& requestedOutput,
    std::string_view entry, std::string coverage,
    const std::vector<interop::NativeLibraryContract>& contracts) const {
    DriverResult out; out.generatedSource = std::string(cpp); out.coverage = std::move(coverage);
    out.buildRecord = DevelopmentBuildRecord{CompilerStage::HostCompiler, std::string(entry), out.coverage};
    if (config_.compilerExecutable.empty()) {
        out.stage = CompilerStage::InternalTool;
        out.diagnostics.push_back(diagnostic(out.stage, "TOOLCHAIN-CONFIG", "host C++ compiler is not configured"));
        return out;
    }
    const auto contractCheck = interop::verifyContracts(contracts);
    if (!contractCheck.ok()) {
        out.stage = CompilerStage::InternalTool;
        out.diagnostics.push_back(diagnostic(out.stage, "INTEROP-CONTRACT", joinErrors(contractCheck.errors)));
        return out;
    }
    for (const auto& contract : contracts) {
        if (contract.applicability != interop::HostApplicability::PosixNativeCpu) {
            out.stage = CompilerStage::InternalTool;
            out.diagnostics.push_back(diagnostic(out.stage, "INTEROP-HOST",
                                                 "native contract is not applicable to the POSIX CPU host"));
            return out;
        }
    }
    TemporaryDirectory temporary;
    if (!temporary.valid()) {
        out.stage = CompilerStage::InternalTool;
        out.diagnostics.push_back(diagnostic(out.stage, "TEMPORARY-DIRECTORY", "cannot create controlled temporary directory"));
        return out;
    }
    const auto sourcePath = temporary.path() / "generated.cpp";
    const auto temporaryArtifact = temporary.path() / "artifact";
    {
        std::ofstream sourceFile(sourcePath, std::ios::binary);
        sourceFile.write(cpp.data(), static_cast<std::streamsize>(cpp.size()));
        if (!sourceFile) {
            out.stage = CompilerStage::InternalTool;
            out.diagnostics.push_back(diagnostic(out.stage, "TEMPORARY-SOURCE", "cannot write generated C++"));
            return out;
        }
    }
    ProcessRequest request{config_.compilerExecutable, config_.compilerArguments};
    request.arguments.push_back("-std=c++20");
    for (const auto& include : config_.includePaths) request.arguments.push_back("-I" + include.string());
    request.arguments.push_back(sourcePath.string());
    for (const auto& library : config_.v0StaticLibraries) request.arguments.push_back(library.string());
    for (const auto& contract : contracts) for (const auto& item : contract.linkItems) {
        if (item.kind == interop::LinkItemKind::StaticArchive) request.arguments.push_back(item.value);
        else if (item.kind == interop::LinkItemKind::SearchPath) request.arguments.push_back("-L" + item.value);
        else request.arguments.push_back("-l" + item.value);
    }
    request.arguments.push_back("-o"); request.arguments.push_back(temporaryArtifact.string());
    auto process = runProcess(request);
    out.subprocess = process;
    out.buildRecord->hostCompilerExitStatus = process.exitStatus;
    if (!process.launched || process.exitStatus != 0) {
        out.stage = CompilerStage::HostCompiler;
        out.diagnostics.push_back(diagnostic(out.stage, "HOST-COMPILER-FAILURE",
            process.launcherError.empty() ? process.standardError : process.launcherError));
        return out;
    }
    std::error_code error;
    const auto output = std::filesystem::absolute(requestedOutput, error);
    if (error || output.empty() || !std::filesystem::exists(output.parent_path())) {
        out.stage = CompilerStage::InternalTool;
        out.diagnostics.push_back(diagnostic(out.stage, "OUTPUT-PATH", "output parent directory does not exist",
                                             requestedOutput.string()));
        return out;
    }
    std::filesystem::copy_file(temporaryArtifact, output,
                               std::filesystem::copy_options::overwrite_existing, error);
    if (error) {
        out.stage = CompilerStage::InternalTool;
        out.diagnostics.push_back(diagnostic(out.stage, "OUTPUT-WRITE", error.message(), output.string()));
        return out;
    }
    std::filesystem::permissions(output, std::filesystem::status(temporaryArtifact).permissions(), error);
    out.success = true; out.stage = CompilerStage::None; out.artifactPath = output;
    out.buildRecord->stage = CompilerStage::None; out.buildRecord->outputArtifact = output;
    return out;
}

DriverResult CompilerDriver::buildNative(const SourceInput& source, const BuildOptions& options) const {
    auto native = emitNativeCpp(source, options.entry);
    if (!native.success) return native;
    auto built = compileCpp(*native.generatedSource, options.output, options.entry,
                            native.coverage, options.nativeContracts);
    built.checked = native.checked; built.region = native.region;
    return built;
}

DriverResult CompilerDriver::runNative(const SourceInput& source, std::string_view entry) const {
    TemporaryDirectory temporary;
    if (!temporary.valid()) {
        DriverResult out; out.stage = CompilerStage::InternalTool;
        out.diagnostics.push_back(diagnostic(out.stage, "TEMPORARY-DIRECTORY", "cannot create run directory"));
        return out;
    }
    BuildOptions options; options.entry = std::string(entry); options.output = temporary.path() / "program";
    auto out = buildNative(source, options);
    if (!out.success) return out;
    auto execution = runProcess({out.artifactPath->string(), {}});
    out.subprocess = execution;
    if (!execution.launched || execution.exitStatus != 0) {
        out.success = false; out.stage = CompilerStage::ArtifactExecution;
        out.diagnostics.push_back(diagnostic(out.stage, "ARTIFACT-EXECUTION-FAILURE",
            execution.launcherError.empty() ? "artifact exited with status " + std::to_string(execution.exitStatus)
                                            : execution.launcherError));
    }
    return out;
}

DriverResult CompilerDriver::buildIntegrationStubForTesting(
    std::string_view cpp, const std::filesystem::path& output,
    const std::vector<interop::NativeLibraryContract>& contracts) const {
    return compileCpp(cpp, output, "<integration-stub>", "fallback: NONE\n", contracts);
}

} // namespace thiran::v0::tooling
