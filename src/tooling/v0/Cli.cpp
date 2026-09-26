#include "tooling/v0/Cli.hpp"

#include "artifact/v0/NativeArtifacts.hpp"
#include "model/v0/ModelBundle.hpp"
#include "tooling/v0/BuildConfig.hpp"
#include "tooling/v0/Driver.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#ifndef TH009_CXX
#define TH009_CXX "c++"
#endif
#ifndef TH009_INCLUDE
#define TH009_INCLUDE "include"
#endif
#ifndef TH009_BUILD
#define TH009_BUILD "."
#endif

namespace thiran::v0::tooling {
namespace {
namespace fs = std::filesystem;
namespace artifact = thiran::v0::artifact;
namespace model = thiran::v0::model;

enum class DiagnosticFormat { Text, Json };

struct SourceOptions {
    std::string source;
    std::string entry = "main";
    std::optional<artifact::NativeBackend> backend;
    std::vector<std::string> extensions;
    fs::path output;
    int device = 0;
    bool verbose = false;
    bool entrySpecified = false;
    bool outputSpecified = false;
    bool deviceSpecified = false;
    DiagnosticFormat diagnostics = DiagnosticFormat::Text;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        auto pattern = (fs::temp_directory_path() / "thiran-cli-XXXXXX").string();
        std::vector<char> bytes(pattern.begin(), pattern.end());
        bytes.push_back('\0');
        if (auto* made = ::mkdtemp(bytes.data())) path_ = made;
    }
    ~TemporaryDirectory() {
        if (!path_.empty()) {
            std::error_code ignored;
            fs::remove_all(path_, ignored);
        }
    }
    bool valid() const noexcept { return !path_.empty(); }
    const fs::path& path() const noexcept { return path_; }
private:
    fs::path path_;
};

std::string json(std::string_view text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char byte : text) {
        switch (byte) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (byte < 0x20) {
                constexpr char digits[] = "0123456789abcdef";
                out << "\\u00" << digits[byte >> 4] << digits[byte & 15];
            } else out << static_cast<char>(byte);
        }
    }
    out << '"';
    return out.str();
}

int cliError(std::string_view message) {
    std::cerr << "error[TH022-CLI]: " << message << '\n';
    return 2;
}

void sourceHelp(std::string_view command) {
    if (command == "check") {
        std::cout << "Usage: thiran check <source.th> [--extension <library>]... "
                     "[--diagnostic-format <text|json>] [--verbose]\n";
    } else if (command == "run") {
        std::cout << "Usage: thiran run <source.th> --backend <cpu|gpu> "
                     "[--entry <name>] [--extension <library>]... [--device <ordinal>] "
                     "[--diagnostic-format <text|json>] [--verbose]\n";
    } else {
        std::cout << "Usage: thiran build <source.th> --backend <cpu|gpu> -o <artifact.tha> "
                     "[--entry <name>] [--extension <library>]... "
                     "[--diagnostic-format <text|json>] [--verbose]\n";
    }
}

void artifactHelp() {
    std::cout <<
        "Usage:\n"
        "  thiran artifact inspect <artifact.tha>\n"
        "  thiran artifact run <artifact.tha> [--device <ordinal>] [--verbose]\n";
}

void modelHelp() {
    std::cout <<
        "Usage:\n"
        "  thiran model inspect <model.thm>\n"
        "  thiran model run <model.thm> --backend <cpu|gpu> --input <f32> "
        "[--device <ordinal>] [--verbose]\n";
}

bool parseInteger(std::string_view spelling, int& value) {
    const auto parsed = std::from_chars(spelling.data(), spelling.data() + spelling.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == spelling.data() + spelling.size() && value >= 0;
}

std::optional<std::string> nextValue(int argc, int& index, std::string_view option) {
    if (index + 1 >= argc) return std::string(option) + " requires a value";
    ++index;
    return {};
}

std::optional<std::string> parseSourceOptions(int argc, char* argv[],
                                              std::string_view command,
                                              SourceOptions& options) {
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--entry") {
            if (auto error = nextValue(argc, index, argument)) return error;
            if (options.entrySpecified) return "--entry may be specified only once";
            options.entrySpecified = true;
            options.entry = argv[index];
        } else if (argument == "--backend") {
            if (auto error = nextValue(argc, index, argument)) return error;
            if (options.backend) return "--backend may be specified only once";
            const std::string value = argv[index];
            if (value == "cpu") options.backend = artifact::NativeBackend::Cpu;
            else if (value == "gpu") options.backend = artifact::NativeBackend::Gpu;
            else return "invalid backend '" + value + "' (expected cpu or gpu)";
        } else if (argument == "--extension") {
            if (auto error = nextValue(argc, index, argument)) return error;
            options.extensions.emplace_back(argv[index]);
        } else if (argument == "-o") {
            if (auto error = nextValue(argc, index, argument)) return error;
            if (options.outputSpecified) return "-o may be specified only once";
            options.outputSpecified = true;
            options.output = argv[index];
        } else if (argument == "--device") {
            if (auto error = nextValue(argc, index, argument)) return error;
            if (options.deviceSpecified) return "--device may be specified only once";
            options.deviceSpecified = true;
            if (!parseInteger(argv[index], options.device)) return "invalid device ordinal";
        } else if (argument == "--diagnostic-format") {
            if (auto error = nextValue(argc, index, argument)) return error;
            const std::string value = argv[index];
            if (value == "text") options.diagnostics = DiagnosticFormat::Text;
            else if (value == "json") options.diagnostics = DiagnosticFormat::Json;
            else return "invalid diagnostic format '" + value + "' (expected text or json)";
        } else if (argument == "--verbose") {
            options.verbose = true;
        } else if (!argument.empty() && argument.front() == '-') {
            return "unknown option '" + argument + "'";
        } else if (options.source.empty()) {
            options.source = argument;
        } else {
            return "unexpected argument '" + argument + "'";
        }
    }
    if (options.source.empty()) return "missing source file";
    if (command == "check" && (options.backend || options.entrySpecified))
        return "check does not accept --backend or --entry";
    if (command == "check" && (options.outputSpecified || options.deviceSpecified))
        return "check received an option that only applies to build or run";
    if (command != "build" && options.outputSpecified) return "-o is only valid for build";
    if (command == "build" && !options.outputSpecified) return "build requires -o <artifact.tha>";
    if (command != "run" && options.deviceSpecified) return "--device is only valid for run";
    if (command != "check" && !options.backend) return std::string(command) + " requires --backend <cpu|gpu>";
    return {};
}

std::string sourceLine(std::string_view bytes, std::uint32_t requested) {
    if (requested == 0) return {};
    std::uint32_t line = 1;
    std::size_t begin = 0;
    while (line < requested) {
        const auto next = bytes.find('\n', begin);
        if (next == std::string_view::npos) return {};
        begin = next + 1;
        ++line;
    }
    const auto end = bytes.find('\n', begin);
    auto result = std::string(bytes.substr(begin, end == std::string_view::npos ? bytes.size() - begin : end - begin));
    if (!result.empty() && result.back() == '\r') result.pop_back();
    return result;
}

void emitDiagnostic(const DriverDiagnostic& item, const SourceInput& source,
                    DiagnosticFormat format) {
    if (format == DiagnosticFormat::Json) {
        std::cerr << "{\"severity\":\"error\",\"code\":" << json(item.category)
                  << ",\"message\":" << json(item.message)
                  << ",\"source\":" << json(item.source);
        if (item.span) {
            std::cerr << ",\"start_line\":" << item.span->begin.line
                      << ",\"start_column\":" << item.span->begin.column
                      << ",\"end_line\":" << item.span->end.line
                      << ",\"end_column\":" << item.span->end.column;
        } else {
            std::cerr << ",\"start_line\":null,\"start_column\":null,"
                         "\"end_line\":null,\"end_column\":null";
        }
        std::cerr << "}\n";
        return;
    }
    if (!item.source.empty()) {
        std::cerr << item.source;
        if (item.span) std::cerr << ':' << item.span->begin.line << ':' << item.span->begin.column;
        std::cerr << ": ";
    }
    std::cerr << "error[" << item.category << "]: " << item.message << '\n';
    if (!item.span || item.source != source.identity) return;
    const auto line = sourceLine(source.bytes, item.span->begin.line);
    if (line.empty()) return;
    std::cerr << "  " << item.span->begin.line << " | " << line << '\n' << "    | ";
    const auto column = std::max<std::uint32_t>(1, item.span->begin.column);
    for (std::uint32_t index = 1; index < column; ++index) std::cerr << ' ';
    std::uint32_t width = 1;
    if (item.span->begin.line == item.span->end.line && item.span->end.column > column)
        width = item.span->end.column - column;
    for (std::uint32_t index = 0; index < width; ++index) std::cerr << '^';
    std::cerr << '\n';
}

int fail(const DriverResult& result, const SourceInput& source, DiagnosticFormat format) {
    for (const auto& item : result.diagnostics) emitDiagnostic(item, source, format);
    if (!result.coverage.empty() && format == DiagnosticFormat::Text) std::cerr << result.coverage;
    return result.stage == CompilerStage::ArtifactExecution ? 4 : 3;
}

int artifactFailure(const artifact::ArtifactError& error, std::string_view operation) {
    std::cerr << "error[" << error.code << "]: " << operation << ": " << error.message << '\n';
    return error.category == artifact::ArtifactErrorCategory::Execution ? 4 : 3;
}

HostToolchainConfig driverToolchain() {
    const fs::path build = TH009_BUILD;
    return {TH009_CXX, configuredHostCompilerArguments(), {TH009_INCLUDE},
            {build / "libthiran_v0_storage.a", build / "libthiran_v0_async.a",
             build / "libthiran_v0_analysis.a", build / "libthiran_v0_semantic.a",
             build / "libthiran_v0_frontend.a", build / "libthiran_v0_extension.a"}};
}

artifact::NativeToolchain artifactToolchain() {
    const auto config = driverToolchain();
    return {config.compilerExecutable, config.compilerArguments,
            config.includePaths, config.v0StaticLibraries};
}

std::optional<std::string> loadExtensions(const SourceOptions& options,
                                          extension::ExtensionRegistry& registry) {
    for (const auto& path : options.extensions) {
        const auto loaded = registry.load(path);
        if (!loaded.ok()) {
            const auto message = "cannot load extension '" + path + "': " + loaded.message;
            if (options.diagnostics == DiagnosticFormat::Json) {
                std::cerr << "{\"severity\":\"error\",\"code\":" << json(loaded.code)
                          << ",\"message\":" << json(message)
                          << ",\"source\":" << json(options.source)
                          << ",\"start_line\":null,\"start_column\":null,"
                             "\"end_line\":null,\"end_column\":null}\n";
            } else {
                std::cerr << "error[" << loaded.code << "]: " << message << '\n';
            }
            return loaded.code;
        }
    }
    if (options.verbose) {
        for (const auto& identity : registry.operationIdentities())
            std::cerr << "extension=" << identity << '\n';
        if (!options.extensions.empty()) std::cerr << "extension_registry_digest=" << registry.digest() << '\n';
    }
    return {};
}

bool samePath(const fs::path& left, const fs::path& right) {
    std::error_code leftError, rightError;
    const auto a = fs::absolute(left, leftError).lexically_normal();
    const auto b = fs::absolute(right, rightError).lexically_normal();
    return !leftError && !rightError && a == b;
}

artifact::ArtifactBuildResult buildArtifact(const backend::TensorRegion& region,
                                             artifact::NativeBackend backend,
                                             const fs::path& output) {
    artifact::ArtifactBuildOptions options;
    options.output = output;
    return backend == artifact::NativeBackend::Cpu ?
        artifact::buildCpuAot(region, artifactToolchain(), options) :
        artifact::buildGpuAot(region, options);
}

void printExecutionEvidence(artifact::NativeBackend backend,
                            const artifact::ArtifactExecutionResult& execution,
                            unsigned compilerInvocations) {
    if (backend == artifact::NativeBackend::Gpu && execution.gpuEvidence) {
        std::cerr << "backend=gpu payload=ptx thiran_ptx_generation=0 driver_jit=1 kernels="
                  << execution.gpuEvidence->kernelLaunches << " fallback=NONE\n";
    } else {
        std::cerr << "backend=cpu payload=elf-shared-object compiler_invocations="
                  << compilerInvocations << " fallback=NONE\n";
    }
}

int sourceCommand(int argc, char* argv[], std::string_view command) {
    if (argc == 3 && (std::string_view(argv[2]) == "--help" || std::string_view(argv[2]) == "-h")) {
        sourceHelp(command);
        return 0;
    }
    SourceOptions options;
    if (auto error = parseSourceOptions(argc, argv, command, options)) return cliError(*error);
    if (command == "build" && samePath(options.source, options.output))
        return cliError("refusing to overwrite the input source");

    DriverDiagnostic inputFailure;
    auto source = readSourceFile(options.source, inputFailure);
    if (!source) {
        emitDiagnostic(inputFailure, {options.source, {}}, options.diagnostics);
        return 3;
    }
    extension::ExtensionRegistry registry;
    if (loadExtensions(options, registry)) return 3;
    CompilerDriver driver(driverToolchain());
    if (command == "check") {
        auto checked = driver.checkSource(*source, &registry);
        if (!checked.success) return fail(checked, *source, options.diagnostics);
        if (options.verbose) std::cerr << "phase=parse,semantic,verify,ownership status=ok\n";
        return 0;
    }

    const auto target = *options.backend == artifact::NativeBackend::Cpu ?
        backend::NativeTarget::Cpu : backend::NativeTarget::Gpu;
    auto lowered = driver.extractNative(*source, options.entry, target, &registry);
    if (!lowered.success) return fail(lowered, *source, options.diagnostics);
    if (options.verbose) {
        std::cerr << "backend=" << (*options.backend == artifact::NativeBackend::Cpu ? "cpu" : "gpu")
                  << " entry=" << options.entry << " fallback=NONE\n";
    }
    if (command == "build") {
        const auto built = buildArtifact(*lowered.region, *options.backend, options.output);
        if (!built.success) return artifactFailure(*built.error, "build failed");
        if (options.verbose)
            std::cerr << "artifact=" << fs::absolute(options.output).string()
                      << " plan_digest=" << built.manifest->planDigest
                      << " payload_digest=" << built.manifest->payloadDigest << '\n';
        return 0;
    }

    TemporaryDirectory temporary;
    if (!temporary.valid()) return cliError("cannot create controlled run directory");
    const auto path = temporary.path() / "program.tha";
    const auto built = buildArtifact(*lowered.region, *options.backend, path);
    if (!built.success) return artifactFailure(*built.error, "native source build failed");
    const auto loaded = artifact::loadArtifact(path);
    if (!loaded.ok()) return artifactFailure(*loaded.error, "temporary artifact load failed");
    const auto executed = artifact::executeArtifact(*loaded.artifact, {}, options.device);
    if (!executed.ok()) return artifactFailure(*executed.error, "native source execution failed");
    std::cout << artifact::formatArtifactValue(*executed.value) << '\n';
    if (options.verbose) printExecutionEvidence(*options.backend, executed, 1);
    return 0;
}

int artifactCommand(int argc, char* argv[]) {
    if (argc == 2 || (argc == 3 && (std::string_view(argv[2]) == "--help" || std::string_view(argv[2]) == "-h"))) {
        artifactHelp();
        return argc == 2 ? 2 : 0;
    }
    if (argc < 4) return cliError("artifact requires inspect or run and an artifact path");
    const std::string command = argv[2];
    if (command != "inspect" && command != "run") return cliError("unknown artifact command '" + command + "'");
    if (std::string_view(argv[3]) == "--help") { artifactHelp(); return 0; }
    int device = 0;
    bool verbose = false;
    bool optionSeen = false;
    for (int index = 4; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--device" && index + 1 < argc) {
            optionSeen = true;
            if (!parseInteger(argv[++index], device)) return cliError("invalid device ordinal");
        } else if (argument == "--verbose") { verbose = true; optionSeen = true; }
        else return cliError("unknown or incomplete artifact option '" + argument + "'");
    }
    if (command == "inspect" && optionSeen) return cliError("artifact inspect accepts no options");
    auto loaded = artifact::loadArtifact(argv[3]);
    if (!loaded.ok()) return artifactFailure(*loaded.error, "cannot load artifact '" + std::string(argv[3]) + "'");
    if (command == "inspect") {
        std::cout << artifact::inspectArtifact(*loaded.artifact);
        return 0;
    }
    if (!loaded.artifact->manifest.entry.parameters.empty())
        return cliError("artifact run supports only zero-parameter V0 entries");
    auto executed = artifact::executeArtifact(*loaded.artifact, {}, device);
    if (!executed.ok()) return artifactFailure(*executed.error, "artifact execution failed");
    std::cout << artifact::formatArtifactValue(*executed.value) << '\n';
    if (verbose) printExecutionEvidence(loaded.artifact->manifest.backend, executed, 0);
    return 0;
}

int modelFailure(const model::ModelError& error, std::string_view operation) {
    std::cerr << "error[" << error.code << "]: " << operation << ": " << error.message << '\n';
    return error.category == model::ModelErrorCategory::Execution ? 4 : 3;
}

int modelCommand(int argc, char* argv[]) {
    if (argc == 2 || (argc == 3 && (std::string_view(argv[2]) == "--help" || std::string_view(argv[2]) == "-h"))) {
        modelHelp();
        return argc == 2 ? 2 : 0;
    }
    if (argc < 4) return cliError("model requires inspect or run and a model path");
    const std::string command = argv[2];
    if (command != "inspect" && command != "run") return cliError("unknown model command '" + command + "'");
    if (std::string_view(argv[3]) == "--help") { modelHelp(); return 0; }
    auto loaded = model::loadModelBundle(argv[3]);
    if (!loaded.ok()) return modelFailure(*loaded.error, "cannot load model '" + std::string(argv[3]) + "'");
    if (command == "inspect") {
        if (argc != 4) return cliError("model inspect accepts no options");
        std::cout << model::inspectModelBundle(*loaded.bundle);
        return 0;
    }
    std::optional<model::ModelBackend> backend;
    std::optional<float> input;
    int device = 0;
    bool verbose = false;
    for (int index = 4; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--backend" && index + 1 < argc) {
            const std::string value = argv[++index];
            if (value == "cpu") backend = model::ModelBackend::Cpu;
            else if (value == "gpu") backend = model::ModelBackend::Gpu;
            else return cliError("invalid model backend '" + value + "'");
        } else if (argument == "--input" && index + 1 < argc) {
            const std::string spelling = argv[++index];
            float value = 0;
            const auto parsed = std::from_chars(spelling.data(), spelling.data() + spelling.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != spelling.data() + spelling.size())
                return cliError("invalid f32 model input");
            input = value;
        } else if (argument == "--device" && index + 1 < argc) {
            if (!parseInteger(argv[++index], device)) return cliError("invalid device ordinal");
        } else if (argument == "--verbose") verbose = true;
        else return cliError("unknown or incomplete model option '" + argument + "'");
    }
    if (!backend || !input) return cliError("model run requires --backend and --input");
    if (loaded.bundle->publicInputs.size() != 1 ||
        loaded.bundle->publicInputs[0] != artifact::ArtifactType{
            artifact::ValueKind::Tensor, storage::DType::F32, 1, {std::uint64_t{1}}})
        return cliError("V0 model CLI requires one Tensor<f32,1>[1] public input");
    const auto value = storage::Tensor::materializeF32({1}, {*input});
    auto executed = model::executeModel(*loaded.bundle, *backend, {value}, device);
    if (!executed.ok()) return modelFailure(*executed.error, "model execution failed");
    std::cout << artifact::formatArtifactValue(*executed.value) << '\n';
    if (verbose) {
        if (*backend == model::ModelBackend::Gpu && executed.gpuEvidence)
            std::cerr << "backend=gpu payload=ptx thiran_ptx_generation=0 driver_jit=1 kernels="
                      << executed.gpuEvidence->kernelLaunches << " fallback=NONE\n";
        else std::cerr << "backend=cpu payload=elf-shared-object compiler_invocations=0 source_parses=0 fallback=NONE\n";
    }
    return 0;
}

} // namespace

void printPrimaryHelp(std::ostream& output) {
    output <<
        "thiran - numerical systems language V0 tooling\n\n"
        "Usage:\n"
        "  thiran check <source.th> [--extension <library>]...\n"
        "  thiran run <source.th> --backend <cpu|gpu> [--extension <library>]...\n"
        "  thiran build <source.th> --backend <cpu|gpu> -o <artifact.tha> [--extension <library>]...\n"
        "  thiran artifact <inspect|run> ...\n"
        "  thiran model <inspect|run> ...\n\n"
        "Backends are explicit; GPU requests never fall back to CPU. Source run/build\n"
        "currently require a zero-parameter entry (default: main). Extensions are\n"
        "loaded only from repeated explicit --extension paths. Artifacts embed lowered\n"
        "extension recipes and run without the extension library. Use\n"
        "'thiran <command> --help' for command-specific help.\n\n"
        "Compatibility commands:\n"
        "  thiran --version\n"
        "  thiran doctor\n"
        "  thiran --plan/--emit-plan/--emit-region-executor ...\n";
}

bool isWorkflowCommand(int argc, char* argv[]) {
    if (argc < 2) return false;
    const std::string command = argv[1];
    if (command == "check" || command == "run" || command == "build" ||
        command == "artifact" || command == "model") return true;
    if (!command.empty() && command.front() != '-' && command != "doctor" &&
        fs::path(command).extension() != ".th") return true;
    return false;
}

int runWorkflowCli(int argc, char* argv[]) {
    try {
        if (argc < 2) { printPrimaryHelp(std::cout); return 2; }
        const std::string command = argv[1];
        if (command == "check" || command == "run" || command == "build")
            return sourceCommand(argc, argv, command);
        if (command == "artifact") return artifactCommand(argc, argv);
        if (command == "model") return modelCommand(argc, argv);
        return cliError("unknown command '" + command + "'");
    } catch (const std::exception& error) {
        std::cerr << "error[TH022-INTERNAL]: tooling request failed: " << error.what() << '\n';
        return 5;
    } catch (...) {
        std::cerr << "error[TH022-INTERNAL]: tooling request failed\n";
        return 5;
    }
}

} // namespace thiran::v0::tooling
