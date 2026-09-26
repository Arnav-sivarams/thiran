#include "tooling/v0/Process.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace tooling = thiran::v0::tooling;

namespace {
int checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void write(const fs::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << bytes;
    if (!output) throw std::runtime_error("cannot write fixture " + path.string());
}
std::vector<char> read(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), {}};
}
void write(const fs::path& path, const std::vector<char>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("cannot write binary fixture");
}
tooling::ProcessResult cli(std::vector<std::string> arguments) {
    return tooling::runProcess({TH022_CLI, std::move(arguments)});
}
fs::path temporary() {
    std::string pattern = "/tmp/th022 workflow XXXXXX";
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
    return pattern;
}
}

int main() {
    const auto root = temporary();
    try {
        const std::string basic =
            "fn main() -> Tensor<i64,2> {\n"
            "  let A = [1,2;3,4]\n  let B = [5,6;7,8]\n  return A + B\n}\n";
        const std::string extension =
            "fn main() -> Tensor<f32,1> {\n"
            "  let x = [1.0,2.0,3.5]\n  return research_square_linear(x)\n}\n";
        const auto source = root / "source with spaces.th";
        const auto extensionSource = root / "research source.th";
        write(source, basic);
        write(extensionSource, extension);

        auto help = cli({"--help"});
        require(help.exitStatus == 0 && help.standardOutput.find("thiran check") != std::string::npos &&
                help.standardOutput.find("thiran artifact") != std::string::npos, "primary help");
        require(cli({"check", "--help"}).exitStatus == 0, "check help");
        require(cli({"run", "--help"}).exitStatus == 0, "run help");
        require(cli({"build", "--help"}).exitStatus == 0, "build help");
        require(cli({"artifact", "--help"}).exitStatus == 0, "artifact help");
        require(cli({"model", "--help"}).exitStatus == 0, "model help");
        auto unknown = cli({"definitely-not-a-command"});
        require(unknown.exitStatus == 2 && unknown.standardError.find("TH022-CLI") != std::string::npos,
                "unknown command");
        require(cli({"check", source.string(), "--no-such-option"}).exitStatus == 2,
                "unknown option");
        require(cli({"run", source.string()}).exitStatus == 2, "missing backend");
        require(cli({"build", source.string(), "--backend", "cpu"}).exitStatus == 2,
                "missing output");
        require(cli({"run", source.string(), "--backend", "wat"}).exitStatus == 2,
                "invalid backend");

        auto missing = cli({"check", (root / "missing.th").string()});
        require(missing.exitStatus != 0 && missing.standardError.find("INPUT-UNREADABLE") != std::string::npos,
                "missing source");
        const auto unreadable = root / "unreadable.th"; write(unreadable, basic);
        fs::permissions(unreadable, fs::perms::none);
        auto unreadableResult = cli({"check", unreadable.string()});
        fs::permissions(unreadable, fs::perms::owner_read | fs::perms::owner_write);
        require(unreadableResult.exitStatus != 0 &&
                unreadableResult.standardError.find("INPUT-UNREADABLE") != std::string::npos,
                "unreadable source");
        const auto empty = root / "empty.th"; write(empty, "");
        require(cli({"check", empty.string()}).exitStatus != 0, "empty source");
        const auto lexical = root / "lexical.th"; write(lexical, "fn main()->i64{return @}\n");
        require(cli({"check", lexical.string()}).exitStatus != 0, "lexical failure");
        const auto parse = root / "parse.th"; write(parse, "fn main( {\n");
        auto parseFailure = cli({"check", parse.string()});
        require(parseFailure.exitStatus != 0 && parseFailure.standardError.find("error[SYNTAX]") != std::string::npos &&
                parseFailure.standardError.find("^") != std::string::npos, "located parse diagnostic");
        auto jsonFailure = cli({"check", parse.string(), "--diagnostic-format", "json"});
        require(jsonFailure.exitStatus != 0 && jsonFailure.standardError.find("\"severity\":\"error\"") != std::string::npos &&
                jsonFailure.standardError.find("\"start_line\":1") != std::string::npos,
                "JSON diagnostics");
        const auto semantic = root / "semantic.th";
        write(semantic, "fn main()->i64 { return missing }\n");
        auto semanticFailure = cli({"check", semantic.string()});
        require(semanticFailure.exitStatus != 0 &&
                semanticFailure.standardError.find("TH005-UNDEFINED-NAME") != std::string::npos,
                "semantic identity");
        const auto typeError = root / "type.th";
        write(typeError, "fn main()->i64 { return 1 + true }\n");
        auto typed = cli({"check", typeError.string()});
        require(typed.exitStatus != 0 && typed.standardError.find("TH005-OPERAND-TYPE") != std::string::npos,
                "type diagnostic");
        const auto rankError = root / "rank.th";
        write(rankError, "fn main()->Tensor<i64,1> { let A=[1,2]; return A*A }\n");
        auto ranked = cli({"check", rankError.string()});
        require(ranked.exitStatus != 0 && ranked.standardError.find("TH005-MATMUL-RANK") != std::string::npos,
                "rank diagnostic");
        const auto ownership = root / "ownership.th";
        write(ownership, "fn main()->i64 { let A=[1,2;3,4]; let B=move(A); return A[0,0] }\n");
        auto ownershipFailure = cli({"check", ownership.string()});
        require(ownershipFailure.exitStatus != 0 &&
                ownershipFailure.standardError.find("TH006-USE-AFTER-MOVE") != std::string::npos,
                "ownership identity");
        auto unloaded = cli({"check", extensionSource.string()});
        require(unloaded.exitStatus != 0 && unloaded.standardError.find("TH005-UNKNOWN-FUNCTION") != std::string::npos,
                "unloaded extension operation");

        auto checked = cli({"check", source.string()});
        require(checked.exitStatus == 0 && checked.standardOutput.empty() && checked.standardError.empty(),
                "valid check streams");
        const auto relativeSource = fs::relative(source, fs::current_path());
        require(cli({"check", relativeSource.string()}).exitStatus == 0, "relative source path");
        const std::string expected =
            "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\","
            "\"shape\":[2,2],\"values\":[6,8,10,12]}\n";
        auto run = cli({"run", source.string(), "--backend", "cpu", "--verbose"});
        require(run.exitStatus == 0 && run.standardOutput == expected &&
                run.standardError.find("compiler_invocations=1") != std::string::npos &&
                run.standardError.find("fallback=NONE") != std::string::npos,
                "native CPU source run");
#if !TH022_NATIVE_GPU
        auto unavailableGpu = cli({"run", source.string(), "--backend", "gpu"});
        require(unavailableGpu.exitStatus != 0 &&
                unavailableGpu.standardError.find("GPU-BACKEND-NOT-BUILT") != std::string::npos,
                "CPU-only explicit GPU failure");
#endif

        const auto artifact = root / "CPU artifact with spaces.tha";
        auto built = cli({"build", source.string(), "--backend", "cpu", "-o", artifact.string(), "--verbose"});
        require(built.exitStatus == 0 && fs::exists(artifact) && built.standardOutput.empty(), "CPU build");
        auto inspected = cli({"artifact", "inspect", artifact.string()});
        require(inspected.exitStatus == 0 && inspected.standardOutput.find("backend=cpu\n") != std::string::npos &&
                inspected.standardOutput.find("payload_kind=elf-shared-object") != std::string::npos,
                "CPU inspect");
        auto artifactRun = cli({"artifact", "run", artifact.string(), "--verbose"});
        require(artifactRun.exitStatus == 0 && artifactRun.standardOutput == expected &&
                artifactRun.standardError.find("compiler_invocations=0") != std::string::npos,
                "CPU artifact run");
        require(cli({"build", source.string(), "--backend", "cpu", "-o", source.string()}).exitStatus == 2,
                "input overwrite refusal");
        require(cli({"build", source.string(), "--backend", "cpu", "-o",
                     (root / "absent" / "output.tha").string()}).exitStatus != 0,
                "missing output parent");
        const auto unsupported = root / "unsupported.th";
        const auto unsupportedArtifact = root / "must-not-exist.tha";
        write(unsupported, "fn main()->Tensor<i64,2>{let A=[1,2;3,4];return A*A}\n");
        auto unsupportedBuild = cli({"build", unsupported.string(), "--backend", "cpu",
                                     "-o", unsupportedArtifact.string()});
        require(unsupportedBuild.exitStatus != 0 && !fs::exists(unsupportedArtifact) &&
                unsupportedBuild.standardError.find("BACKEND-UNSUPPORTED") != std::string::npos &&
                unsupportedBuild.standardError.find("fallback: NONE") != std::string::npos,
                "unsupported operation has no fallback or output");

        const auto extensionCopy = root / "research extension with spaces.so";
        fs::copy_file(TH022_EXTENSION, extensionCopy);
        require(cli({"check", extensionSource.string(), "--extension", extensionCopy.string()}).exitStatus == 0,
                "extension check");
        auto extensionRun = cli({"run", extensionSource.string(), "--backend", "cpu",
                                 "--extension", extensionCopy.string()});
        require(extensionRun.exitStatus == 0 && extensionRun.standardOutput.find("[2,6,15.75]") != std::string::npos,
                "extension CPU run");
        const auto extensionArtifact = root / "extension.tha";
        require(cli({"build", extensionSource.string(), "--backend", "cpu", "--extension",
                     extensionCopy.string(), "-o", extensionArtifact.string()}).exitStatus == 0,
                "extension CPU build");
        fs::remove(extensionCopy);
        auto pluginFree = cli({"artifact", "run", extensionArtifact.string()});
        require(pluginFree.exitStatus == 0 && pluginFree.standardOutput.find("[2,6,15.75]") != std::string::npos,
                "plugin-free extension artifact");
        auto missingExtension = cli({"check", extensionSource.string(), "--extension",
                                     (root / "gone.so").string()});
        require(missingExtension.exitStatus != 0 && missingExtension.standardError.find("TH021-LOAD") != std::string::npos,
                "missing extension diagnostic");
        auto badAbi = cli({"check", extensionSource.string(), "--extension", TH022_BAD_ABI});
        require(badAbi.exitStatus != 0 && badAbi.standardError.find("TH021-ABI") != std::string::npos,
                "extension ABI diagnostic");
        const auto invalidSharedObject = root / "not-an-extension.so";
        write(invalidSharedObject, "not an ELF shared object\n");
        require(cli({"check", extensionSource.string(), "--extension",
                     invalidSharedObject.string()}).standardError.find("TH021-LOAD") != std::string::npos,
                "invalid shared object diagnostic");
        require(cli({"check", extensionSource.string(), "--extension",
                     TH022_MALFORMED}).standardError.find("TH021-DESCRIPTOR") != std::string::npos,
                "malformed descriptor diagnostic");
        auto duplicate = cli({"check", extensionSource.string(), "--extension", TH022_EXTENSION,
                              "--extension", TH022_DUPLICATE_OP});
        require(duplicate.exitStatus != 0 && duplicate.standardError.find("TH021-DUPLICATE-OP") != std::string::npos,
                "duplicate extension operation diagnostic");

        require(cli({"artifact", "inspect", (root / "missing.tha").string()}).exitStatus != 0,
                "missing artifact");
        auto bytes = read(artifact);
        write(root / "truncated.tha", std::vector<char>(bytes.begin(), bytes.begin() + 12));
        auto truncated = cli({"artifact", "inspect", (root / "truncated.tha").string()});
        require(truncated.exitStatus != 0 && truncated.standardError.find("ARTIFACT-TRUNCATED") != std::string::npos,
                "truncated artifact");
        auto corrupt = bytes; corrupt.back() ^= 1;
        write(root / "bad-digest.tha", corrupt);
        auto badDigest = cli({"artifact", "inspect", (root / "bad-digest.tha").string()});
        require(badDigest.exitStatus != 0 && badDigest.standardError.find("digest mismatch") != std::string::npos,
                "bad artifact digest");
        auto badAbiBytes = bytes;
        badAbiBytes[12] = 99; badAbiBytes[13] = badAbiBytes[14] = badAbiBytes[15] = 0;
        write(root / "bad-abi.tha", badAbiBytes);
        auto artifactAbi = cli({"artifact", "inspect", (root / "bad-abi.tha").string()});
        require(artifactAbi.exitStatus != 0 && artifactAbi.standardError.find("ARTIFACT-COMPILER-ABI") != std::string::npos,
                "artifact ABI diagnostic");

        std::cout << "WorkflowCliTests PASS " << checks << " checks\n";
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "WorkflowCliTests FAIL after " << checks << " checks: " << error.what() << '\n';
        fs::remove_all(root);
        return 1;
    }
}
