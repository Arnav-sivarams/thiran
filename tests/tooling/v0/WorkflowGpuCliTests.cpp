#include "tooling/v0/Process.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
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
    std::ofstream output(path); output << bytes;
    if (!output) throw std::runtime_error("fixture write failed");
}
tooling::ProcessResult cli(std::vector<std::string> arguments) {
    return tooling::runProcess({TH022_CLI, std::move(arguments)});
}
}

int main() {
    std::string pattern = "/tmp/th022-gpu-cli-XXXXXX";
    if (!::mkdtemp(pattern.data())) return 1;
    const fs::path root = pattern;
    try {
        const auto source = root / "basic.th";
        write(source, "fn main()->Tensor<i64,1>{let A=[1,2,3];let B=[4,5,6];return A+B}\n");
        auto run = cli({"run", source.string(), "--backend", "gpu", "--verbose"});
        if (run.exitStatus != 0 &&
            (run.standardError.find("GPU-DRIVER") != std::string::npos ||
             run.standardError.find("CUDA") != std::string::npos)) {
            std::cerr << "physical CUDA device unavailable\n";
            fs::remove_all(root);
            return 77;
        }
        require(run.exitStatus == 0 && run.standardOutput.find("[5,7,9]") != std::string::npos &&
                run.standardError.find("backend=gpu") != std::string::npos &&
                run.standardError.find("kernels=") != std::string::npos &&
                run.standardError.find("fallback=NONE") != std::string::npos,
                "physical GPU source run");
        const auto artifact = root / "gpu.tha";
        require(cli({"build", source.string(), "--backend", "gpu", "-o", artifact.string()}).exitStatus == 0,
                "GPU artifact build");
        auto inspect = cli({"artifact", "inspect", artifact.string()});
        require(inspect.exitStatus == 0 && inspect.standardOutput.find("backend=gpu") != std::string::npos &&
                inspect.standardOutput.find("payload_kind=ptx") != std::string::npos,
                "GPU artifact inspect");
        auto artifactRun = cli({"artifact", "run", artifact.string(), "--verbose"});
        require(artifactRun.exitStatus == 0 && artifactRun.standardOutput.find("[5,7,9]") != std::string::npos &&
                artifactRun.standardError.find("thiran_ptx_generation=0") != std::string::npos &&
                artifactRun.standardError.find("fallback=NONE") != std::string::npos,
                "physical GPU artifact run");

        const auto extensionSource = root / "extension.th";
        write(extensionSource,
              "fn main()->Tensor<f32,1>{let x=[1.0,2.0,3.5];return research_square_linear(x)}\n");
        const auto extension = root / "research.so";
        fs::copy_file(TH022_EXTENSION, extension);
        auto extensionRun = cli({"run", extensionSource.string(), "--backend", "gpu",
                                 "--extension", extension.string(), "--verbose"});
        require(extensionRun.exitStatus == 0 && extensionRun.standardOutput.find("[2,6,15.75]") != std::string::npos &&
                extensionRun.standardError.find("fallback=NONE") != std::string::npos,
                "extension physical GPU run");
        const auto cpuOnlySource = root / "cpu-only-extension.th";
        write(cpuOnlySource,
              "fn main()->Tensor<f32,1>{let x=[1.0,2.0];return research_cpu_only(x)}\n");
        auto unsupportedExtension = cli({"run", cpuOnlySource.string(), "--backend", "gpu",
                                         "--extension", extension.string()});
        require(unsupportedExtension.exitStatus != 0 &&
                unsupportedExtension.standardError.find("BACKEND-UNSUPPORTED") != std::string::npos &&
                unsupportedExtension.standardError.find("fallback: NONE") != std::string::npos,
                "GPU-unsupported extension did not fail without fallback");
        const auto extensionArtifact = root / "extension-gpu.tha";
        require(cli({"build", extensionSource.string(), "--backend", "gpu", "--extension",
                     extension.string(), "-o", extensionArtifact.string()}).exitStatus == 0,
                "extension GPU build");
        fs::remove(extension);
        auto pluginFree = cli({"artifact", "run", extensionArtifact.string(), "--verbose"});
        require(pluginFree.exitStatus == 0 && pluginFree.standardOutput.find("[2,6,15.75]") != std::string::npos &&
                pluginFree.standardError.find("fallback=NONE") != std::string::npos,
                "plugin-free physical GPU extension artifact");
        require(cli({"run", source.string(), "--backend", "gpu", "--device", "999999"}).exitStatus != 0,
                "invalid GPU device");
        std::cout << "WorkflowGpuCliTests PASS " << checks << " checks\n";
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "WorkflowGpuCliTests FAIL after " << checks << " checks: " << error.what() << '\n';
        fs::remove_all(root);
        return 1;
    }
}
