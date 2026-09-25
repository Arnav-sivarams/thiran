#include "artifact/v0/NativeArtifacts.hpp"

#include <charconv>
#include <iostream>
#include <string>

namespace artifact = thiran::v0::artifact;

namespace {
void help() {
    std::cout <<
        "thiran-artifact - native artifact V0 inspector/runtime\n\n"
        "Usage:\n"
        "  thiran-artifact inspect <artifact.tha>\n"
        "  thiran-artifact run <artifact.tha> [--device <ordinal>]\n\n"
        "run currently exposes the zero-parameter CLI entry. The C++ runtime API\n"
        "supports typed scalar/tensor arguments. No compiler or framework fallback.\n";
}
int loadFailure(const artifact::ArtifactError& failure) {
    std::cerr << failure.code << ": " << failure.message << '\n';
    return failure.category == artifact::ArtifactErrorCategory::Load ? 3 : 4;
}
}

int main(int argc, char** argv) {
    if (argc == 1 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        help();
        return 0;
    }
    const std::string command = argv[1];
    if ((command != "inspect" && command != "run") || argc < 3) {
        help();
        return 2;
    }
    int device = 0;
    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--device" && index + 1 < argc) {
            const std::string spelling = argv[++index];
            const auto parsed = std::from_chars(spelling.data(), spelling.data() + spelling.size(), device);
            if (parsed.ec != std::errc{} || parsed.ptr != spelling.data() + spelling.size()) {
                std::cerr << "invalid device ordinal\n";
                return 2;
            }
        } else {
            std::cerr << "unexpected argument: " << argument << '\n';
            return 2;
        }
    }
    auto loaded = artifact::loadArtifact(argv[2]);
    if (!loaded.ok()) return loadFailure(*loaded.error);
    if (command == "inspect") {
        std::cout << artifact::inspectArtifact(*loaded.artifact);
        return 0;
    }
    if (!loaded.artifact->manifest.entry.parameters.empty()) {
        std::cerr << "ARTIFACT-CLI-ARGUMENTS: run CLI accepts only zero-parameter entries\n";
        return 2;
    }
    const auto backend = loaded.artifact->manifest.backend;
    auto executed = artifact::executeArtifact(*loaded.artifact, {}, device);
    if (!executed.ok()) {
        if (executed.error->code.starts_with("TH-SPEC-")) {
            std::cout << "{\"status\":\"error\",\"error_id\":\""
                      << executed.error->code << "\"}\n";
            return 0;
        }
        return loadFailure(*executed.error);
    }
    std::cout << artifact::formatArtifactValue(*executed.value) << '\n';
    if (backend == artifact::NativeBackend::Gpu && executed.gpuEvidence) {
        std::cerr << "backend=gpu payload=ptx thiran_ptx_generation=0 driver_jit=1 kernels="
                  << executed.gpuEvidence->kernelLaunches << " fallback=NONE\n";
    } else {
        std::cerr << "backend=cpu payload=elf-shared-object compiler_invocations=0 fallback=NONE\n";
    }
    return 0;
}
