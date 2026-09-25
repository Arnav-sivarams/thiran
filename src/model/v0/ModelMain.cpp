#include "model/v0/ModelBundle.hpp"

#include <charconv>
#include <iostream>
#include <string>

namespace artifact = thiran::v0::artifact;
namespace model = thiran::v0::model;
namespace storage = thiran::v0::storage;

namespace {

void help() {
    std::cout <<
        "thiran-model - native model bundle V0 inspector/runtime\n\n"
        "Usage:\n"
        "  thiran-model inspect <model.thm>\n"
        "  thiran-model run <model.thm> --backend <cpu|gpu> --input <f32> [--device <ordinal>]\n\n"
        "The run command accepts public model input only. Frozen parameters are\n"
        "loaded from the bundle. No compiler, evaluator, or framework fallback.\n";
}

int fail(const model::ModelError& error) {
    std::cerr << error.code << ": " << error.message << '\n';
    return error.category == model::ModelErrorCategory::Load ? 3 : 4;
}

} // namespace

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
    auto loaded = model::loadModelBundle(argv[2]);
    if (!loaded.ok()) return fail(*loaded.error);
    if (command == "inspect") {
        if (argc != 3) { std::cerr << "inspect accepts no additional arguments\n"; return 2; }
        std::cout << model::inspectModelBundle(*loaded.bundle);
        return 0;
    }

    std::optional<model::ModelBackend> backend;
    std::optional<float> input;
    int device = 0;
    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--backend" && index + 1 < argc) {
            const std::string spelling = argv[++index];
            if (spelling == "cpu") backend = model::ModelBackend::Cpu;
            else if (spelling == "gpu") backend = model::ModelBackend::Gpu;
            else { std::cerr << "invalid backend: " << spelling << '\n'; return 2; }
        } else if (argument == "--input" && index + 1 < argc) {
            const std::string spelling = argv[++index];
            float value = 0.0f;
            const auto parsed = std::from_chars(spelling.data(), spelling.data() + spelling.size(), value);
            if (parsed.ec != std::errc{} || parsed.ptr != spelling.data() + spelling.size()) {
                std::cerr << "invalid f32 public input\n";
                return 2;
            }
            input = value;
        } else if (argument == "--device" && index + 1 < argc) {
            const std::string spelling = argv[++index];
            const auto parsed = std::from_chars(spelling.data(), spelling.data() + spelling.size(), device);
            if (parsed.ec != std::errc{} || parsed.ptr != spelling.data() + spelling.size()) {
                std::cerr << "invalid device ordinal\n";
                return 2;
            }
        } else {
            std::cerr << "unexpected or incomplete argument: " << argument << '\n';
            return 2;
        }
    }
    if (!backend || !input) {
        std::cerr << "run requires explicit --backend and --input\n";
        return 2;
    }
    if (loaded.bundle->publicInputs.size() != 1 ||
        loaded.bundle->publicInputs[0] != artifact::ArtifactType{
            artifact::ValueKind::Tensor, storage::DType::F32, 1, {std::uint64_t{1}}}) {
        std::cerr << "MODEL-CLI-ABI: V0 CLI requires one scalar-spelled Tensor<f32,1>[1] public input\n";
        return 2;
    }
    auto value = storage::Tensor::materializeF32({1}, {*input});
    auto executed = model::executeModel(*loaded.bundle, *backend, {value}, device);
    if (!executed.ok()) return fail(*executed.error);
    std::cout << artifact::formatArtifactValue(*executed.value) << '\n';
    if (*backend == model::ModelBackend::Gpu && executed.gpuEvidence) {
        std::cerr << "backend=gpu payload=ptx thiran_ptx_generation=0 driver_jit=1 kernels="
                  << executed.gpuEvidence->kernelLaunches
                  << " observations=" << executed.gpuEvidence->observations
                  << " released_reservations=" << executed.gpuEvidence->releasedReservations
                  << " fallback=NONE\n";
    } else {
        std::cerr << "backend=cpu payload=elf-shared-object compiler_invocations=0 source_parses=0 fallback=NONE\n";
    }
    return 0;
}
