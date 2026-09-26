#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {
int checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
std::string read(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}
void contains(const std::string& bytes, const std::string& needle) {
    require(bytes.find(needle) != std::string::npos, "missing frozen marker: " + needle);
}
}

int main() {
    try {
        const fs::path source = THIRAN_SOURCE_DIR;
        const auto specification = read(source / "docs/spec/ROBUSTNESS_REPRODUCIBILITY_V0.md");
        const auto harness = read(source / "robustness/th024/qualify.py");
        const auto manifest = read(source / "robustness/results/TH-024-machine.json");
        contains(specification, "Generator seed: `240024`");
        contains(specification, "Diagnostic repetitions: 20");
        contains(specification, "CPU soak: 1,000 iterations");
        contains(specification, "GPU soak: 500 iterations");
        contains(specification, "Extension shared objects and native payloads remain trusted native code");
        contains(harness, "os.replace(temporary, MANIFEST)");
        contains(harness, "ARTIFACT_BUILDS = 10");
        contains(harness, "DIAGNOSTIC_REPEATS = 20");
        contains(manifest, "\"checkpoint\": \"TH-024\"");
        contains(manifest, "3b9a0cb0147fc28180c76c013e633c22eeb247fb");
        contains(manifest, "04f7c9d26d4dd749d308328dd69bd5f83f0c4c3786fe380de65a7d4fc0858e31");
        std::cout << "TH024StructureTests PASS " << checks << " checks\n";
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "TH024StructureTests FAIL after " << checks << " checks: " << failure.what() << '\n';
        return 1;
    }
}
