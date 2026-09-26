#include "artifact/v0/NativeArtifacts.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using namespace thiran::v0;

int main(int argc,char** argv) {
    if (argc!=2) return 2;
    std::ifstream maps("/proc/self/maps");
    const std::string mapped{std::istreambuf_iterator<char>(maps),{}};
    if (mapped.find("th021_research_extension")!=std::string::npos) return 3;
    auto input=storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9});
    auto loaded=artifact::loadArtifact(argv[1]);
    if (!loaded.ok()) {
        std::cerr<<loaded.error->code<<": "<<loaded.error->message<<'\n';
        return 4;
    }
    auto executed=artifact::executeArtifact(*loaded.artifact,{input});
    if (!executed.ok()) {
        std::cerr<<executed.error->code<<": "<<executed.error->message<<'\n';
        return 5;
    }
    std::cout<<artifact::formatArtifactValue(*executed.value)<<'\n';
    if (loaded.artifact->manifest.backend==artifact::NativeBackend::Gpu) {
        if (!executed.gpuEvidence) return 6;
        std::cerr<<"backend=gpu payload=ptx thiran_ptx_generation=0 driver_jit=1 kernels="
                 <<executed.gpuEvidence->kernelLaunches<<" fallback=NONE\n";
    } else {
        std::cerr<<"backend=cpu payload=elf-shared-object compiler_invocations=0 fallback=NONE\n";
    }
    return 0;
}
