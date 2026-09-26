#include "artifact/v0/NativeArtifacts.hpp"
#include "backend/v0/NativeCpu.hpp"
#include "backend/v0/NativeGpu.hpp"
#include "extension/v0/Extension.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"
#include "tooling/v0/BuildConfig.hpp"

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs=std::filesystem;
using namespace thiran::v0;
namespace e=extension;namespace s=semantic;namespace b=backend;namespace a=artifact;
static int checks=0;
static void require(bool yes,const std::string& message){++checks;if(!yes)throw std::runtime_error(message);}
static fs::path temporary(){std::string pattern="/tmp/th021-extension-gpu-XXXXXX";if(!::mkdtemp(pattern.data()))throw std::runtime_error("mkdtemp failed");return pattern;}
static s::RuntimeValue runtimeTensor(const storage::Tensor& tensor){return s::RuntimeValue{s::RuntimeTensor{s::TypeKind::F32,
    std::vector<std::int64_t>(tensor.descriptor().shape.begin(),tensor.descriptor().shape.end()),{},tensor.logicalF32Values()}};}
static a::NativeToolchain toolchain(){const fs::path build=TH021_BUILD;return {TH021_CXX,tooling::configuredHostCompilerArguments(),{TH021_INCLUDE},
    {build/"libthiran_v0_storage.a",build/"libthiran_v0_async.a",build/"libthiran_v0_analysis.a",build/"libthiran_v0_semantic.a",build/"libthiran_v0_frontend.a",build/"libthiran_v0_extension.a"}};}
static std::pair<s::Module,b::TensorRegion> compileRegion(){
    e::ExtensionRegistry registry;auto loaded=registry.load(TH021_EXTENSION);if(!loaded.ok())throw std::runtime_error(loaded.message);
    auto parsed=parse("fn f(x:Tensor<f32,1>)->Tensor<f32,1>{return research_square_linear(x)}","<th021-gpu>");
    auto analyzed=s::analyze(*parsed.module,registry);if(!analyzed.module)throw std::runtime_error(analyzed.diagnostics.front().format());
    auto facts=analysis::analyze(*analyzed.module);auto lowered=b::extractStrictGpu(*analyzed.module,facts,"f",false);
    if(!lowered.ok())throw std::runtime_error(lowered.diagnostic);
    return {*analyzed.module,*lowered.region};
}
static b::TensorRegion compileRatioRegion(){
    e::ExtensionRegistry registry;auto loaded=registry.load(TH021_EXTENSION);if(!loaded.ok())throw std::runtime_error(loaded.message);
    auto parsed=parse("fn ratio(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{return research_checked_ratio(x,y)}","<th021-gpu-trap>");
    auto analyzed=s::analyze(*parsed.module,registry);if(!analyzed.module)throw std::runtime_error(analyzed.diagnostics.front().format());
    auto facts=analysis::analyze(*analyzed.module);auto lowered=b::extractStrictGpu(*analyzed.module,facts,"ratio",false);
    if(!lowered.ok())throw std::runtime_error(lowered.diagnostic);
    return *lowered.region;
}
static int child(const fs::path& path){
    std::ifstream maps("/proc/self/maps");std::string mapped{std::istreambuf_iterator<char>(maps),{}};
    if(mapped.find("th021_research_extension")!=std::string::npos)return 8;
    auto input=storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9});
    auto result=a::loadAndExecuteArtifact(path,{input});
    if(!result.ok()){if(result.error)std::cerr<<result.error->code<<": "<<result.error->message<<'\n';return 9;}
    return std::get<storage::Tensor>(*result.value).logicalF32Values()==
        std::vector<float>({2,-.25f,0,3.75f,12,59.8125f,72})?0:10;
}
int main(int argc,char** argv){
 if(argc==3&&std::string(argv[1])=="--artifact-child")return child(argv[2]);
 auto device=b::probeNativeGpu();
 if(!device.deviceAvailable){std::cerr<<"TH021 GPU unavailable: "<<(device.error?device.error->message:"no physical device")<<'\n';return 77;}
 const auto root=temporary();
 try{
    auto [module,region]=compileRegion(); // registry and .so are gone here; region is self-describing.
    auto input=storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9});
    const std::vector<float> expected{2,-.25f,0,3.75f,12,59.8125f,72};
    auto reference=s::evaluateCall(module,"f",{runtimeTensor(input)});require(reference.ok,"reference failed");
    auto facts=analysis::analyze(module);auto cpuLowered=b::extractStrictNative(module,facts,"f",false);require(cpuLowered.ok(),"CPU lowering failed in GPU comparison");
    auto cpuEntry=a::specializeEntry(*cpuLowered.region,{input});auto cpuBuilt=a::buildCpuAot(*cpuLowered.region,toolchain(),{root/"comparison-cpu.tha",{},cpuEntry});
    auto cpu=cpuBuilt.success?a::loadAndExecuteArtifact(root/"comparison-cpu.tha",{input}):a::ArtifactExecutionResult{};
    require(cpu.ok()&&std::get<storage::Tensor>(*cpu.value).logicalF32Values()==expected,"CPU/reference mismatch in three-backend comparison");
    auto gpu=b::executeNativeGpu(region,{input});require(gpu.ok(),gpu.error?gpu.error->message:"GPU failed");
    require(std::get<storage::Tensor>(*gpu.value).logicalF32Values()==expected,"GPU/reference mismatch");
    require(gpu.evidence.device.name.find("NVIDIA")!=std::string::npos&&gpu.evidence.kernelLaunches==1,"physical GPU evidence missing");
    auto zero=storage::Tensor::empty(storage::DType::F32,{0});auto zeroRun=b::executeNativeGpu(region,{zero});
    require(zeroRun.ok()&&std::get<storage::Tensor>(*zeroRun.value).logicalF32Values().empty(),"zero-size GPU extension failed");
    auto invalid=b::executeNativeGpu(region,{input},device.deviceCount+10);require(!invalid.ok()&&invalid.error&&invalid.error->category==b::GpuErrorCategory::InvalidDevice,"invalid GPU device accepted");
    auto ratioRegion=compileRatioRegion();
    auto ratio=b::executeNativeGpu(ratioRegion,{storage::Tensor::materializeF32({2},{4,3}),storage::Tensor::materializeF32({2},{2,0})});
    require(!ratio.ok()&&ratio.error&&ratio.error->code=="TH021-DIVIDE-BY-ZERO","native GPU MayTrap operation published output");

    auto entry=a::specializeEntry(region,{input});auto built=a::buildGpuAot(region,{root/"extension-gpu.tha",{},entry});
    require(built.success,built.error?built.error->message:"GPU AOT failed");
    auto artifactRun=a::loadAndExecuteArtifact(root/"extension-gpu.tha",{input});
    require(artifactRun.ok()&&std::get<storage::Tensor>(*artifactRun.value).logicalF32Values()==expected,"GPU AOT mismatch");
    require(artifactRun.gpuEvidence&&artifactRun.gpuEvidence->kernelLaunches==1,"GPU AOT physical evidence missing");

    a::JitCompiler jit;auto j1=jit.compileGpu(region,{input});auto j2=jit.compileGpu(region,{input});
    require(j1.ok()&&j2.ok()&&!j1.cacheHit&&j2.cacheHit,"GPU extension JIT/cache failed");
    auto jitRun=j1.executable->execute({input});require(jitRun.ok()&&std::get<storage::Tensor>(*jitRun.value).logicalF32Values()==expected,"physical GPU JIT mismatch");
    auto changed=region;for(auto& node:changed.nodes)if(node.extensionOperation){auto& op=*node.extensionOperation;op.semanticVersion="1.0.1";op.canonicalIdentity=op.extensionId+"::"+op.name+"@1.0.1/abi1";op.descriptorDigest=e::operationDigest(op);}
    auto j3=jit.compileGpu(changed,{input});require(j3.ok()&&!j3.cacheHit&&j3.executable->cacheKey()!=j1.executable->cacheKey(),"changed extension GPU JIT identity hit stale cache");
    auto changedRecipe=region;for(auto& node:changedRecipe.nodes)if(node.extensionOperation){auto& op=*node.extensionOperation;op.forward.nodes.back().opcode=e::ScalarOpcode::Subtract;op.descriptorDigest=e::operationDigest(op);}
    auto j4=jit.compileGpu(changedRecipe,{input});require(j4.ok()&&!j4.cacheHit&&j4.executable->cacheKey()!=j1.executable->cacheKey(),"changed GPU lowering recipe hit stale cache");
    auto changedRecipeRun=j4.executable->execute({input});require(changedRecipeRun.ok()&&
        std::get<storage::Tensor>(*changedRecipeRun.value).logicalF32Values()==std::vector<float>({6,.75f,0,.75f,6,45.3125f,90}),
        "changed GPU lowering recipe did not execute its new semantics");
    auto shortInput=storage::Tensor::materializeF32({3},{-1,0,2.5f});auto j5=jit.compileGpu(region,{shortInput});auto j6=jit.compileGpu(region,{shortInput});
    require(j5.ok()&&j6.ok()&&!j5.cacheHit&&j6.cacheHit&&j5.executable->cacheKey()!=j1.executable->cacheKey(),"GPU shape specialization cache identity failed");
    require(!jit.compileGpu(region,{storage::Tensor::materializeI64({3},{1,2,3})}).ok()&&
        !jit.compileGpu(region,{storage::Tensor::materializeF32({1,3},{1,2,3})}).ok(),"incompatible GPU JIT specialization was accepted");

    const auto fresh=root/"fresh-runtime-only";fs::create_directory(fresh);fs::copy_file(root/"extension-gpu.tha",fresh/"program.tha");
    fs::copy_file(TH021_RUNTIME,fresh/"thiran-extension-runtime");fs::permissions(fresh/"thiran-extension-runtime",fs::perms::owner_read|fs::perms::owner_write|fs::perms::owner_exec,fs::perm_options::add);
    require(!fs::exists(fresh/fs::path(TH021_EXTENSION).filename()),"extension .so copied beside runtime-only GPU artifact");
    require(std::distance(fs::directory_iterator(fresh),fs::directory_iterator{})==2,"runtime-only GPU deployment contains compiler/build files");
    const auto prior=fs::current_path();const char* oldPath=std::getenv("PATH");const std::string savedPath=oldPath?oldPath:"";
    fs::current_path(fresh);::setenv("PATH","/nonexistent",1);::setenv("CXX","/nonexistent",1);
    auto runtimeOnly=tooling::runProcess({(fresh/"thiran-extension-runtime").string(),{"program.tha"}});
    ::setenv("PATH",savedPath.c_str(),1);::unsetenv("CXX");fs::current_path(prior);
    require(runtimeOnly.launched&&runtimeOnly.exitStatus==0&&runtimeOnly.standardOutput==
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"f32\",\"shape\":[7],\"values\":[2,-0.25,0,3.75,12,59.8125,72]}\n"&&
        runtimeOnly.standardError.find("thiran_ptx_generation=0")!=std::string::npos&&
        runtimeOnly.standardError.find("driver_jit=1")!=std::string::npos&&runtimeOnly.standardError.find("fallback=NONE")!=std::string::npos,
        "compiler/plugin-absent GPU deployment failed: "+runtimeOnly.standardOutput+runtimeOnly.standardError);
    std::cout<<"V0ExtensionGpuIntegrationTests PASS "<<checks<<" checks\n"
             <<"device="<<device.name<<"\ndriver_version="<<device.driverVersion<<"\ncompute_capability="
             <<device.computeMajor<<'.'<<device.computeMinor<<"\nreference_cpu_gpu=agree kernels="<<gpu.evidence.kernelLaunches
             <<" aot_plugin_runtime_dependency=NONE fallback=NONE\n";
    fs::remove_all(root);return 0;
 }catch(const std::exception& failure){std::cerr<<"V0ExtensionGpuIntegrationTests FAIL after "<<checks<<" checks: "<<failure.what()<<'\n';fs::remove_all(root);return 1;}
}
