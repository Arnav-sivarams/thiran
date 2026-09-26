#include "artifact/v0/NativeArtifacts.hpp"
#include "autodiff/v0/Autodiff.hpp"
#include "backend/v0/NativeCpu.hpp"
#include "extension/v0/Extension.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"
#include "tooling/v0/BuildConfig.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs=std::filesystem;
using namespace thiran::v0;
namespace e=extension; namespace s=semantic; namespace b=backend; namespace a=artifact;
static int checks=0;
static void require(bool yes,const std::string& message){++checks;if(!yes)throw std::runtime_error(message);}
static fs::path temporary(){std::string pattern="/tmp/th021-extension-XXXXXX";if(!::mkdtemp(pattern.data()))throw std::runtime_error("mkdtemp failed");return pattern;}
static s::Module compile(const std::string& text,e::ExtensionRegistry& registry){
    auto parsed=parse(text,"<th021-test>");require(parsed.module.has_value(),"parse failed: "+text);
    auto analyzed=s::analyze(*parsed.module,registry);require(analyzed.module.has_value(),analyzed.diagnostics.empty()?"analysis failed":analyzed.diagnostics.front().format());
    require(s::verify(*analyzed.module).ok,"semantic verifier rejected extension module");return *analyzed.module;
}
static s::RuntimeValue tensor(std::vector<std::int64_t> shape,std::vector<float> values){return s::RuntimeValue{s::RuntimeTensor{s::TypeKind::F32,std::move(shape),{},std::move(values)}};}
static b::TensorRegion region(const s::Module& module,const std::string& name,bool gpu=false,bool standalone=false){
    auto facts=analysis::analyze(module);require(facts.ok(),"ownership rejected extension");
    auto lowered=gpu?b::extractStrictGpu(module,facts,name,standalone):b::extractStrictNative(module,facts,name,standalone);
    require(lowered.ok(),lowered.diagnostic);return *lowered.region;
}
static a::NativeToolchain toolchain(){const fs::path build=TH021_BUILD;return {TH021_CXX,tooling::configuredHostCompilerArguments(),{TH021_INCLUDE},
    {build/"libthiran_v0_storage.a",build/"libthiran_v0_async.a",build/"libthiran_v0_analysis.a",build/"libthiran_v0_semantic.a",build/"libthiran_v0_frontend.a",build/"libthiran_v0_extension.a"}};}
static bool close(float x,float y,float tolerance=2e-4f){return std::fabs(x-y)<=tolerance;}

static int artifactChild(const fs::path& path){
    std::ifstream maps("/proc/self/maps");std::string text{std::istreambuf_iterator<char>(maps),{}};
    if(text.find("th021_research_extension")!=std::string::npos)return 8;
    auto input=storage::Tensor::materializeF32({7},{-2.0f,-0.5f,0.0f,1.5f,3.0f,7.25f,-9.0f});
    auto result=a::loadAndExecuteArtifact(path,{input});
    if(!result.ok()){if(result.error)std::cerr<<result.error->code<<": "<<result.error->message<<'\n';return 9;}
    auto values=std::get<storage::Tensor>(*result.value).logicalF32Values();
    return values==std::vector<float>({2.0f,-0.25f,0.0f,3.75f,12.0f,59.8125f,72.0f})?0:10;
}

int main(int argc,char** argv){
 if(argc==3&&std::string(argv[1])=="--artifact-child")return artifactChild(argv[2]);
 const auto root=temporary();
 try{
    e::ExtensionRegistry registry;
    require(!registry.load((root/"missing.so").string()).ok(),"missing library accepted");
    {std::ofstream invalid(root/"invalid.so");invalid<<"not an ELF shared object";}
    require(!registry.load((root/"invalid.so").string()).ok(),"invalid library accepted");
    require(registry.load(TH021_NO_ENTRY).code=="TH021-ENTRY","missing entry symbol accepted");
    require(registry.load(TH021_BAD_ABI).code=="TH021-ABI","ABI mismatch accepted");
    require(registry.load(TH021_INIT_FAIL).code=="TH021-INITIALIZATION","initialization failure accepted");
    require(registry.load(TH021_ZERO_OPS).code=="TH021-DESCRIPTOR","zero-operation extension accepted");
    require(registry.load(TH021_MALFORMED).code=="TH021-DESCRIPTOR"&&!registry.find("bad_fixture_op"),
            "malformed extension was not transactionally rejected");
    auto loaded=registry.load(TH021_EXTENSION);require(loaded.ok(),loaded.message);
    require(registry.find("research_square_linear")&&registry.find("research_pair_mix")&&registry.find("research_cpu_only")&&registry.find("research_checked_ratio"),"fixture operations missing");
    require(!registry.load(TH021_EXTENSION).ok(),"duplicate extension identity accepted");
    const e::ScalarNodeV0 identityNodes[]={{e::ScalarOpcode::Input,0,0,0,0}};
    const e::OperationDescriptorV0 collisionOp{"research_square_linear","1.0.0",1,1,2,0,true,e::EffectV0::Pure,e::Reference,false,{identityNodes,1,0,1},nullptr};
    const e::ExtensionDescriptorV0 collision{e::extensionAbiVersion,"org.thiran.collision","1.0.0",&collisionOp,1,nullptr};
    require(registry.registerExtension(collision).code=="TH021-DUPLICATE-OP","cross-extension operation collision accepted");
    const e::DerivativeV0 malformedDerivative{1,1,false,nullptr,0};
    const e::OperationDescriptorV0 malformedDerivativeOp{"bad_gradient","1.0.0",1,1,2,0,true,e::EffectV0::Pure,e::Reference,false,{identityNodes,1,0,1},&malformedDerivative};
    const e::ExtensionDescriptorV0 malformedDerivativeExtension{e::extensionAbiVersion,"org.thiran.bad.gradient","1.0.0",&malformedDerivativeOp,1,nullptr};
    e::ExtensionRegistry derivativeRegistry;require(derivativeRegistry.registerExtension(malformedDerivativeExtension).code=="TH021-DESCRIPTOR","malformed derivative accepted");
    const e::OperationDescriptorV0 invalidShapeOp{"bad_shape","1.0.0",1,1,2,0,false,e::EffectV0::Pure,e::Reference,false,{identityNodes,1,0,1},nullptr};
    const e::ExtensionDescriptorV0 invalidShapeExtension{e::extensionAbiVersion,"org.thiran.bad.shape","1.0.0",&invalidShapeOp,1,nullptr};
    e::ExtensionRegistry invalidShapeRegistry;require(invalidShapeRegistry.registerExtension(invalidShapeExtension).code=="TH021-DESCRIPTOR","invalid shape rule accepted");
    const e::ScalarNodeV0 invalidRecipeNodes[]={{e::ScalarOpcode::Input,0,0,0,0},{e::ScalarOpcode::Add,0,2,0,0}};
    const e::OperationDescriptorV0 invalidRecipeOp{"bad_recipe","1.0.0",1,1,2,0,true,e::EffectV0::Pure,e::Reference,false,{invalidRecipeNodes,2,1,1},nullptr};
    const e::ExtensionDescriptorV0 invalidRecipeExtension{e::extensionAbiVersion,"org.thiran.bad.recipe","1.0.0",&invalidRecipeOp,1,nullptr};
    e::ExtensionRegistry invalidRecipeRegistry;require(invalidRecipeRegistry.registerExtension(invalidRecipeExtension).code=="TH021-DESCRIPTOR","invalid lowering recipe accepted");
    const auto digest=registry.digest();
    e::ExtensionRegistry deterministic;require(deterministic.load(TH021_EXTENSION).ok()&&deterministic.digest()==digest,"registry digest is nondeterministic");
    const e::OperationDescriptorV0 orderOpA{"order_a","1.0.0",1,1,2,0,true,e::EffectV0::Pure,e::Reference,false,{identityNodes,1,0,1},nullptr};
    const e::OperationDescriptorV0 orderOpB{"order_b","1.0.0",1,1,2,0,true,e::EffectV0::Pure,e::Reference,false,{identityNodes,1,0,1},nullptr};
    const e::ExtensionDescriptorV0 orderA{e::extensionAbiVersion,"org.thiran.order.a","1.0.0",&orderOpA,1,nullptr};
    const e::ExtensionDescriptorV0 orderB{e::extensionAbiVersion,"org.thiran.order.b","1.0.0",&orderOpB,1,nullptr};
    e::ExtensionRegistry forwardOrder,reverseOrder;
    require(forwardOrder.registerExtension(orderA).ok()&&forwardOrder.registerExtension(orderB).ok()&&
            reverseOrder.registerExtension(orderB).ok()&&reverseOrder.registerExtension(orderA).ok()&&
            forwardOrder.freeze().ok()&&reverseOrder.freeze().ok()&&forwardOrder.digest()==reverseOrder.digest(),
            "registration order changed frozen registry identity");

    auto parsedUnknown=parse("fn f(x:Tensor<f32,1>)->Tensor<f32,1>{return research_square_linear(x)}","<unknown>");
    auto unknown=s::analyze(*parsedUnknown.module);require(!unknown.module&&unknown.diagnostics.front().category=="TH005-UNKNOWN-FUNCTION","unloaded operation did not fail clearly");
    auto module=compile("fn f(x:Tensor<f32,1>)->Tensor<f32,1>{return research_square_linear(x)}",registry);
    require(registry.frozen()&&!registry.load(TH021_EXTENSION).ok()&&registry.registerExtension(orderA).code=="TH021-REGISTRY-FROZEN","registry did not freeze before compilation");
    auto input=tensor({7},{-2.0f,-0.5f,0.0f,1.5f,3.0f,7.25f,-9.0f});
    auto reference=s::evaluateCall(module,"f",{input});require(reference.ok,"reference extension execution failed");
    auto referenceValues=std::get<s::RuntimeTensor>(reference.value->data).f32Values;
    require(referenceValues==std::vector<float>({2.0f,-0.25f,0.0f,3.75f,12.0f,59.8125f,72.0f}),"reference mathematics mismatch");
    auto zero=s::evaluateCall(module,"f",{tensor({0},{})});require(zero.ok&&std::get<s::RuntimeTensor>(zero.value->data).f32Values.empty(),"zero-sized reference failed");
    auto rejectSource=[&](const std::string& source,const std::string& code){auto p=parse(source,"<th021-reject>");auto result=s::analyze(*p.module,registry);return !result.module&&!result.diagnostics.empty()&&result.diagnostics.front().category==code;};
    require(rejectSource("fn f(x:Tensor<f32,1>)->Tensor<f32,1>{return research_square_linear(x,x)}","TH021-ARITY"),"wrong extension arity accepted");
    require(rejectSource("fn f(x:Tensor<i64,1>)->Tensor<i64,1>{return research_square_linear(x)}","TH021-DTYPE"),"wrong extension dtype accepted");
    require(rejectSource("fn f(x:Tensor<f32,3>)->Tensor<f32,3>{return research_square_linear(x)}","TH021-RANK"),"wrong extension rank accepted");

    auto facts=analysis::analyze(module);require(facts.ok()&&facts.functionEffects.at(1).kinds==0,"extension effects are not existing TH-006 Pure");
    bool hasFresh=false;for(const auto& [id,value]:facts.valueProvenance.at(1))if(value.kind==analysis::ProvenanceKind::Fresh&&id>module.functions[0].parameters[0].id)hasFresh=true;
    require(hasFresh,"extension result is not fresh ownership provenance");
    auto differentiated=autodiff::differentiate(module,facts,{1,{0}});require(differentiated.ok(),differentiated.diagnostics.empty()?"AD failed":differentiated.diagnostics.front().message);
    require(std::any_of(differentiated.saves.begin(),differentiated.saves.end(),[](const auto& save){return save.reason.find("research derivative saved primal input")!=std::string::npos;}),"AD primal save missing");
    auto gradient=autodiff::executeVjp(differentiated,{input},tensor({7},{1,1,1,1,1,1,1}));require(gradient.ok,"extension VJP execution failed");
    auto gradientValues=std::get<s::RuntimeTensor>(gradient.value->data).f32Values;
    const std::vector<float> expectedGradient{-3,0,1,4,7,15.5f,-17};
    for(std::size_t k=0;k<expectedGradient.size();++k)require(close(gradientValues[k],expectedGradient[k]),"hand-derived gradient mismatch");
    for(float x:std::vector<float>{-2.0f,-0.5f,0.0f,1.5f,3.25f}){
        const float h=1e-3f;const float finite=(((x+h)*(x+h)+(x+h))-((x-h)*(x-h)+(x-h)))/(2*h);
        require(close(finite,2*x+1,4e-3f),"finite-difference gradient mismatch");
    }

    auto nativeRegion=region(module,"f");
    require(nativeRegion.nodes.back().op==b::RegionOp::Extension&&nativeRegion.nodes.back().extensionOperation,"generic TensorRegion extension node missing");
    auto malformedRankModule=module;
    for(auto& step:malformedRankModule.functions[0].body.steps)if(auto* instruction=std::get_if<s::Instruction>(&step);instruction&&instruction->extensionOperation){
        instruction->extensionOperation->minimumRank=2;instruction->extensionOperation->maximumRank=2;
        instruction->extensionOperation->descriptorDigest=e::operationDigest(*instruction->extensionOperation);
    }
    require(!s::verify(malformedRankModule).ok,"semantic verifier accepted extension rank outside its descriptor");
    auto malformedRankRegion=nativeRegion;
    malformedRankRegion.nodes.back().extensionOperation->minimumRank=2;
    malformedRankRegion.nodes.back().extensionOperation->maximumRank=2;
    malformedRankRegion.nodes.back().extensionOperation->descriptorDigest=e::operationDigest(*malformedRankRegion.nodes.back().extensionOperation);
    require(!b::verifyRegion(malformedRankRegion).ok(),"TensorRegion verifier accepted extension rank outside its descriptor");
    auto plan=b::buildPhysicalPlan(nativeRegion,b::PhysicalDevice::Host);require(plan.ok(),"extension physical plan failed");
    b::PhysicalPlanningObligations saveObligations;for(const auto& save:differentiated.saves)saveObligations.savedForBackward.insert(save.primal);
    auto savedPlan=b::buildPhysicalPlan(nativeRegion,b::PhysicalDevice::Host,{},saveObligations);require(savedPlan.ok(),"AD-save physical plan failed");
    bool savedProtected=false;for(const auto& value:savedPlan.plan->values)savedProtected|=value.classification==b::PhysicalValueClass::SavedForBackward;
    require(savedProtected,"AD-saved extension input was not protected by ordinary planner obligation");
    auto cpuEntry=a::specializeEntry(nativeRegion,{storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9})});
    auto cpuBuild=a::buildCpuAot(nativeRegion,toolchain(),{root/"extension.tha",{},cpuEntry});require(cpuBuild.success,cpuBuild.error?cpuBuild.error->message:"CPU AOT failed");
    auto cpuRun=a::loadAndExecuteArtifact(root/"extension.tha",{storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9})});
    require(cpuRun.ok()&&std::get<storage::Tensor>(*cpuRun.value).logicalF32Values()==referenceValues,"native CPU/reference mismatch");
    auto jit=std::make_unique<a::JitCompiler>(toolchain());auto j1=jit->compileCpu(nativeRegion,{storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9})});
    auto j2=jit->compileCpu(nativeRegion,{storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9})});require(j1.ok()&&j2.ok()&&!j1.cacheHit&&j2.cacheHit,"CPU extension JIT/cache failed");
    auto changed=nativeRegion;auto& changedOp=*changed.nodes.back().extensionOperation;changedOp.semanticVersion="1.0.1";changedOp.canonicalIdentity=changedOp.extensionId+"::"+changedOp.name+"@1.0.1/abi1";changedOp.descriptorDigest=e::operationDigest(changedOp);
    auto j3=jit->compileCpu(changed,{storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9})});require(j3.ok()&&!j3.cacheHit&&j3.executable->cacheKey()!=j1.executable->cacheKey(),"changed extension identity hit stale JIT cache");
    auto changedRecipe=nativeRegion;auto& recipeOp=*changedRecipe.nodes.back().extensionOperation;
    recipeOp.forward.nodes.back().opcode=e::ScalarOpcode::Subtract;recipeOp.descriptorDigest=e::operationDigest(recipeOp);
    auto j4=jit->compileCpu(changedRecipe,{storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9})});
    require(j4.ok()&&!j4.cacheHit&&j4.executable->cacheKey()!=j1.executable->cacheKey(),"changed CPU lowering recipe hit stale JIT cache");
    auto changedRecipeRun=j4.executable->execute({storage::Tensor::materializeF32({7},{-2,-.5f,0,1.5f,3,7.25f,-9})});
    require(changedRecipeRun.ok()&&std::get<storage::Tensor>(*changedRecipeRun.value).logicalF32Values()==
        std::vector<float>({6,.75f,0,.75f,6,45.3125f,90}),"changed CPU lowering recipe did not execute its new semantics");
    auto shortInput=storage::Tensor::materializeF32({3},{-1,0,2.5f});
    auto j5=jit->compileCpu(nativeRegion,{shortInput});auto j6=jit->compileCpu(nativeRegion,{shortInput});
    require(j5.ok()&&j6.ok()&&!j5.cacheHit&&j6.cacheHit&&j5.executable->cacheKey()!=j1.executable->cacheKey(),"CPU shape specialization cache identity failed");
    auto zeroInput=storage::Tensor::empty(storage::DType::F32,{0});auto zeroJit=jit->compileCpu(nativeRegion,{zeroInput});
    auto zeroNative=zeroJit.ok()?zeroJit.executable->execute({zeroInput}):a::ArtifactExecutionResult{};
    require(zeroNative.ok()&&std::get<storage::Tensor>(*zeroNative.value).logicalF32Values().empty(),"zero-size native CPU extension failed");
    auto wrongDtypeJit=jit->compileCpu(nativeRegion,{storage::Tensor::materializeI64({3},{1,2,3})});
    auto wrongRankJit=jit->compileCpu(nativeRegion,{storage::Tensor::materializeF32({1,3},{1,2,3})});
    require(!wrongDtypeJit.ok()&&!wrongRankJit.ok(),"incompatible CPU JIT specialization was accepted");

    const auto freshDir=root/"fresh-runtime-only";fs::create_directory(freshDir);
    fs::copy_file(root/"extension.tha",freshDir/"program.tha");fs::copy_file(TH021_RUNTIME,freshDir/"thiran-extension-runtime");
    fs::permissions(freshDir/"thiran-extension-runtime",fs::perms::owner_read|fs::perms::owner_write|fs::perms::owner_exec,fs::perm_options::add);
    require(!fs::exists(freshDir/fs::path(TH021_EXTENSION).filename()),"extension library copied into runtime-only deployment directory");
    require(std::distance(fs::directory_iterator(freshDir),fs::directory_iterator{})==2,"runtime-only deployment contains compiler/build files");
    const auto prior=fs::current_path();const char* oldPath=std::getenv("PATH");const std::string savedPath=oldPath?oldPath:"";
    fs::current_path(freshDir);::setenv("PATH","/nonexistent",1);::setenv("CXX","/nonexistent",1);
    auto runtimeOnly=tooling::runProcess({(freshDir/"thiran-extension-runtime").string(),{"program.tha"}});
    ::setenv("PATH",savedPath.c_str(),1);::unsetenv("CXX");fs::current_path(prior);
    require(runtimeOnly.launched&&runtimeOnly.exitStatus==0&&runtimeOnly.standardOutput==
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"f32\",\"shape\":[7],\"values\":[2,-0.25,0,3.75,12,59.8125,72]}\n"&&
        runtimeOnly.standardError.find("compiler_invocations=0 fallback=NONE")!=std::string::npos,
        "compiler/plugin-absent CPU deployment failed: "+runtimeOnly.standardOutput+runtimeOnly.standardError);

    auto pairRegistry=std::make_unique<e::ExtensionRegistry>();require(pairRegistry->load(TH021_EXTENSION).ok(),"second registry load failed");
    auto pairModule=compile("fn g(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{return research_pair_mix(x,y)}",*pairRegistry);
    auto pair=s::evaluateCall(pairModule,"g",{tensor({3},{1,2,3}),tensor({3},{4,5,6})});require(pair.ok&&std::get<s::RuntimeTensor>(pair.value->data).f32Values==std::vector<float>({5,12,21}),"second operation generic dispatch failed");
    require(rejectSource("fn g(a:f32,b:f32,c:f32)->Tensor<f32,1>{let x=[a,b];let y=[c];return research_pair_mix(x,y)}","TH021-SHAPE"),"known extension shape mismatch passed semantic analysis");
    auto wrongShape=s::evaluateCall(pairModule,"g",{tensor({2},{1,2}),tensor({3},{1,2,3})});require(!wrongShape.ok&&wrongShape.errorId=="TH021-SHAPE","runtime extension shape mismatch accepted");
    auto ratioModule=compile("fn ratio(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{return research_checked_ratio(x,y)}",*pairRegistry);
    auto ratioReference=s::evaluateCall(ratioModule,"ratio",{tensor({2},{4,3}),tensor({2},{2,0})});
    require(!ratioReference.ok&&ratioReference.errorId=="TH021-DIVIDE-BY-ZERO","reference MayTrap operation did not fail explicitly");
    auto ratioRegion=region(ratioModule,"ratio");
    auto ratioEntry=a::specializeEntry(ratioRegion,{storage::Tensor::materializeF32({2},{4,3}),storage::Tensor::materializeF32({2},{2,0})});
    auto ratioBuild=a::buildCpuAot(ratioRegion,toolchain(),{root/"ratio.tha",{},ratioEntry});require(ratioBuild.success,"MayTrap CPU AOT build failed");
    auto ratioRun=a::loadAndExecuteArtifact(root/"ratio.tha",{storage::Tensor::materializeF32({2},{4,3}),storage::Tensor::materializeF32({2},{2,0})});
    require(!ratioRun.ok()&&ratioRun.error&&ratioRun.error->code=="TH021-DIVIDE-BY-ZERO","native CPU MayTrap operation published output");
    auto gpuUnsupported=region(compile("fn c(x:Tensor<f32,1>)->Tensor<f32,1>{return research_cpu_only(x)}",*pairRegistry),"c",false);
    (void)gpuUnsupported;
    auto cpuOnlyFacts=analysis::analyze(compile("fn c(x:Tensor<f32,1>)->Tensor<f32,1>{return research_cpu_only(x)}",*pairRegistry));
    // The registry is already frozen and can still be reused for compilation.
    auto cpuOnlyModule=compile("fn c(x:Tensor<f32,1>)->Tensor<f32,1>{return research_cpu_only(x)}",*pairRegistry);
    auto rejectedGpu=b::extractStrictGpu(cpuOnlyModule,cpuOnlyFacts,"c",false);require(!rejectedGpu.ok()&&rejectedGpu.coverage.find("fallback: NONE")!=std::string::npos,"CPU-only extension fell back on GPU");

    auto fusionRegistry=std::make_unique<e::ExtensionRegistry>();require(fusionRegistry->load(TH021_EXTENSION).ok(),"fusion registry load failed");
    auto fusionModule=compile("fn h(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{let a=x+y;let b=research_square_linear(a);return b.*y}",*fusionRegistry);
    auto fusionRegion=region(fusionModule,"h");auto fusionPlan=b::buildPhysicalPlan(fusionRegion,b::PhysicalDevice::Host);
    require(fusionPlan.ok()&&fusionPlan.plan->fusionGroups.size()==1&&fusionPlan.plan->fusionGroups[0].nodes.size()==3,"built-in/extension/built-in fusion failed");
    auto multiModule=compile("fn h(x:Tensor<f32,1>,y:Tensor<f32,1>)->Tensor<f32,1>{let a=research_square_linear(x);let b=a+y;return a.*b}",*fusionRegistry);
    auto multiRegion=region(multiModule,"h");auto multiPlan=b::buildPhysicalPlan(multiRegion,b::PhysicalDevice::Host);
    require(multiPlan.ok()&&multiPlan.plan->fusionGroups.size()>1,"multiple-consumer extension value fused illegally");

    auto conditional=compile("fn choose(x:Tensor<f32,1>,flag:bool)->Tensor<f32,1>{if flag{return research_square_linear(x)}else{return x}}",*fusionRegistry);
    require(s::evaluateCall(conditional,"choose",{tensor({2},{2,-2}),s::RuntimeValue{true}}).format().find("[6,2]")!=std::string::npos,"extension in conditional failed");
    auto loop=compile("fn looped(x:Tensor<f32,1>,n:i64)->Tensor<f32,1>{for i in 0:n { research_square_linear(x) };return research_square_linear(x)}",*fusionRegistry);
    require(s::evaluateCall(loop,"looped",{tensor({1},{2}),s::RuntimeValue{std::int64_t{3}}}).ok,"extension in structured loop failed");
    auto scan=compile("fn step(x:Tensor<f32,1>,state:f32)->(f32,f32){let y=research_square_linear(x);return(y[0],state)}\nfn run(xs:Tensor<f32,2>,state:f32)->(Tensor<f32,1>,f32){return scan(step,xs,state)}",*fusionRegistry);
    auto scanRun=s::evaluateCall(scan,"run",{tensor({2,1},{2,-2}),s::RuntimeValue{0.0f}});
    require(scanRun.ok&&std::get<s::RuntimeTensor>(std::get<s::RuntimeTuple>(scanRun.value->data)[0].data).f32Values==std::vector<float>({6,2}),"extension in forward Scan step failed");

    auto noDerivative=autodiff::differentiate(pairModule,analysis::analyze(pairModule),{1,{0}});
    require(!noDerivative.ok()&&!noDerivative.diagnostics.empty()&&noDerivative.diagnostics.front().code=="AD-ELIGIBILITY-EXTENSION","missing derivative did not explicitly reject AD");
    std::cout<<"V0ExtensionTests PASS "<<checks<<" checks\nregistry_digest="<<digest<<"\n"
             <<"gradient=";for(float value:gradientValues)std::cout<<value<<',';std::cout<<"\n"
             <<"cpu_aot_plugin_runtime_dependency=NONE fallback=NONE\n";
    fs::remove_all(root);return 0;
 }catch(const std::exception& failure){std::cerr<<"V0ExtensionTests FAIL after "<<checks<<" checks: "<<failure.what()<<'\n';fs::remove_all(root);return 1;}
}
