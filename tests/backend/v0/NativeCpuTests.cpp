#include "backend/v0/NativeCpu.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace fs=std::filesystem;
using namespace thiran::v0;
namespace s=thiran::v0::semantic;
namespace b=thiran::v0::backend;
static int checks=0;
static void require(bool yes,const std::string& what) { ++checks; if(!yes) throw std::runtime_error(what); }
static s::Module source(const std::string& text) {
    auto p=parse(text,"<th008-test>"); require(p.module.has_value(),"parse failed");
    auto a=s::analyze(*p.module); require(a.module.has_value(),a.diagnostics.empty()?"analyze failed":a.diagnostics.front().format());
    require(s::verify(*a.module).ok,"semantic verifier failed"); return *a.module;
}
static b::NativeResult extract(const s::Module& m,const std::string& name="main",bool standalone=true) {
    auto facts=analysis::analyze(m); require(facts.ok(),"ownership qualification failed");
    return b::extractStrictNative(m,facts,name,standalone);
}
static fs::path temporary() {
    std::string pattern="/tmp/th008-native-XXXXXX";
    auto ptr=pattern.data();
    if(!mkdtemp(ptr)) throw std::runtime_error("mkdtemp failed");
    return pattern;
}
static std::string read(const fs::path& p) { std::ifstream f(p); return {std::istreambuf_iterator<char>(f),{}}; }
struct Run { std::string output,error,command; int exit=0; fs::path artifact; };
static Run compileRun(const std::string& cpp,const fs::path& dir,const std::string& stem) {
    fs::path src=dir/(stem+".cpp"),exe=dir/stem,build=TH008_BUILD;
    { std::ofstream f(src); f<<cpp; }
    std::string command=std::string(TH008_CXX)+" -std=c++20 "+TH008_CXX_FLAGS+" -I"+TH008_INCLUDE+" "+src.string()+
        " "+(build/"libthiran_v0_storage.a").string()+" "+(build/"libthiran_v0_async.a").string()+
        " "+(build/"libthiran_v0_analysis.a").string()+
        " "+(build/"libthiran_v0_semantic.a").string()+" "+(build/"libthiran_v0_frontend.a").string()+
        " -o "+exe.string()+" 2>"+(dir/(stem+".compile.err")).string();
    int status=std::system(command.c_str());
    require(status==0,"host compile failed: "+read(dir/(stem+".compile.err")));
    std::string run=exe.string()+" >"+(dir/(stem+".out")).string()+" 2>"+(dir/(stem+".err")).string();
    status=std::system(run.c_str());
    return {read(dir/(stem+".out")),read(dir/(stem+".err")),command,WIFEXITED(status)?WEXITSTATUS(status):255,exe};
}
static std::string expected(const s::Module& m) { return s::evaluateCall(m,"main",{}).format()+"\n"; }
static bool equivalent(const std::string& oracle,const std::string& native) { return oracle==native; }
int main() {
 try {
    auto dir=temporary();
    const std::string literal="fn main() -> Tensor<i64,2> {\nlet A = [1, 2; 3, 4]\nreturn A\n}";
    const std::string add="fn main() -> Tensor<i64,2> {\nlet A = [1, 2; 3, 4]\nlet B = [5, 6; 7, 8]\nreturn A + B\n}";
    const std::string alias="fn main() -> Tensor<i64,2> {\nlet A = [1, 2; 3, 4]\nlet B = A\nreturn A + B\n}";
    const std::string index="fn main() -> i64 {\nlet A = [1, 2; 3, 4]\nreturn A[1, 0]\n}";
    auto ml=source(literal), ma=source(add), malias=source(alias), mi=source(index);
    auto rl=extract(ml),ra=extract(ma),ralias=extract(malias),ri=extract(mi);
    require(rl.ok()&&ra.ok()&&ralias.ok()&&ri.ok(),"supported extraction failed: "+ra.diagnostic);
    require(ra.region->dump()==extract(ma).region->dump(),"N12 region dump not deterministic");
    auto cpp=b::emitCpp20(*ra.region,true);
    require(cpp==b::emitCpp20(*extract(ma).region,true),"N11 C++ not deterministic");
    require(cpp.find("checked_add")!=std::string::npos && cpp.find("Tensor::materializeI64")!=std::string::npos,
            "N02 checked/storage code absent");
    auto nl=compileRun(b::emitCpp20(*rl.region,true),dir,"literal");
    auto na=compileRun(cpp,dir,"add");
    auto nalias=compileRun(b::emitCpp20(*ralias.region,true),dir,"alias");
    auto ni=compileRun(b::emitCpp20(*ri.region,true),dir,"index");
    require(nl.exit==0&&equivalent(expected(ml),nl.output),"N01 literal/reference mismatch");
    require(na.exit==0&&equivalent(expected(ma),na.output),"N02/N03 add/reference mismatch");
    require(na.output=="{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[6,8,10,12]}\n", "N09 canonical tensor mismatch");
    require(nalias.exit==0&&equivalent(expected(malias),nalias.output),"N04 alias mismatch");
    auto aliasCpp=b::emitCpp20(*ralias.region,true);
    require(aliasCpp.find("deepCopy")==std::string::npos && aliasCpp.find("Tensor v")!=std::string::npos,
            "N04 alias induced deep copy");
    require(ni.exit==0&&equivalent(expected(mi),ni.output),"N05/N10 index/reference mismatch");
    require(ni.output=="{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":3}\n","N10 scalar output mismatch");
    require(fs::exists(na.artifact)&&fs::file_size(na.artifact)>0,"N16 artifact absent");
    auto again=compileRun(cpp,dir,"fresh_process");
    require(again.exit==0&&again.output==na.output,"N16 fresh process mismatch");
    require(!equivalent(expected(ma),"{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":0}\n"),"N18 mismatch harness blind");
    require(ra.coverage.find("Add native-cpu")!=std::string::npos && ra.coverage.find("fallback: NONE")!=std::string::npos,"N15 coverage absent");
    const std::string pick="fn pick(x: Tensor<i64,2>, i: i64) -> i64 {\nreturn x[i, 0]\n}";
    auto mp=source(pick); auto rp=extract(mp,"pick",false); require(rp.ok(),"dynamic pick extraction: "+rp.diagnostic);
    std::string wrapper="int main() { Tensor x=Tensor::materializeI64({2,2},{1,2,3,4});\n"
        "  try { std::cout << native_f1(x,1) << '\\n'; } catch (...) { return 2; }\n"
        "  try { (void)native_f1(x,2); return 3; }\n"
        "  catch (const SemanticFailure& e) { std::cout << e.id << '\\n'; return 0; }\n}\n";
    auto np=compileRun(b::emitCpp20(*rp.region,false,wrapper),dir,"pick_dynamic");
    require(np.exit==0&&np.output=="3\nTH-SPEC-BOUNDS\n","N06/N07 runtime bounds mismatch: "+np.output+np.error);
    s::RuntimeValue x; x.data=s::RuntimeTensor{s::TypeKind::I64,{2,2},{1,2,3,4},{}};
    s::RuntimeValue good; good.data=std::int64_t{1};
    s::RuntimeValue bad; bad.data=std::int64_t{2};
    require(s::evaluateCall(mp,"pick",{x,good}).format()=="{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":3}",
        "N06 reference valid dynamic bounds mismatch");
    require(s::evaluateCall(mp,"pick",{x,bad}).format()=="{\"status\":\"error\",\"error_id\":\"TH-SPEC-BOUNDS\"}",
        "N07 reference invalid dynamic bounds mismatch");
    const std::string dynAdd="fn add(x: Tensor<i64,2>, y: Tensor<i64,2>) -> Tensor<i64,2> {\nreturn x + y\n}";
    auto mo=source(dynAdd); auto ro=extract(mo,"add",false); require(ro.ok(),"dynamic Add extraction: "+ro.diagnostic);
    std::string overflowWrapper="int main() { Tensor x=Tensor::materializeI64({1,1},{9223372036854775807LL});\n"
        " Tensor y=Tensor::materializeI64({1,1},{1});\n"
        " try { (void)native_f1(x,y); return 3; } catch (const SemanticFailure& e) { std::cout<<e.id<<'\\n'; return 0; }\n}\n";
    auto no=compileRun(b::emitCpp20(*ro.region,false,overflowWrapper),dir,"overflow_dynamic");
    require(no.exit==0&&no.output=="TH-SPEC-I64-OVERFLOW\n","N08 runtime overflow mismatch: "+no.output+no.error);
    s::RuntimeValue maxTensor; maxTensor.data=s::RuntimeTensor{s::TypeKind::I64,{1,1},{9223372036854775807LL},{}};
    s::RuntimeValue oneTensor; oneTensor.data=s::RuntimeTensor{s::TypeKind::I64,{1,1},{1},{}};
    require(s::evaluateCall(mo,"add",{maxTensor,oneTensor}).format()=="{\"status\":\"error\",\"error_id\":\"TH-SPEC-I64-OVERFLOW\"}",
        "N08 reference dynamic overflow mismatch");
    auto broadcast=source("fn main() -> Tensor<i64,2> {\nlet A=[1,2;3,4]\nlet B=[5,6]\nreturn A + B\n}");
    auto rb=extract(broadcast);
    require(!rb.ok()&&rb.diagnostic.find("BACKEND-UNSUPPORTED")!=std::string::npos&&
            rb.diagnostic.find("TH-SPEC-BROADCAST")==std::string::npos,
            "valid semantic broadcast was mislabeled or native-admitted");
    auto matmul=source("fn main() -> Tensor<i64,2> {\nlet A=[1,2;3,4]\nreturn A * A\n}");
    auto rm=extract(matmul); require(!rm.ok()&&rm.diagnostic.find("BACKEND-UNSUPPORTED")!=std::string::npos&&
        rm.coverage.find("unsupported-native")!=std::string::npos,"N13 matmul not strict-native unsupported");
    auto f32=source("fn f(x: f32) -> f32 { return x*x }");
    auto rf32=extract(f32,"f",false); require(!rf32.ok()&&rf32.diagnostic.find("BACKEND-UNSUPPORTED")!=std::string::npos&&
        rf32.coverage.find("fallback: NONE")!=std::string::npos,"TH010 f32 was native-admitted or gained fallback");
    auto cf=source("fn main() -> i64 {\nif true { return 1 } else { return 2 }\n}");
    auto rcf=extract(cf); require(!rcf.ok()&&rcf.diagnostic.find("BACKEND-UNSUPPORTED")!=std::string::npos,
        "N14 structured control not strict-native unsupported");
    const std::string fusedSource=
        "fn f(x:Tensor<i64,1>,y:Tensor<i64,1>,c:Tensor<i64,1>,d:Tensor<i64,1>)->Tensor<i64,1>{\n"
        "let t1=x+y\nlet t2=t1.*c\nlet t3=t2-d\nreturn t3+y\n}";
    auto mfused=source(fusedSource); auto rfused=extract(mfused,"f",false);
    require(rfused.ok(),"TH015 CPU fused extraction failed: "+rfused.diagnostic);
    auto fusedPlan=b::buildPhysicalPlan(*rfused.region,b::PhysicalDevice::Host);
    auto conservativePlan=b::buildPhysicalPlan(*rfused.region,b::PhysicalDevice::Host,{true,false});
    require(fusedPlan.ok()&&conservativePlan.ok(),"TH015 CPU planning failed");
    require(fusedPlan.plan->fusionGroups.size()==1&&fusedPlan.plan->fusionGroups[0].nodes.size()==4,
        "TH015 CPU chain did not form one fusion group");
    std::size_t logicalTemps=0,physicalTemps=0;
    for(const auto& value:conservativePlan.plan->values)
        logicalTemps+=value.classification==b::PhysicalValueClass::Temporary&&value.value==value.root;
    for(const auto& slot:conservativePlan.plan->slots) physicalTemps+=slot.reusable&&!slot.external;
    require(logicalTemps==3&&physicalTemps==2,
        "TH015 CPU conservative plan did not reuse one non-overlapping temp slot");
    const std::string fusedWrapper=
        "int main(){Tensor x=Tensor::materializeI64({2},{1,2});Tensor y=Tensor::materializeI64({2},{3,4});"
        "Tensor c=Tensor::materializeI64({2},{2,2});Tensor d=Tensor::materializeI64({2},{1,1});"
        "try{auto r=native_f1(x,y,c,d);auto z=r.logicalI64Values();if(z!=std::vector<std::int64_t>{10,15})return 4;"
        "std::cout<<\"ok\\n\";return 0;}catch(...){return 5;}}\n";
    const auto fusedCpp=b::emitCpp20(*rfused.region,false,fusedWrapper);
    const auto unfusedCpp=b::emitCpp20(*rfused.region,false,fusedWrapper,{true,false});
    require(fusedCpp.find("fusion_groups=1")!=std::string::npos&&
        std::count(fusedCpp.begin(),fusedCpp.end(),'\n')<std::count(unfusedCpp.begin(),unfusedCpp.end(),'\n'),
        "TH015 CPU emission did not consume the fusion plan");
    auto fusedRun=compileRun(fusedCpp,dir,"th015_fused");
    auto unfusedRun=compileRun(unfusedCpp,dir,"th015_unfused");
    require(fusedRun.exit==0&&unfusedRun.exit==0&&fusedRun.output=="ok\n"&&fusedRun.output==unfusedRun.output,
        "TH015 CPU enabled/disabled execution mismatch: "+fusedRun.error+unfusedRun.error);

    const std::string overflowFusedWrapper=
        "int main(){Tensor x=Tensor::materializeI64({1},{std::numeric_limits<std::int64_t>::max()});"
        "Tensor y=Tensor::materializeI64({1},{1});Tensor c=Tensor::materializeI64({1},{0});"
        "Tensor d=Tensor::materializeI64({1},{0});try{(void)native_f1(x,y,c,d);return 3;}"
        "catch(const SemanticFailure&e){std::cout<<e.id<<'\\n';return 0;}}\n";
    auto fusedOverflow=compileRun(b::emitCpp20(*rfused.region,false,overflowFusedWrapper),dir,"th015_fused_overflow");
    require(fusedOverflow.exit==0&&fusedOverflow.output=="TH-SPEC-I64-OVERFLOW\n",
        "fused intermediate i64 overflow did not preserve checked failure");

    auto mf32=source(
        "fn f(x:Tensor<f32,1>,y:Tensor<f32,1>,c:Tensor<f32,1>,d:Tensor<f32,1>)->Tensor<f32,1>{\n"
        "let t1=x+y\nlet t2=t1.*c\nlet t3=t2-d\nreturn t3+y\n}");
    auto rf32Native=extract(mf32,"f",false);
    require(rf32Native.ok(),"TH015 f32 CPU extraction failed: "+rf32Native.diagnostic);
    const std::string f32Wrapper=
        "int main(){Tensor x=Tensor::materializeF32({2},{1,2});Tensor y=Tensor::materializeF32({2},{3,4});"
        "Tensor c=Tensor::materializeF32({2},{2,2});Tensor d=Tensor::materializeF32({2},{1,1});"
        "auto r=native_f1(x,y,c,d);if(r.logicalF32Values()!=std::vector<float>{10,15})return 4;"
        "std::cout<<\"ok\\n\";return 0;}\n";
    const auto f32FusedCpp=b::emitCpp20(*rf32Native.region,false,f32Wrapper);
    const auto f32UnfusedCpp=b::emitCpp20(*rf32Native.region,false,f32Wrapper,{false,false});
    require(f32FusedCpp.find("fast-math")==std::string::npos&&f32FusedCpp.find("fma(")==std::string::npos&&
        f32FusedCpp.find("volatile float element_v")!=std::string::npos,
        "f32 fusion introduced fast-math/contraction text");
    auto f32Fused=compileRun(f32FusedCpp,dir,"th015_f32_fused");
    auto f32Unfused=compileRun(f32UnfusedCpp,dir,"th015_f32_unfused");
    require(f32Fused.exit==0&&f32Unfused.exit==0&&f32Fused.output=="ok\n"&&f32Fused.output==f32Unfused.output,
        "f32 fused/unfused native CPU result mismatch");
    s::RuntimeValue fx; fx.data=s::RuntimeTensor{s::TypeKind::F32,{2},{},{1,2}};
    s::RuntimeValue fy; fy.data=s::RuntimeTensor{s::TypeKind::F32,{2},{},{3,4}};
    s::RuntimeValue fc; fc.data=s::RuntimeTensor{s::TypeKind::F32,{2},{},{2,2}};
    s::RuntimeValue fd; fd.data=s::RuntimeTensor{s::TypeKind::F32,{2},{},{1,1}};
    auto f32Reference=s::evaluateCall(mf32,"f",{fx,fy,fc,fd});
    require(f32Reference.ok&&f32Reference.value&&
        std::get<s::RuntimeTensor>(f32Reference.value->data).f32Values==std::vector<float>({10,15}),
        "f32 semantic reference disagreed with native fused result");
    // Independent manually malformed TensorRegions, not extractor-produced failures.
    b::TensorRegion base; base.function=1; base.name="manual"; base.outputType=s::tensor(s::scalar(s::TypeKind::I64),2);
    s::ShapeFact sh{{2,2}};
    auto manualNode=[&](s::ValueId id,b::RegionOp op,std::vector<s::ValueId> dependencies={}) {
        b::RegionNode node; node.id=id; node.op=op; node.type=base.outputType; node.shape=sh;
        node.dependencies=std::move(dependencies); return node;
    };
    base.nodes={manualNode(1,b::RegionOp::Input),manualNode(2,b::RegionOp::Input),
                manualNode(3,b::RegionOp::Add,{1,2})}; base.inputs={1,2}; base.output=3;
    require(b::verifyRegion(base).ok(),"manual base invalid");
    auto negative=[&](const b::TensorRegion& r,const std::string& id) {
        auto v=b::verifyRegion(r); bool found=false; for(auto& e:v.errors) found|=e.find(id)!=std::string::npos;
        require(found,"negative region verifier missed "+id);
    };
    {auto r=base; r.nodes[1].id=1; negative(r,"TRV01");}
    {auto r=base; r.nodes[2].dependencies={1,99}; negative(r,"TRV02");}
    {auto r=base; r.nodes[2].dependencies={1,3}; negative(r,"TRV03");}
    {auto r=base; r.nodes[2].type=s::scalar(s::TypeKind::I64); negative(r,"TRV04");}
    {auto r=base; r.nodes[1].type=s::tensor(s::scalar(s::TypeKind::I64),1); negative(r,"TRV05");}
    {auto r=base; r.nodes[1].shape={{2,3}}; negative(r,"TRV06");}
    {auto r=base; r.nodes[2].op=b::RegionOp::Index; r.nodes[2].dependencies={1}; r.nodes[2].indices={}; negative(r,"TRV07");}
    {auto r=base; r.nodes[2].op=b::RegionOp::Index; r.nodes[2].dependencies={1}; r.nodes[2].indices={1,2}; negative(r,"TRV08");}
    {auto r=base; r.output=99; negative(r,"TRV09");}
    {auto r=base; r.nodes[2].op=b::RegionOp::Unsupported; negative(r,"TRV10");}
    std::cout<<"V0NativeCpuTests PASS "<<checks<<" checks\n"
             <<"artifact="<<na.artifact<<"\ncompile="<<na.command<<"\n"
             <<"expected="<<expected(ma)<<"observed="<<na.output
             <<"bounds="<<np.output<<"overflow="<<no.output;
    return 0;
 } catch(const std::exception& e) { std::cerr<<"V0NativeCpuTests FAIL: "<<e.what()<<'\n'; return 1; }
}
