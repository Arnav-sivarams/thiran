#include "autodiff/v0/Autodiff.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Verifier.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace s=thiran::v0::semantic;
namespace a=thiran::v0::analysis;
namespace ad=thiran::v0::autodiff;
using thiran::v0::parse;
namespace {
int checks=0;
void expect(bool yes,const std::string& message) { ++checks; if (!yes) throw std::runtime_error(message); }
struct Program { s::Module source; a::OwnershipAnalysisResult ownership; ad::DifferentiationResult differentiated; };
Program compile(const std::string& text,std::vector<std::size_t> wrt={0},std::size_t function=0) {
    auto parsed=parse(text,"autodiff.th");
    expect(parsed.module.has_value(),"parse failed for `"+text+"`: "+(parsed.diagnostics.empty()?std::string{}:parsed.diagnostics[0].format()));
    auto analyzed=s::analyze(*parsed.module);
    expect(analyzed.module.has_value(),"semantic failed: "+(analyzed.diagnostics.empty()?std::string{}:analyzed.diagnostics[0].format()));
    expect(s::verify(*analyzed.module).ok,"source semantic verifier failed");
    auto ownership=a::analyze(*analyzed.module);
    expect(ownership.ok(),"ownership analysis failed");
    auto id=analyzed.module->functions.at(function).id;
    auto differentiated=ad::differentiate(*analyzed.module,ownership,{id,std::move(wrt)});
    return {std::move(*analyzed.module),std::move(ownership),std::move(differentiated)};
}
s::RuntimeValue f(float x) { return s::RuntimeValue{x}; }
s::RuntimeValue t(std::vector<std::int64_t> shape,std::vector<float> values) {
    return s::RuntimeValue{s::RuntimeTensor{s::TypeKind::F32,std::move(shape),{},std::move(values)}};
}
float scalar(const s::Observation& o) {
    expect(o.ok,"expected successful scalar observation: "+o.format());
    auto* x=std::get_if<float>(&o.value->data); expect(x!=nullptr,"expected f32 scalar: "+o.format()); return *x;
}
std::vector<s::RuntimeValue> tuple(const s::Observation& o) {
    expect(o.ok,"expected successful tuple observation: "+o.format());
    auto* x=std::get_if<s::RuntimeTuple>(&o.value->data); expect(x!=nullptr,"expected tuple: "+o.format()); return *x;
}
const s::RuntimeTensor& tensor(const s::RuntimeValue& value) {
    auto* x=std::get_if<s::RuntimeTensor>(&value.data); expect(x && x->dtype==s::TypeKind::F32,"expected f32 tensor"); return *x;
}
void close(float actual,float expected,float tolerance=1e-5f) {
    expect(std::fabs(actual-expected)<=tolerance,"numeric mismatch actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
void tensorClose(const s::RuntimeValue& value,const std::vector<std::int64_t>& shape,const std::vector<float>& expected,float tolerance=1e-5f) {
    const auto& x=tensor(value); expect(x.shape==shape,"tensor shape mismatch"); expect(x.f32Values.size()==expected.size(),"tensor size mismatch");
    for (std::size_t k=0;k<expected.size();++k) close(x.f32Values[k],expected[k],tolerance);
}
void requireAd(const Program& p,const std::string& id) {
    expect(p.differentiated.ok(),id+" transformation failed: "+(p.differentiated.diagnostics.empty()?std::string{}:p.differentiated.diagnostics[0].code+" "+p.differentiated.diagnostics[0].message));
    expect(s::verify(*p.differentiated.module).ok,id+" generated semantic IR failed verification");
    expect(ad::verify(p.source,p.differentiated).empty(),id+" AD audit failed");
}
bool has(const ad::DifferentiationResult& r,const std::string& code) {
    for (const auto& d:r.diagnostics) if (d.code==code) return true; return false;
}
bool auditHas(const std::vector<ad::AdDiagnostic>& r,const std::string& code) {
    for (const auto& d:r) if (d.code==code) return true; return false;
}
float primal(const Program& p,std::vector<s::RuntimeValue> args) {
    return scalar(s::evaluateCall(p.source,p.source.functions[0].name,args));
}
float central(const Program& p,std::vector<s::RuntimeValue> args,std::size_t parameter,float epsilon=1e-3f) {
    auto plus=args,minus=args;
    std::get<float>(plus[parameter].data)+=epsilon; std::get<float>(minus[parameter].data)-=epsilon;
    return (primal(p,plus)-primal(p,minus))/(2.0f*epsilon);
}
void scalarMatrix() {
    auto p1=compile("fn f(x: f32) -> f32 { return x*x }"); requireAd(p1,"AD01"); close(scalar(ad::executeGrad(p1.differentiated,{f(3)})),6);
    auto p2=compile("fn f(x: f32) -> f32 { return x*x+x }"); requireAd(p2,"AD02"); close(scalar(ad::executeGrad(p2.differentiated,{f(3)})),7);
    expect(p2.differentiated.dump().find(" = add:f32")!=std::string::npos,"AD02 accumulation not explicit in generated IR");
    auto p3=compile("fn f(x: f32, y: f32) -> f32 { return x*y }",{0,1}); requireAd(p3,"AD03");
    auto g3=tuple(ad::executeGrad(p3.differentiated,{f(2),f(5)})); close(std::get<float>(g3[0].data),5); close(std::get<float>(g3[1].data),2);
    auto p4=compile("fn f(x: f32, y: f32) -> f32 { return x*x }",{0,1}); requireAd(p4,"AD04");
    auto g4=tuple(ad::executeGrad(p4.differentiated,{f(4),f(9)})); close(std::get<float>(g4[0].data),8); close(std::get<float>(g4[1].data),0);
    expect(p4.differentiated.dump().find("zero_like:f32")!=std::string::npos,"AD04 missing explicit ZeroLike");
    auto subtract=compile("fn f(x: f32, y: f32) -> f32 { return x-y }",{0,1}); requireAd(subtract,"subtract");
    auto gs=tuple(ad::executeGrad(subtract.differentiated,{f(4),f(9)})); close(std::get<float>(gs[0].data),1); close(std::get<float>(gs[1].data),-1);
    close(central(p2,{f(3)},0),7,0.002f);
}
void tensorMatrix() {
    auto p5=compile("fn f(x: Tensor<f32,1>) -> f32 { return sum(x .* x, 0) }"); requireAd(p5,"AD05");
    tensorClose(*ad::executeGrad(p5.differentiated,{t({3},{1,2,3})}).value,{3},{2,4,6});
    auto p6=compile("fn f(x: Tensor<f32,2>) -> f32 { let y=x .* x; let s=sum(y,0); return sum(s,0) }"); requireAd(p6,"AD06");
    tensorClose(*ad::executeGrad(p6.differentiated,{t({2,2},{1,2,3,4})}).value,{2,2},{2,4,6,8});
    auto p7=compile("fn f(a: Tensor<f32,2>, b: Tensor<f32,2>) -> f32 { let y=a*b; return sum(sum(y,0),0) }",{0}); requireAd(p7,"AD07");
    auto A=t({2,2},{1,2,3,4}),B=t({2,2},{5,6,7,8});
    tensorClose(*ad::executeGrad(p7.differentiated,{A,B}).value,{2,2},{11,15,11,15});
    auto p8=compile("fn f(a: Tensor<f32,2>, b: Tensor<f32,2>) -> f32 { return sum(sum(a*b,0),0) }",{0,1}); requireAd(p8,"AD08");
    auto g8=tuple(ad::executeGrad(p8.differentiated,{A,B})); tensorClose(g8[0],{2,2},{11,15,11,15}); tensorClose(g8[1],{2,2},{4,4,6,6});
    auto p9=compile("fn f(x: Tensor<f32,2>) -> f32 { return sum(sum(x.T,0),0) }"); requireAd(p9,"AD09");
    tensorClose(*ad::executeGrad(p9.differentiated,{A}).value,{2,2},{1,1,1,1});
    auto sumAxis1=compile("fn f(x: Tensor<f32,2>) -> f32 { return sum(sum(x,1),0) }"); requireAd(sumAxis1,"sum-axis-1");
    tensorClose(*ad::executeGrad(sumAxis1.differentiated,{t({2,3},{1,2,3,4,5,6})}).value,{2,3},{1,1,1,1,1,1});
    auto p10=compile("fn f(x: Tensor<f32,2>, b: Tensor<f32,1>) -> f32 { return sum(sum(x+b,0),0) }",{0,1}); requireAd(p10,"AD10");
    auto g10=tuple(ad::executeGrad(p10.differentiated,{t({2,3},{1,2,3,4,5,6}),t({3},{10,20,30})}));
    tensorClose(g10[0],{2,3},{1,1,1,1,1,1}); tensorClose(g10[1],{3},{2,2,2});
    auto p11=compile("fn f(x: Tensor<f32,2>, b: Tensor<f32,1>) -> f32 { return sum(sum(x .* b,0),0) }",{0,1}); requireAd(p11,"AD11");
    auto g11=tuple(ad::executeGrad(p11.differentiated,{t({2,3},{1,2,3,4,5,6}),t({3},{10,20,30})}));
    tensorClose(g11[0],{2,3},{10,20,30,10,20,30}); tensorClose(g11[1],{3},{5,7,9});
    auto scaling=compile("fn f(x: Tensor<f32,2>, a: f32) -> f32 { return sum(sum(x*a,0),0) }",{0,1}); requireAd(scaling,"tensor-scalar");
    auto scalingGrad=tuple(ad::executeGrad(scaling.differentiated,{t({2,2},{1,2,3,4}),f(3)}));
    tensorClose(scalingGrad[0],{2,2},{3,3,3,3}); close(std::get<float>(scalingGrad[1].data),10);
    auto p12=compile("fn f(x: f32) -> f32 { return x*stop_gradient(x) }"); requireAd(p12,"AD12"); close(scalar(ad::executeGrad(p12.differentiated,{f(3)})),3);
    auto p13=compile("fn f(x: Tensor<f32,2>) -> f32 { let y=x .* x; return sum(sum(y+y,0),0) }"); requireAd(p13,"AD13");
    tensorClose(*ad::executeGrad(p13.differentiated,{A}).value,{2,2},{4,8,12,16});
    auto p14=compile("fn f(x: Tensor<f32,1>) -> Tensor<f32,1> { return x .* x }"); requireAd(p14,"AD14");
    tensorClose(*ad::executeVjp(p14.differentiated,{t({3},{1,2,3})},t({3},{2,3,4})).value,{3},{4,12,24});
    auto finiteTensor=[&](const Program& p,std::vector<s::RuntimeValue> args,std::size_t parameter,std::size_t element) {
        constexpr float epsilon=1e-3f; auto plus=args,minus=args;
        std::get<s::RuntimeTensor>(plus[parameter].data).f32Values[element]+=epsilon;
        std::get<s::RuntimeTensor>(minus[parameter].data).f32Values[element]-=epsilon;
        return (primal(p,plus)-primal(p,minus))/(2*epsilon);
    };
    close(finiteTensor(p6,{A},0,2),6,0.01f);
    close(finiteTensor(p8,{A,B},0,1),15,0.03f);
    close(finiteTensor(p8,{A,B},1,2),6,0.03f);
}
void rejectionAndFailureMatrix() {
    auto i64=compile("fn f(x: i64, y: f32) -> f32 { return y*y }",{0});
    expect(!i64.differentiated.ok() && has(i64.differentiated,"ADV02"),"AD15 i64 WRT not rejected");
    auto boolSource=parse("fn f(x: bool, y: f32) -> f32 { return y*y }","bool.th"); auto boolSem=s::analyze(*boolSource.module); auto boolOwn=a::analyze(*boolSem.module);
    auto badBool=ad::differentiate(*boolSem.module,boolOwn,{1,{0}}); expect(has(badBool,"ADV02"),"AD16 bool WRT not rejected");
    auto p17=compile("fn f(x: f32, flag: bool) -> f32 { if flag { return x*x } else { return x+x } }",{0});
    expect(has(p17.differentiated,"AD-ELIGIBILITY-CONTROL"),"AD17 control flow not rejected");
    auto p18=compile("fn f(x: borrow mut Tensor<f32,2>) -> f32 { return sum(sum(x,0),0) }",{0});
    expect(has(p18.differentiated,"AD-ELIGIBILITY-ACCESS") || has(p18.differentiated,"AD-ELIGIBILITY-EFFECT"),"AD18 mutation not rejected");
    auto p19=compile("fn g(x: f32) -> f32 { return x*x }\nfn f(x: f32) -> f32 { return g(x) }",{0},1);
    expect(has(p19.differentiated,"AD-ELIGIBILITY-OP"),"AD19 ordinary call not rejected");
    auto p20=compile("fn f(a: Tensor<f32,2>, b: Tensor<f32,2>) -> f32 { return sum(sum(a*b,0),0) }",{0}); requireAd(p20,"AD20");
    auto failure=ad::executeGrad(p20.differentiated,{t({2,3},{1,2,3,4,5,6}),t({2,2},{1,2,3,4})});
    expect(!failure.ok && failure.errorId=="TH-SPEC-SHAPE","AD20 forward failure was not preserved");
}
void metadataEffectsAndDeterminism() {
    auto square=compile("fn f(x: Tensor<f32,2>) -> f32 { return sum(sum(x .* x,0),0) }"); requireAd(square,"AD21/22");
    bool savedX=false,resource=false; for (const auto& save:square.differentiated.saves) {
        savedX|=save.reason.find("Multiply")!=std::string::npos; resource|=!save.resources.empty();
        expect(!save.implicitCopy,"save became implicit copy"); expect(s::differentiableType(save.type),"save type lost");
    }
    expect(savedX,"x*x did not save x"); expect(resource,"tensor save lost ResourceId provenance");
    auto mm=compile("fn f(a: Tensor<f32,2>, b: Tensor<f32,2>) -> f32 { return sum(sum(a*b,0),0) }",{0,1}); requireAd(mm,"save-matmul");
    int matmulSaves=0; for (const auto& save:mm.differentiated.saves) matmulSaves+=save.reason.find("Matmul")!=std::string::npos;
    expect(matmulSaves>=2,"matmul did not save both operands");
    auto mmFacts=a::analyze(*mm.differentiated.module); expect(mmFacts.ok(),"generated matmul ownership/effect analysis failed");
    expect(a::auditFacts(*mm.differentiated.module,mmFacts).empty(),"generated matmul ownership fact audit failed");
    auto add=compile("fn f(x: f32, y: f32) -> f32 { return x+y }",{0,1}); requireAd(add,"save-add");
    for (const auto& save:add.differentiated.saves) expect(save.reason.find("broadcast")==std::string::npos,"scalar Add saved operands solely for local derivative");
    auto repeat=ad::differentiate(square.source,square.ownership,{1,{0}});
    expect(repeat.ok(),"AD23 repeat transformation failed"); expect(square.differentiated.dump()==repeat.dump(),"AD23 transformation is nondeterministic");
    auto generatedFacts=a::analyze(*square.differentiated.module); expect(generatedFacts.ok(),"generated ownership analysis failed: "+
        (generatedFacts.diagnostics.empty()?std::string{}:generatedFacts.diagnostics[0].format()));
    expect(generatedFacts.functionEffects.at(2).kinds & static_cast<a::EffectSet>(a::EffectKind::MayTrap),"generated shape-helper backward missing MayTrap");
    auto scalarPure=compile("fn f(x: f32) -> f32 { return x*x }"); requireAd(scalarPure,"effect-pure");
    auto scalarFacts=a::analyze(*scalarPure.differentiated.module); expect(scalarFacts.functionEffects.at(2).kinds==0,"scalar backward not Pure");
    expect(a::auditFacts(*square.differentiated.module,generatedFacts).empty(),"generated ownership fact audit failed");
}
void adversarialMatrix() {
    auto base=compile("fn f(x: f32, y: f32) -> f32 { return x*x }",{0,1}); requireAd(base,"ADV-base");
    auto bad=base.differentiated; bad.wrtParameters[0]=99; expect(auditHas(ad::verify(base.source,bad),"ADV01"),"ADV01 missing");
    auto integer=compile("fn f(x: i64, y: f32) -> f32 { return y*y }",{1}); requireAd(integer,"ADV02-base");
    bad=integer.differentiated; bad.wrtParameters={0}; expect(auditHas(ad::verify(integer.source,bad),"ADV02"),"ADV02 missing");
    bad=base.differentiated; bad.gradients[0].type=s::scalar(s::TypeKind::I64); expect(auditHas(ad::verify(base.source,bad),"ADV03"),"ADV03 missing");
    bad=base.differentiated; bad.saves[0].primal=999; expect(auditHas(ad::verify(base.source,bad),"ADV04"),"ADV04 missing");
    bad=base.differentiated; bad.saves[0].slot=2; expect(auditHas(ad::verify(base.source,bad),"ADV05"),"ADV05 missing");
    bad=base.differentiated; bad.module->functions[1].parameters.erase(bad.module->functions[1].parameters.begin()); expect(auditHas(ad::verify(base.source,bad),"ADV06"),"ADV06 missing");
    bad=base.differentiated; for (auto& step:bad.module->functions[1].body.steps) if (auto* i=std::get_if<s::Instruction>(&step);i&&i->op==s::Op::Add) { i->type=s::scalar(s::TypeKind::I64); break; }
    expect(auditHas(ad::verify(base.source,bad),"ADV07"),"ADV07 missing");
    auto broadcast=compile("fn f(x: Tensor<f32,2>, b: Tensor<f32,1>) -> f32 { return sum(sum(x+b,0),0) }",{0,1}); requireAd(broadcast,"ADV-shapes");
    bad=broadcast.differentiated; for (auto& step:bad.module->functions[1].body.steps) if (auto* i=std::get_if<s::Instruction>(&step);i&&i->op==s::Op::ReduceToShape) { i->type=s::scalar(s::TypeKind::I64); break; }
    expect(auditHas(ad::verify(broadcast.source,bad),"ADV08"),"ADV08 malformed reduction missing");
    bad=broadcast.differentiated; for (auto& step:bad.module->functions[1].body.steps) if (auto* i=std::get_if<s::Instruction>(&step);i&&i->op==s::Op::BroadcastToShape) { i->type=s::scalar(s::TypeKind::I64); break; }
    expect(auditHas(ad::verify(broadcast.source,bad),"ADV09"),"ADV09 malformed broadcast missing");
    bad=base.differentiated; bad.gradients.pop_back(); expect(auditHas(ad::verify(base.source,bad),"ADV10"),"ADV10 missing");
    auto unused=base.differentiated; bool changed=false;
    for (auto& step:unused.module->functions[1].body.steps) if (auto* i=std::get_if<s::Instruction>(&step);i&&i->op==s::Op::ZeroLike) { i->type=s::scalar(s::TypeKind::I64); changed=true; break; }
    expect(changed && !s::verify(*unused.module).ok,"malformed ZeroLike accepted");
    auto nonDiffZero=base.differentiated;
    for (auto& step:nonDiffZero.module->functions[1].body.steps) if (auto* i=std::get_if<s::Instruction>(&step);i&&i->op==s::Op::ZeroLike) {
        for (auto& p:nonDiffZero.module->functions[1].parameters) if (p.id==i->operands[0]) p.type=s::scalar(s::TypeKind::I64);
        i->type=s::scalar(s::TypeKind::I64); break;
    }
    expect(!s::verify(*nonDiffZero.module).ok,"ZeroLike on non-differentiable type accepted");
    auto helperModule=[&](s::Op op,std::vector<std::int64_t> sourceShape,std::vector<std::int64_t> targetShape,std::uint32_t axis) {
        s::Module m; m.source="manual-ad-helper"; m.initializer.id=1; m.initializer.terminated=true;
        s::Function fn; fn.id=1; fn.name="helper"; fn.generated=true; fn.sourceFunction=1; fn.generatedRole="backward";
        fn.span={{0,0,1,1},{0,0,1,1}}; fn.body.id=1; fn.body.terminated=true;
        auto shapeFact=[](const std::vector<std::int64_t>& values) { s::ShapeFact x; for (auto v:values) x.extents.push_back(v); return x; };
        auto st=s::tensor(s::scalar(s::TypeKind::F32),static_cast<std::uint32_t>(sourceShape.size()));
        auto tt=s::tensor(s::scalar(s::TypeKind::F32),static_cast<std::uint32_t>(targetShape.size()));
        fn.parameters={{1,"value",st,shapeFact(sourceShape),fn.span,1,s::AccessMode::Read},
                       {2,"target",tt,shapeFact(targetShape),fn.span,2,s::AccessMode::Read}};
        s::Instruction i; i.id=3; i.op=op; i.type=tt; i.shape=shapeFact(targetShape); i.span=fn.span; i.operands={1,2}; i.axis=axis;
        fn.body.steps={i,s::Flow{s::Flow::Kind::Return,3,fn.span}}; fn.result=tt; m.functions.push_back(std::move(fn)); return m;
    };
    expect(!s::verify(helperModule(s::Op::ReduceToShape,{2,3},{2},0)).ok,"known-incompatible ReduceToShape accepted");
    expect(!s::verify(helperModule(s::Op::BroadcastToShape,{3},{2,2},0)).ok,"known-incompatible BroadcastToShape accepted");
    bad=base.differentiated; bad.module->functions[1].generatedRole.clear(); expect(!s::verify(*bad.module).ok,"invalid generated provenance accepted");
}
void robustnessAndDependencyAudit() {
    auto zero=compile("fn f(x: Tensor<f32,2>) -> f32 { return sum(sum(x .* x,0),0) }"); requireAd(zero,"zero-size");
    tensorClose(*ad::executeGrad(zero.differentiated,{t({0,3},{})}).value,{0,3},{});
    auto nan=compile("fn f(x: f32) -> f32 { return x*x }"); requireAd(nan,"nan-inf");
    expect(std::isnan(scalar(ad::executeGrad(nan.differentiated,{f(std::nanf(""))}))),"NaN gradient did not propagate");
    expect(std::isinf(scalar(ad::executeGrad(nan.differentiated,{f(INFINITY)}))),"Inf gradient did not propagate");
    std::string combined;
    for (const char* path:{"/include/autodiff/v0/Autodiff.hpp","/src/autodiff/v0/Autodiff.cpp"}) {
        std::ifstream in(std::string(THIRAN_SOURCE_DIR)+path); std::ostringstream content; content<<in.rdbuf(); combined+=content.str();
    }
    for (const char* forbidden:{"torch/","PyTorch","jax","TensorFlow","numpy","Triton"})
        expect(combined.find(forbidden)==std::string::npos,std::string("AD24 forbidden dependency: ")+forbidden);
    for (const char* forbidden:{"thread_local","global tape","recording mode","hidden runtime map"})
        expect(combined.find(forbidden)==std::string::npos,std::string("AD25 hidden tape marker: ")+forbidden);
}
}
int main() {
    try {
        static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
        scalarMatrix(); tensorMatrix(); rejectionAndFailureMatrix(); metadataEffectsAndDeterminism(); adversarialMatrix(); robustnessAndDependencyAudit();
        std::cout << "PASS " << checks << " checks\n";
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
