#include "analysis/v0/Ownership.hpp"
#include "autodiff/v0/Autodiff.hpp"
#include "backend/v0/NativeCpu.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace s=thiran::v0::semantic;
namespace a=thiran::v0::analysis;
namespace ad=thiran::v0::autodiff;
namespace b=thiran::v0::backend;
using thiran::v0::parse;

namespace {
int checks=0;
void expect(bool value,const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
bool diagnostic(const s::AnalysisResult& result,const std::string& category) {
    for (const auto& item:result.diagnostics) if (item.category==category) return true;
    return false;
}
bool diagnostic(const a::OwnershipAnalysisResult& result,const std::string& category) {
    for (const auto& item:result.diagnostics) if (item.category==category) return true;
    return false;
}
bool diagnostic(const ad::DifferentiationResult& result,const std::string& code) {
    for (const auto& item:result.diagnostics) if (item.code==code) return true;
    return false;
}
s::AnalysisResult analyze(const std::string& text) {
    auto parsed=parse(text,"scan.th");
    expect(parsed.module.has_value(),"parse failed: "+(parsed.diagnostics.empty()?std::string{}:parsed.diagnostics[0].format()));
    return s::analyze(*parsed.module);
}
s::Module module(const std::string& text) {
    auto result=analyze(text);
    expect(result.module.has_value(),"analysis failed: "+(result.diagnostics.empty()?std::string{}:result.diagnostics[0].format()));
    expect(s::verify(*result.module).ok,"semantic verifier rejected analyzer output");
    return std::move(*result.module);
}
s::RuntimeValue i(std::int64_t value) { return s::RuntimeValue{value}; }
s::RuntimeValue f(float value) { return s::RuntimeValue{value}; }
s::RuntimeValue boolean(bool value) { return s::RuntimeValue{value}; }
s::RuntimeValue i64Tensor(std::vector<std::int64_t> shape,std::vector<std::int64_t> values) {
    return s::RuntimeValue{s::RuntimeTensor{s::TypeKind::I64,std::move(shape),std::move(values),{}}};
}
s::RuntimeValue f32Tensor(std::vector<std::int64_t> shape,std::vector<float> values) {
    return s::RuntimeValue{s::RuntimeTensor{s::TypeKind::F32,std::move(shape),{},std::move(values)}};
}
const s::RuntimeTuple& tuple(const s::Observation& observation) {
    expect(observation.ok,"expected scan success: "+observation.format());
    const auto* result=std::get_if<s::RuntimeTuple>(&observation.value->data);
    expect(result!=nullptr,"scan result was not a tuple");
    return *result;
}
const s::RuntimeTensor& tensor(const s::RuntimeValue& value) {
    const auto* result=std::get_if<s::RuntimeTensor>(&value.data);
    expect(result!=nullptr,"expected tensor runtime value");
    return *result;
}
const s::Structured* scanIn(const s::Function& function) {
    for (const auto& step:function.body.steps) if (const auto* structured=std::get_if<s::Structured>(&step);
        structured && structured->kind==s::Structured::Kind::Scan) return structured;
    return nullptr;
}

const std::string scalarSource=R"THIRAN(
fn scalar_step(x: i64, state: i64) -> (i64, i64) {
    let next = state + x
    return (next, next)
}
fn scalar_scan(xs: Tensor<i64,1>, initial: i64) -> (Tensor<i64,1>, i64) {
    return scan(scalar_step, xs, initial)
}
)THIRAN";

void coreExecution() {
    auto semantic=module(scalarSource);
    const auto* scan=scanIn(semantic.functions[1]);
    expect(scan && scan->stepFunction==semantic.functions[0].id,"scan is not retained as structured IR");
    expect(s::dump(semantic).find("= scan @1")!=std::string::npos,"structured dump omitted scan");

    auto multi=s::evaluateCall(semantic,"scalar_scan",{i64Tensor({3},{1,2,3}),i(0)});
    const auto& result=tuple(multi);
    expect(tensor(result[0]).shape==std::vector<std::int64_t>{3},"multi-step output shape changed");
    expect(tensor(result[0]).values==std::vector<std::int64_t>({1,3,6}),"outputs are not in iteration order");
    expect(std::get<std::int64_t>(result[1].data)==6,"final scalar state is wrong");

    auto one=s::evaluateCall(semantic,"scalar_scan",{i64Tensor({1},{7}),i(2)});
    const auto& oneResult=tuple(one);
    expect(tensor(oneResult[0]).values==std::vector<std::int64_t>{9} &&
        std::get<std::int64_t>(oneResult[1].data)==9,"single-element scan failed");

    auto zero=s::evaluateCall(semantic,"scalar_scan",{i64Tensor({0},{}),i(std::numeric_limits<std::int64_t>::max())});
    const auto& zeroResult=tuple(zero);
    expect(tensor(zeroResult[0]).shape==std::vector<std::int64_t>{0} && tensor(zeroResult[0]).values.empty(),
        "zero-length scalar output is not a typed empty tensor");
    expect(std::get<std::int64_t>(zeroResult[1].data)==std::numeric_limits<std::int64_t>::max(),
        "zero-length scan did not preserve initial state");
    expect(multi.format()==s::evaluateCall(semantic,"scalar_scan",{i64Tensor({3},{1,2,3}),i(0)}).format(),
        "repeated scan execution is nondeterministic");
    std::vector<std::int64_t> manyValues(64,1);
    auto many=s::evaluateCall(semantic,"scalar_scan",{i64Tensor({64},manyValues),i(0)});
    expect(many.ok && tensor(tuple(many)[0]).values.back()==64,"many finite iterations did not complete in order");
}

void tensorAndTupleState() {
    auto tensorState=module(R"THIRAN(
fn step(x: Tensor<i64,1>, state: Tensor<i64,1>) -> (i64, Tensor<i64,1>) {
    let next = state + x
    return (sum(next, 0), next)
}
fn run(xs: Tensor<i64,2>, state: Tensor<i64,1>) -> (Tensor<i64,1>, Tensor<i64,1>) {
    return scan(step, xs, state)
}
)THIRAN");
    auto observation=s::evaluateCall(tensorState,"run",{i64Tensor({2,2},{1,2,3,4}),i64Tensor({2},{0,0})});
    const auto& result=tuple(observation);
    expect(tensor(result[0]).values==std::vector<std::int64_t>({3,10}),"tensor-state outputs are wrong");
    expect(tensor(result[1]).values==std::vector<std::int64_t>({4,6}),"tensor carried state is wrong");

    auto tupleState=module(R"THIRAN(
fn step(x: i64, state: (i64, i64)) { return (x, state) }
fn run(xs: Tensor<i64,1>, a: i64, b: i64) {
    let state = (a, b)
    return scan(step, xs, state)
}
)THIRAN");
    auto tupleObservation=s::evaluateCall(tupleState,"run",{i64Tensor({2},{4,5}),i(8),i(9)});
    const auto& outer=tuple(tupleObservation);
    expect(tensor(outer[0]).values==std::vector<std::int64_t>({4,5}),"tuple-state scan output is wrong");
    const auto& finalState=std::get<s::RuntimeTuple>(outer[1].data);
    expect(std::get<std::int64_t>(finalState[0].data)==8 && std::get<std::int64_t>(finalState[1].data)==9,
        "tuple carried state changed");

    auto rankZeroState=module(R"THIRAN(
fn step(x:i64,state:Tensor<i64,0>)->(i64,Tensor<i64,0>){return(x,state)}
fn run(xs:Tensor<i64,1>,state:Tensor<i64,0>)->(Tensor<i64,1>,Tensor<i64,0>){return scan(step,xs,state)}
)THIRAN");
    auto rankZero=s::evaluateCall(rankZeroState,"run",{i64Tensor({2},{6,7}),i64Tensor({}, {11})});
    const auto& rankZeroResult=tuple(rankZero);
    expect(tensor(rankZeroResult[0]).shape==std::vector<std::int64_t>{2} &&
        tensor(rankZeroResult[1]).shape.empty() && tensor(rankZeroResult[1]).values==std::vector<std::int64_t>{11},
        "rank-0 tensor state was confused with scalar state");

    auto stacked=module(R"THIRAN(
fn step(x: i64, state: i64) -> (Tensor<i64,1>, i64) {
    let next = state + x
    return ([next, next], next)
}
fn run(xs: Tensor<i64,1>, state: i64) -> (Tensor<i64,2>, i64) { return scan(step, xs, state) }
)THIRAN");
    auto empty=s::evaluateCall(stacked,"run",{i64Tensor({0},{}),i(3)});
    const auto& emptyResult=tuple(empty);
    expect(tensor(emptyResult[0]).shape==std::vector<std::int64_t>({0,2}) && tensor(emptyResult[0]).values.empty(),
        "zero-length tensor output did not use the structural shape contract");
    auto values=s::evaluateCall(stacked,"run",{i64Tensor({2},{1,2}),i(0)});
    expect(tensor(tuple(values)[0]).values==std::vector<std::int64_t>({1,1,3,3}),"rank-1 outputs were not stacked");
}

void structuredControlAndCaptures() {
    auto controlled=module(R"THIRAN(
fn step(x: i64, state: i64, take: bool) -> (i64, i64) {
    if take {
        let next = state + x
        return (next, next)
    } else {
        return (x, state)
    }
}
fn run(xs: Tensor<i64,1>, state: i64, take: bool) -> (Tensor<i64,1>, i64) {
    if take { return scan(step, xs, state, take) }
    else { return scan(step, xs, state, take) }
}
)THIRAN");
    auto yes=s::evaluateCall(controlled,"run",{i64Tensor({3},{1,2,3}),i(0),boolean(true)});
    expect(tensor(tuple(yes)[0]).values==std::vector<std::int64_t>({1,3,6}),"conditional step true path failed");
    auto no=s::evaluateCall(controlled,"run",{i64Tensor({3},{1,2,3}),i(9),boolean(false)});
    const auto& noResult=tuple(no);
    expect(tensor(noResult[0]).values==std::vector<std::int64_t>({1,2,3}) &&
        std::get<std::int64_t>(noResult[1].data)==9,"scan under structured conditional failed");

    auto loopStep=module(R"THIRAN(
fn step(x:i64,state:i64)->(i64,i64){
    let mut next=state
    for k in 0:x { next=next+1 }
    return(next,next)
}
fn run(xs:Tensor<i64,1>,state:i64)->(Tensor<i64,1>,i64){return scan(step,xs,state)}
)THIRAN");
    auto loopResult=s::evaluateCall(loopStep,"run",{i64Tensor({2},{2,3}),i(0)});
    expect(tensor(tuple(loopResult)[0]).values==std::vector<std::int64_t>({2,5}),
        "structured loop inside scan step failed");
}

void staticRejections() {
    auto reject=[&](const std::string& text,const std::string& code) {
        auto result=analyze(text);
        expect(!result.module && diagnostic(result,code),"missing static rejection "+code);
    };
    reject("fn step(x:i64,s:i64)->(i64,i64){return(x,s)}\nfn run(x:i64,s:i64){return scan(step,x,s)}",
        "TH012-INPUT-TYPE");
    reject("fn step(x:bool,s:i64)->(i64,i64){return(s,s)}\nfn run(x:Tensor<i64,1>,s:i64){return scan(step,x,s)}",
        "TH012-INPUT-TYPE");
    reject("fn step(x:i64,s:i64)->(i64,i64){return(x,s)}\nfn run(x:Tensor<i64,1>,s:bool){return scan(step,x,s)}",
        "TH012-STATE-TYPE");
    reject("fn step(x:i64)->(i64,i64){return(x,x)}\nfn run(x:Tensor<i64,1>,s:i64){return scan(step,x,s)}",
        "TH012-STEP-ARITY");
    reject("fn step(x:i64,s:i64)->i64{return x}\nfn run(x:Tensor<i64,1>,s:i64){return scan(step,x,s)}",
        "TH012-STEP-RESULT");
    reject("fn step(x:i64,s:i64)->(i64,bool){return(x,true)}\nfn run(x:Tensor<i64,1>,s:i64){return scan(step,x,s)}",
        "TH012-NEXT-STATE-TYPE");
    reject("fn step(x:i64,s:Buffer<u8>)->(i64,Buffer<u8>){return(x,s)}\nfn run(x:Tensor<i64,1>,s:Buffer<u8>){return scan(step,x,s)}",
        "TH012-STATE-TYPE");
    reject(R"THIRAN(
fn step(x:i64,s:i64)->(Tensor<i64,1>,i64){if true{return([x,x],s)}else{return([x,x,x],s)}}
fn run(x:Tensor<i64,1>,s:i64){return scan(step,x,s)}
)THIRAN","TH012-OUTPUT-SHAPE");
    reject("fn step(x:i64,s:i64,p:i64)->(i64,i64){return(x,s)}\nfn run(x:Tensor<i64,1>,s:i64){return scan(step,x,s,true)}",
        "TH012-CAPTURE-TYPE");
    reject(R"THIRAN(
fn step(x:i64,s:Tensor<i64,1>)->(i64,Tensor<i64,1>){return(x,[x,x])}
fn run(xs:Tensor<i64,1>){let state=[0,0,0];return scan(step,xs,state)}
)THIRAN","TH012-STATE-SHAPE");
}

void failureAndDynamicChecks() {
    auto semantic=module(scalarSource);
    auto first=s::evaluateCall(semantic,"scalar_scan",{
        i64Tensor({1},{1}),i(std::numeric_limits<std::int64_t>::max())});
    expect(!first.ok && first.errorId=="TH-SPEC-I64-OVERFLOW","first-iteration failure was not propagated");
    auto middle=s::evaluateCall(semantic,"scalar_scan",{
        i64Tensor({3},{1,1,1}),i(std::numeric_limits<std::int64_t>::max()-1)});
    expect(!middle.ok && middle.errorId=="TH-SPEC-I64-OVERFLOW","middle-iteration failure was not propagated");
    auto final=s::evaluateCall(semantic,"scalar_scan",{
        i64Tensor({2},{1,1}),i(std::numeric_limits<std::int64_t>::max()-1)});
    expect(!final.ok && final.errorId=="TH-SPEC-I64-OVERFLOW","final-iteration failure was not propagated");
    auto negative=s::evaluateCall(semantic,"scalar_scan",{i64Tensor({-1},{}),i(0)});
    expect(!negative.ok && negative.errorId=="TH-SPEC-SHAPE","negative extent was not checked");
    auto excessive=s::evaluateCall(semantic,"scalar_scan",{i64Tensor({1000001},{}),i(0)});
    expect(!excessive.ok && excessive.errorId=="TH005-RESOURCE-LIMIT","large bound was not checked before iteration");
    auto overflowExtent=s::evaluateCall(semantic,"scalar_scan",{
        i64Tensor({std::numeric_limits<std::int64_t>::max()},{}),i(0)});
    expect(!overflowExtent.ok && overflowExtent.errorId=="TH005-RESOURCE-LIMIT",
        "extent conversion/size overflow was not rejected before iteration");

    auto dynamicState=module(R"THIRAN(
fn step(x:Tensor<i64,1>,state:Tensor<i64,1>)->(i64,Tensor<i64,1>){return(sum(x,0),x)}
fn run(xs:Tensor<i64,2>,state:Tensor<i64,1>)->(Tensor<i64,1>,Tensor<i64,1>){return scan(step,xs,state)}
)THIRAN");
    auto changed=s::evaluateCall(dynamicState,"run",{i64Tensor({1,2},{1,2}),i64Tensor({3},{0,0,0})});
    expect(!changed.ok && changed.errorId=="TH012-STATE-SHAPE","dynamic state-shape change was not rejected");

    auto outputContract=module(R"THIRAN(
fn step(x:i64,state:i64)->(Tensor<i64,1>,i64){return([x,x],state)}
fn run(xs:Tensor<i64,1>,state:i64)->(Tensor<i64,2>,i64){return scan(step,xs,state)}
)THIRAN");
    auto* mutableScan=const_cast<s::Structured*>(scanIn(outputContract.functions[1]));
    mutableScan->outputElementShape.extents={3};
    expect(s::verify(outputContract).ok,"runtime-shape fixture should remain structurally verifier-valid");
    auto changedOutput=s::evaluateCall(outputContract,"run",{i64Tensor({2},{1,2}),i(0)});
    expect(!changedOutput.ok && changedOutput.errorId=="TH012-OUTPUT-SHAPE","dynamic output-shape violation was not rejected immediately");
}

void ownershipEffectsAndAliases() {
    auto aliasing=module(R"THIRAN(
fn step(x:Tensor<f32,1>,state:Tensor<f32,1>)->(f32,Tensor<f32,1>){return(sum(x,0),state)}
fn run(xs:Tensor<f32,2>,state:Tensor<f32,1>)->(Tensor<f32,1>,Tensor<f32,1>){return scan(step,xs,state)}
)THIRAN");
    auto facts=a::analyze(aliasing);
    expect(facts.ok(),"legal immutable recurrent aliasing was rejected");
    const auto* scan=scanIn(aliasing.functions[1]);
    const auto& provenance=facts.valueProvenance.at(aliasing.functions[1].id).at(scan->result);
    expect(provenance.elements.size()==2 && provenance.elements[0].kind==a::ProvenanceKind::Fresh,
        "collected scan outputs are not a fresh logical resource");
    expect(provenance.elements[1].kind==a::ProvenanceKind::PossibleAlias,
        "final state lost legal alias provenance");
    expect(facts.functionEffects.at(aliasing.functions[1].id).kinds==0 &&
        !facts.functionEffects.at(aliasing.functions[1].id).pureTensorCandidate,
        "pure scan effect/extraction summary is wrong");
    expect(s::dump(aliasing).find("copy") == std::string::npos,
        "plain recurrence inserted a hidden per-step copy");

    auto liveAlias=module(R"THIRAN(
fn step(x:Tensor<f32,1>,state:Tensor<f32,1>)->(f32,Tensor<f32,1>){return(sum(x,0),state)}
fn run(xs:Tensor<f32,2>,state:Tensor<f32,1>)->Tensor<f32,1>{
    let alias=state
    scan(step,xs,state)
    return alias
}
)THIRAN");
    expect(a::analyze(liveAlias).ok(),"immutable alias of initial state was consumed or invalidated by scan");
    auto aliasResult=s::evaluateCall(liveAlias,"run",{f32Tensor({1,2},{1,2}),f32Tensor({2},{7,8})});
    expect(aliasResult.ok && tensor(*aliasResult.value).f32Values==std::vector<float>({7,8}),
        "initial-state alias was not preserved logically");

    auto escaping=module(R"THIRAN(
fn step(x:Tensor<i64,1>,state:Tensor<i64,1>)->(i64,Tensor<i64,1>){return(sum(x,0),x)}
fn run(xs:Tensor<i64,2>,state:Tensor<i64,1>)->(Tensor<i64,1>,Tensor<i64,1>){return scan(step,xs,state)}
)THIRAN");
    auto escaped=a::analyze(escaping);
    expect(!escaped.ok() && diagnostic(escaped,"TH012-INPUT-BORROW-ESCAPE"),
        "borrowed leading-axis input escaped into final state");

    auto copied=module(R"THIRAN(
fn step(x:Tensor<i64,1>,state:Tensor<i64,1>)->(i64,Tensor<i64,1>){let next=copy(state);return(sum(x,0),next)}
fn run(xs:Tensor<i64,2>,state:Tensor<i64,1>)->(Tensor<i64,1>,Tensor<i64,1>){return scan(step,xs,state)}
)THIRAN");
    expect(a::analyze(copied).ok(),"explicit copy(state) was rejected");
    auto moved=module(R"THIRAN(
fn step(x:Tensor<i64,1>,state:Tensor<i64,1>)->(i64,Tensor<i64,1>){let next=move(state);return(sum(x,0),next)}
fn run(xs:Tensor<i64,2>,state:Tensor<i64,1>)->(Tensor<i64,1>,Tensor<i64,1>){return scan(step,xs,state)}
)THIRAN");
    expect(a::analyze(moved).ok(),"explicit consuming recurrent state transition was rejected");

    auto illegalMutation=module(R"THIRAN(
fn touch(x:borrow mut Tensor<i64,1>)->i64{return sum(x,0)}
fn step(x:Tensor<i64,1>,state:Tensor<i64,1>)->(i64,Tensor<i64,1>){
    let mut local=state
    let alias=local
    let y=touch(borrow mut local)
    return(sum(alias,0)+y,local)
}
fn run(xs:Tensor<i64,2>,state:Tensor<i64,1>)->(Tensor<i64,1>,Tensor<i64,1>){return scan(step,xs,state)}
)THIRAN");
    auto mutationFacts=a::analyze(illegalMutation);
    expect(!mutationFacts.ok() && diagnostic(mutationFacts,"TH012-UNSUPPORTED-EFFECT"),
        "mutation beneath scan was not rejected at the recurrence boundary");

    auto trapping=module(scalarSource);
    auto trappingFacts=a::analyze(trapping);
    expect(trappingFacts.ok() && (trappingFacts.functionEffects.at(2).kinds&
        static_cast<a::EffectSet>(a::EffectKind::MayTrap)),"step MayTrap effect did not compose into scan");
    auto rng=trapping;
    for (auto& item:rng.functions[0].body.steps) if (auto* instruction=std::get_if<s::Instruction>(&item);
        instruction && instruction->op==s::Op::Add) { instruction->effect=s::EffectClass::Rng; break; }
    expect(s::verify(rng).ok,"effect rejection fixture is malformed semantic IR");
    auto rngFacts=a::analyze(rng);
    expect(!rngFacts.ok() && diagnostic(rngFacts,"TH012-UNSUPPORTED-EFFECT"),
        "unsupported scan step effect was not rejected");
}

void malformedAdAndNative() {
    auto semantic=module(scalarSource);
    auto malformed=semantic;
    auto* scan=const_cast<s::Structured*>(scanIn(malformed.functions[1]));
    scan->stepFunction=99;
    expect(!s::verify(malformed).ok,"malformed scan step target passed verification");
    malformed=semantic;
    scan=const_cast<s::Structured*>(scanIn(malformed.functions[1]));
    scan->result=1;
    expect(!s::verify(malformed).ok,"duplicate scan result ID passed verification");
    malformed=semantic;
    scan=const_cast<s::Structured*>(scanIn(malformed.functions[1]));
    scan->outputElementType=s::scalar(s::TypeKind::Bool);
    expect(!s::verify(malformed).ok,"malformed scan output type passed verification");

    auto f32Recurrence=module(R"THIRAN(
fn step(x:f32,state:f32,a:f32)->(f32,f32){let next=a*state+x;return(next,next)}
fn run(xs:Tensor<f32,1>,state:f32,a:f32)->(Tensor<f32,1>,f32){return scan(step,xs,state,a)}
)THIRAN");
    auto ownership=a::analyze(f32Recurrence);
    expect(ownership.ok(),"f32 recurrence ownership failed");
    auto reused=s::evaluateCall(f32Recurrence,"run",{f32Tensor({3},{1,2,3}),f(1),f(2)});
    const auto& reusedResult=tuple(reused);
    expect(tensor(reusedResult[0]).f32Values==std::vector<float>({3,8,19}) &&
        std::get<float>(reusedResult[1].data)==19,"explicit capture was not reused at every timestep");
    auto differentiated=ad::differentiate(f32Recurrence,ownership,{2,{0,1,2}});
    expect(!differentiated.ok() && diagnostic(differentiated,"AD-ELIGIBILITY-SCAN"),
        "unsupported recurrent AD was not rejected explicitly");

    auto nativeFacts=a::analyze(semantic);
    auto native=b::extractStrictNative(semantic,nativeFacts,"scalar_scan",false);
    expect(!native.ok() && native.diagnostic.find("BACKEND-UNSUPPORTED: Scan")!=std::string::npos,
        "scan was silently admitted to native CPU");
    expect(native.coverage.find("Scan unsupported-native")!=std::string::npos &&
        native.coverage.find("fallback: NONE")!=std::string::npos,
        "native scan rejection did not prove absence of fallback");
}
}

int main() {
    try {
        coreExecution();
        tensorAndTupleState();
        structuredControlAndCaptures();
        staticRejections();
        failureAndDynamicChecks();
        ownershipEffectsAndAliases();
        malformedAdAndNative();
        std::cout << "V0ScanTests PASS " << checks << " checks\n";
    } catch (const std::exception& error) {
        std::cerr << "V0ScanTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
