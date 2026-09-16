#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "training/v0/Training.hpp"
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace s=thiran::v0::semantic;
namespace tr=thiran::v0::training;
namespace ad=thiran::v0::autodiff;
using thiran::v0::parse;

namespace {
int checks=0;
void expect(bool yes,const std::string& message) { ++checks; if (!yes) throw std::runtime_error(message); }
void close(float actual,float expected,float tolerance=1e-5f) {
    expect(std::fabs(actual-expected)<=tolerance,"numeric mismatch actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
bool has(const std::vector<tr::TrainingDiagnostic>& diagnostics,const std::string& code) {
    for (const auto& diagnostic:diagnostics) if (diagnostic.code==code) return true;
    return false;
}
s::RuntimeValue f(float value) { return s::RuntimeValue{value}; }
s::RuntimeValue t(std::vector<std::int64_t> shape,std::vector<float> values) {
    return s::RuntimeValue{s::RuntimeTensor{s::TypeKind::F32,std::move(shape),{},std::move(values)}};
}
float scalar(const s::RuntimeValue& value) { return std::get<float>(value.data); }
const s::RuntimeTensor& tensor(const s::RuntimeValue& value) { return std::get<s::RuntimeTensor>(value.data); }
float only(const s::RuntimeValue& value) { return tensor(value).f32Values.at(0); }
bool equal(const s::RuntimeValue& a,const s::RuntimeValue& b) {
    if (a.data.index()!=b.data.index()) return false;
    if (const auto* x=std::get_if<float>(&a.data)) return std::bit_cast<std::uint32_t>(*x)==std::bit_cast<std::uint32_t>(std::get<float>(b.data));
    if (const auto* x=std::get_if<std::int64_t>(&a.data)) return *x==std::get<std::int64_t>(b.data);
    if (const auto* x=std::get_if<bool>(&a.data)) return *x==std::get<bool>(b.data);
    if (const auto* x=std::get_if<s::RuntimeTensor>(&a.data)) {
        const auto& y=std::get<s::RuntimeTensor>(b.data);
        if (x->dtype!=y.dtype || x->shape!=y.shape || x->values!=y.values || x->f32Values.size()!=y.f32Values.size()) return false;
        for (std::size_t k=0;k<x->f32Values.size();++k)
            if (std::bit_cast<std::uint32_t>(x->f32Values[k])!=std::bit_cast<std::uint32_t>(y.f32Values[k])) return false;
        return true;
    }
    const auto& x=std::get<s::RuntimeTuple>(a.data); const auto& y=std::get<s::RuntimeTuple>(b.data);
    if (x.size()!=y.size()) return false;
    for (std::size_t k=0;k<x.size();++k) if (!equal(x[k],y[k])) return false;
    return true;
}
bool equal(const tr::TrainingState& a,const tr::TrainingState& b) {
    if (a.planSignature!=b.planSignature || a.step!=b.step || a.parameters.size()!=b.parameters.size() ||
        a.optimizer.kind!=b.optimizer.kind || a.optimizer.step!=b.optimizer.step ||
        a.optimizer.velocities.size()!=b.optimizer.velocities.size()) return false;
    for (std::size_t k=0;k<a.parameters.size();++k)
        if (a.parameters[k].id!=b.parameters[k].id || a.parameters[k].type!=b.parameters[k].type ||
            a.parameters[k].shape!=b.parameters[k].shape || !equal(a.parameters[k].value,b.parameters[k].value)) return false;
    for (std::size_t k=0;k<a.optimizer.velocities.size();++k)
        if (a.optimizer.velocities[k].id!=b.optimizer.velocities[k].id ||
            a.optimizer.velocities[k].type!=b.optimizer.velocities[k].type ||
            a.optimizer.velocities[k].shape!=b.optimizer.velocities[k].shape ||
            !equal(a.optimizer.velocities[k].value,b.optimizer.velocities[k].value)) return false;
    return true;
}

const std::string linearSource=R"THIRAN(fn loss(
    x: Tensor<f32,2>,
    target: Tensor<f32,2>,
    weight: Tensor<f32,2>,
    bias: Tensor<f32,1>
) -> f32 {
    let prediction = x * weight + bias
    let error = prediction - target
    let squared = error .* error
    let reduced = sum(squared, 0)
    return sum(reduced, 0)
}
)THIRAN";

s::Module module(const std::string& source) {
    auto parsed=parse(source,"training.th");
    expect(parsed.module.has_value(),"training source parse failed");
    auto analyzed=s::analyze(*parsed.module);
    expect(analyzed.module.has_value(),"training source semantic analysis failed: "+
        (analyzed.diagnostics.empty()?std::string{}:analyzed.diagnostics[0].format()));
    return std::move(*analyzed.module);
}

tr::TrainingPlan plan(const s::Module& source,tr::OptimizerSpec optimizer,
                      std::vector<std::size_t> trainables={3,2},const std::string& function="loss") {
    auto built=tr::createTrainingPlan(source,{function,std::move(trainables),optimizer});
    expect(built.ok(),"TRAIN01 plan construction failed: "+
        (built.diagnostics.empty()?std::string{}:built.diagnostics[0].code+" "+built.diagnostics[0].message));
    return std::move(*built.plan);
}

std::vector<tr::RuntimeInput> batch() {
    return {{0,t({4,1},{0,1,2,3})},{1,t({4,1},{1,3,5,7})}};
}

tr::TrainingState initial(const tr::TrainingPlan& plan) {
    auto initialized=tr::initializeTrainingState(plan,{t({1},{0}),t({1,1},{0})});
    expect(initialized.ok(),"TRAIN02 state initialization failed");
    return std::move(*initialized.state);
}

void firstStepAndOrdering() {
    auto source=module(linearSource);
    auto p=plan(source,{tr::OptimizerKind::SGD,0.02f,0.0f});
    expect(p.parameters.size()==2 && p.parameters[0].sourceParameterName=="bias" && p.parameters[1].sourceParameterName=="weight",
        "parameter descriptors did not retain deliberate plan order");
    expect(p.parameters[0].id!=p.parameters[1].id && p.parameters[0].position==0 && p.parameters[1].position==1,
        "ParameterId/position is not deterministic");
    auto state0=initial(p), snapshot=state0;
    auto result=tr::trainingStep(p,state0,batch());
    expect(result.ok(),"TRAIN03 one training step failed");
    close(*result.loss,84.0f);

    std::vector<s::RuntimeValue> sourceArguments={batch()[0].value,batch()[1].value,state0.parameters[1].value,state0.parameters[0].value};
    auto directLoss=s::evaluateCall(source,"loss",sourceArguments);
    expect(directLoss.ok,"TRAIN04 direct loss evaluation failed"); close(*result.loss,scalar(*directLoss.value));
    auto directGrad=ad::executeGrad(p.differentiated,sourceArguments);
    expect(directGrad.ok,"TRAIN05 direct TH-010 grad failed");
    const auto& directTuple=std::get<s::RuntimeTuple>(directGrad.value->data);
    expect(result.gradients[0].id==p.parameters[0].id && result.gradients[1].id==p.parameters[1].id,
        "parameter gradient order did not follow plan");
    expect(equal(result.gradients[0].value,directTuple[0]) && equal(result.gradients[1].value,directTuple[1]),
        "TRAIN05 training gradients disagree with direct TH-010 execution");
    close(only(result.gradients[0].value),-32.0f); close(only(result.gradients[1].value),-68.0f);
    close(only(result.nextState->parameters[0].value),0.64f); close(only(result.nextState->parameters[1].value),1.36f);
    expect(equal(state0,snapshot),"TRAIN07 input TrainingState was mutated");
    expect(result.nextState->step==1 && result.nextState->optimizer.step==1,"TRAIN09 step did not increment exactly once");
    auto second=tr::trainingStep(p,*result.nextState,batch());
    expect(second.ok() && second.nextState->step==2,"TRAIN08 next step did not consume prior state");
    auto dump=tr::dumpTrainingStep(p,result);
    expect(dump.find("step=0 loss=84")!=std::string::npos && dump.find("id=")!=std::string::npos &&
        dump.find("Tensor<f32,1> shape=[1]")!=std::string::npos,"developer observation is incomplete");
}

struct Trace { std::vector<float> losses; tr::TrainingState final; };
Trace train(const tr::TrainingPlan& plan,std::size_t steps) {
    auto state=initial(plan); Trace trace;
    for (std::size_t k=0;k<steps;++k) {
        auto result=tr::trainingStep(plan,state,batch()); expect(result.ok(),"repeated training step failed");
        trace.losses.push_back(*result.loss); state=std::move(*result.nextState);
    }
    trace.final=std::move(state); return trace;
}

void convergenceDeterminismAndReuse() {
    auto source=module(linearSource);
    auto p=plan(source,{tr::OptimizerKind::SGD,0.02f,0.0f});
    const auto transformDump=p.differentiated.dump();
    auto a=train(p,200),b=train(p,200);
    expect(a.losses.front()==84.0f && a.losses.back()<0.000001f && a.losses.back()<a.losses.front()/1000000.0f,
        "TRAIN11 SGD loss did not decrease substantially");
    close(only(a.final.parameters[1].value),2.0f,0.001f);
    close(only(a.final.parameters[0].value),1.0f,0.001f);
    std::cout << "SGD200 loss=" << a.losses.back() << " weight=" << only(a.final.parameters[1].value)
              << " bias=" << only(a.final.parameters[0].value) << '\n';
    expect(a.losses.size()==b.losses.size(),"TRAIN13 trace length differs");
    for (std::size_t k=0;k<a.losses.size();++k)
        expect(std::bit_cast<std::uint32_t>(a.losses[k])==std::bit_cast<std::uint32_t>(b.losses[k]),"TRAIN13 loss trace is nondeterministic");
    expect(equal(a.final,b.final),"TRAIN13 final state is nondeterministic");
    expect(p.generationCount==1 && p.differentiated.dump()==transformDump,"TRAIN22/23 plan was rebuilt or redifferentiated");
}

void momentumMatrix() {
    auto source=module(linearSource);
    auto p=plan(source,{tr::OptimizerKind::SGDMomentum,0.01f,0.5f});
    auto state0=initial(p);
    expect(state0.optimizer.velocities.size()==2 && only(state0.optimizer.velocities[0].value)==0.0f &&
        only(state0.optimizer.velocities[1].value)==0.0f,"TRAIN14 momentum velocity is not ZeroLike");
    auto first=tr::trainingStep(p,state0,batch()); expect(first.ok(),"momentum first step failed");
    close(only(first.nextState->optimizer.velocities[0].value),-32.0f);
    close(only(first.nextState->optimizer.velocities[1].value),-68.0f);
    close(only(first.nextState->parameters[0].value),0.32f);
    close(only(first.nextState->parameters[1].value),0.68f);
    auto plain=plan(source,{tr::OptimizerKind::SGD,0.01f,0.0f});
    auto plainState=initial(plain); auto plainFirst=tr::trainingStep(plain,plainState,batch());
    expect(plainFirst.ok() && equal(plainFirst.nextState->parameters[0].value,first.nextState->parameters[0].value) &&
        equal(plainFirst.nextState->parameters[1].value,first.nextState->parameters[1].value),
        "TRAIN15 zero-velocity momentum first update differs from plain SGD");
    auto second=tr::trainingStep(p,*first.nextState,batch()); expect(second.ok(),"momentum second step failed");
    close(only(second.gradients[0].value),-21.28f,2e-5f); close(only(second.gradients[1].value),-45.12f,2e-5f);
    close(only(second.nextState->optimizer.velocities[0].value),-37.28f,2e-5f);
    close(only(second.nextState->optimizer.velocities[1].value),-79.12f,2e-5f);
    close(only(second.nextState->parameters[0].value),0.6928f,2e-5f);
    close(only(second.nextState->parameters[1].value),1.4712f,2e-5f);
    expect(second.nextState->optimizer.velocities[0].id==p.parameters[0].id &&
        second.nextState->optimizer.velocities[1].id==p.parameters[1].id,"TRAIN17 momentum ordering changed");
    auto again0=initial(p); auto again1=tr::trainingStep(p,again0,batch()); auto again2=tr::trainingStep(p,*again1.nextState,batch());
    expect(again2.ok() && equal(*second.nextState,*again2.nextState),"momentum execution is nondeterministic");
}

void unusedAndIeee() {
    auto source=module("fn loss(x: f32, unused: f32) -> f32 { return x*x }");
    auto sgd=plan(source,{tr::OptimizerKind::SGD,0.1f,0.0f},{1});
    auto initialized=tr::initializeTrainingState(sgd,{f(3)}); expect(initialized.ok(),"unused state init failed");
    auto result=tr::trainingStep(sgd,*initialized.state,{{0,f(2)}}); expect(result.ok(),"unused training step failed");
    close(scalar(result.gradients[0].value),0.0f); close(scalar(result.nextState->parameters[0].value),3.0f);
    expect(sgd.differentiated.dump().find("zero_like:f32")!=std::string::npos,"TRAIN18 explicit ZeroLike absent");

    auto momentum=plan(source,{tr::OptimizerKind::SGDMomentum,0.1f,0.5f},{1});
    auto momentumState=tr::initializeTrainingState(momentum,{f(3)}); expect(momentumState.ok(),"unused momentum init failed");
    momentumState.state->optimizer.velocities[0].value=f(4);
    auto momentumResult=tr::trainingStep(momentum,*momentumState.state,{{0,f(2)}});
    expect(momentumResult.ok(),"zero-gradient momentum step failed");
    close(scalar(momentumResult.nextState->optimizer.velocities[0].value),2.0f);
    close(scalar(momentumResult.nextState->parameters[0].value),2.8f);

    auto zeroRate=plan(source,{tr::OptimizerKind::SGD,0.0f,0.0f},{1});
    auto zeroState=tr::initializeTrainingState(zeroRate,{f(3)}); auto zeroStep=tr::trainingStep(zeroRate,*zeroState.state,{{0,f(2)}});
    expect(zeroStep.ok() && scalar(zeroStep.nextState->parameters[0].value)==3.0f,"zero learning rate changed parameter");
    tr::GradientValue nanGradient{zeroRate.parameters[0].id,s::scalar(s::TypeKind::F32),{},f(std::nanf(""))};
    auto nan=tr::applyOptimizer(zeroRate,*zeroState.state,{nanGradient});
    expect(nan.ok() && std::isnan(scalar(nan.state->parameters[0].value)),"TRAIN24 NaN gradient did not propagate");
    nanGradient.value=f(INFINITY); auto infinity=tr::applyOptimizer(zeroRate,*zeroState.state,{nanGradient});
    expect(infinity.ok() && std::isnan(scalar(infinity.state->parameters[0].value)),
        "TRAIN24 IEEE zero-times-infinity/subtract policy did not propagate NaN");

    auto emptySource=module("fn loss(x: f32, unused: Tensor<f32,1>) -> f32 { return x*x }");
    auto emptyPlan=plan(emptySource,{tr::OptimizerKind::SGD,0.1f,0.0f},{1});
    auto emptyInitialized=tr::initializeTrainingState(emptyPlan,{t({0},{})});
    auto emptyStep=tr::trainingStep(emptyPlan,*emptyInitialized.state,{{0,f(2)}});
    expect(emptyStep.ok() && tensor(emptyStep.nextState->parameters[0].value).shape==std::vector<std::int64_t>{0},
        "zero-sized legal parameter update failed");
}

void planAndOptimizerAdversarial() {
    auto source=module(linearSource);
    auto invalid=[&](tr::TrainingPlanRequest request,const std::string& code) {
        auto result=tr::createTrainingPlan(source,request); expect(!result.ok() && has(result.diagnostics,code),code+" was not rejected");
    };
    invalid({"missing",{2},{tr::OptimizerKind::SGD,0.1f,0}},"TRN01");
    invalid({"loss",{9},{tr::OptimizerKind::SGD,0.1f,0}},"TRN01");
    invalid({"loss",{2,2},{tr::OptimizerKind::SGD,0.1f,0}},"TRN02");
    auto nondiff=module("fn loss(x: i64, y: f32) -> f32 { return y*y }");
    auto trn03=tr::createTrainingPlan(nondiff,{"loss",{0},{tr::OptimizerKind::SGD,0.1f,0}});
    expect(!trn03.ok() && has(trn03.diagnostics,"TRN03"),"TRN03 was not rejected");
    auto vectorLoss=module("fn loss(x: Tensor<f32,1>) -> Tensor<f32,1> { return x .* x }");
    auto trn04=tr::createTrainingPlan(vectorLoss,{"loss",{0},{tr::OptimizerKind::SGD,0.1f,0}});
    expect(!trn04.ok() && has(trn04.diagnostics,"TRN04"),"TRN04 was not rejected");
    auto access=module("fn loss(x: borrow mut Tensor<f32,1>) -> f32 { return sum(x,0) }");
    auto accessResult=tr::createTrainingPlan(access,{"loss",{0},{tr::OptimizerKind::SGD,0.1f,0}});
    expect(!accessResult.ok() && has(accessResult.diagnostics,"TRN-ACCESS"),"incompatible trainable access mode accepted");
    auto control=module("fn loss(x: f32, flag: bool) -> f32 { if flag { return x*x } else { return x+x } }");
    auto controlResult=tr::createTrainingPlan(control,{"loss",{0},{tr::OptimizerKind::SGD,0.1f,0}});
    expect(!controlResult.ok() && has(controlResult.diagnostics,"AD-ELIGIBILITY-CONTROL"),"unsupported AD function accepted as plan");

    invalid({"loss",{2},{static_cast<tr::OptimizerKind>(99),0.1f,0}},"OPT01");
    invalid({"loss",{2},{tr::OptimizerKind::SGD,std::nanf(""),0}},"OPT02");
    invalid({"loss",{2},{tr::OptimizerKind::SGD,INFINITY,0}},"OPT03");
    invalid({"loss",{2},{tr::OptimizerKind::SGD,-0.1f,0}},"OPT04");
    invalid({"loss",{2},{tr::OptimizerKind::SGDMomentum,0.1f,-0.1f}},"OPT05");
    invalid({"loss",{2},{tr::OptimizerKind::SGDMomentum,0.1f,std::nanf("")}},"OPT05");
    invalid({"loss",{2},{tr::OptimizerKind::SGDMomentum,0.1f,1.0f}},"OPT06");

    auto momentum=plan(source,{tr::OptimizerKind::SGDMomentum,0.01f,0.5f});
    auto state=initial(momentum);
    auto bad=state; bad.optimizer.velocities.pop_back(); expect(has(tr::verifyOptimizerState(momentum,bad),"OPT07"),"OPT07 missing");
    bad=state; bad.optimizer.velocities[0].type=s::scalar(s::TypeKind::F32); expect(has(tr::verifyOptimizerState(momentum,bad),"OPT08"),"OPT08 missing");
    bad=state; bad.optimizer.velocities[0].shape={2}; expect(has(tr::verifyOptimizerState(momentum,bad),"OPT09"),"OPT09 missing");
    bad=state; bad.optimizer.step=1; expect(has(tr::verifyOptimizerState(momentum,bad),"OPT10"),"OPT10 missing");
    bad=state; bad.optimizer.kind=static_cast<tr::OptimizerKind>(99); expect(has(tr::verifyOptimizerState(momentum,bad),"OPT01"),"state OPT01 missing");

    auto corrupted=momentum; corrupted.parameters[1].id=corrupted.parameters[0].id;
    expect(has(tr::verifyTrainingPlan(corrupted),"TRN-DUPLICATE-PARAMETER-ID"),"duplicate ParameterId accepted");
    corrupted=momentum; corrupted.differentiated.gradients.pop_back();
    expect(has(tr::verifyTrainingPlan(corrupted),"TRN15"),"TRN15 corrupted AD metadata accepted");
}

void stateInputAndFailureAdversarial() {
    auto source=module(linearSource);
    auto p=plan(source,{tr::OptimizerKind::SGD,0.02f,0}); auto state=initial(p);
    auto missing=tr::trainingStep(p,state,{{0,batch()[0].value}}); expect(!missing.ok() && has(missing.diagnostics,"TRN05"),"TRN05 missing input accepted");
    auto duplicate=tr::trainingStep(p,state,{{0,batch()[0].value},{0,batch()[0].value},{1,batch()[1].value}});
    expect(!duplicate.ok() && has(duplicate.diagnostics,"TRN05"),"duplicate runtime input accepted");
    auto supplied=tr::trainingStep(p,state,{{0,batch()[0].value},{1,batch()[1].value},{2,state.parameters[1].value}});
    expect(!supplied.ok() && has(supplied.diagnostics,"TRN06"),"TRN06 trainable runtime input accepted");
    auto bad=state; bad.parameters.pop_back(); auto count=tr::trainingStep(p,bad,batch()); expect(!count.ok() && has(count.diagnostics,"TRN07"),"TRN07 accepted");
    bad=state; bad.parameters[0].type=s::scalar(s::TypeKind::F32); auto dtype=tr::trainingStep(p,bad,batch()); expect(!dtype.ok() && has(dtype.diagnostics,"TRN08"),"TRN08 accepted");
    bad=state; bad.parameters[0].shape={2}; auto shape=tr::trainingStep(p,bad,batch()); expect(!shape.ok() && has(shape.diagnostics,"TRN09"),"TRN09 accepted");
    auto gradients=tr::trainingStep(p,state,batch()).gradients; gradients[0].shape={2};
    auto gradient=tr::applyOptimizer(p,state,gradients); expect(!gradient.ok() && has(gradient.diagnostics,"TRN10"),"TRN10 accepted");
    bad=state; ++bad.planSignature; auto incompatible=tr::trainingStep(p,bad,batch()); expect(!incompatible.ok() && has(incompatible.diagnostics,"TRN11"),"TRN11 accepted");
    auto reordered=state; std::swap(reordered.parameters[0],reordered.parameters[1]);
    auto order=tr::trainingStep(p,reordered,batch()); expect(!order.ok() && has(order.diagnostics,"TRN14"),"TRN14 accepted");

    auto snapshot=state;
    auto badBatch=batch(); badBatch[0].value=t({4,2},{0,0,1,1,2,2,3,3});
    auto forward=tr::trainingStep(p,state,badBatch);
    expect(!forward.ok() && has(forward.diagnostics,"TH-SPEC-SHAPE") && !forward.nextState,"TRN12 forward failure not preserved");
    expect(equal(state,snapshot),"TRN12 forward failure mutated input state");
    bad=state; bad.optimizer.velocities.push_back({p.parameters[0].id,p.parameters[0].type,{1},t({1},{0})});
    auto badSnapshot=bad; auto optimizer=tr::trainingStep(p,bad,batch());
    expect(!optimizer.ok() && has(optimizer.diagnostics,"OPT07") && !optimizer.nextState,"TRN13 malformed optimizer state accepted");
    expect(equal(bad,badSnapshot),"TRN13 malformed optimizer state mutated parameters");
}

void dependencyAudit() {
    std::string combined;
    for (const char* path:{"/include/training/v0/Training.hpp","/src/training/v0/Training.cpp"}) {
        std::ifstream in(std::string(THIRAN_SOURCE_DIR)+path); std::ostringstream text; text<<in.rdbuf(); combined+=text.str();
    }
    for (const char* forbidden:{"torch/","PyTorch","numpy","JAX","TensorFlow","autograd","Triton","Graph.hpp","Region.hpp"})
        expect(combined.find(forbidden)==std::string::npos,std::string("TRAIN25 forbidden dependency: ")+forbidden);
}
}

int main() {
    try {
        static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
        firstStepAndOrdering();
        convergenceDeterminismAndReuse();
        momentumMatrix();
        unusedAndIeee();
        planAndOptimizerAdversarial();
        stateInputAndFailureAdversarial();
        dependencyAudit();
        std::cout << "PASS " << checks << " checks\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
