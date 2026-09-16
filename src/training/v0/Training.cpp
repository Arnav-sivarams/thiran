#include "training/v0/Training.hpp"
#include "semantic/v0/Verifier.hpp"
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <type_traits>

namespace thiran::v0::training {
namespace {
using namespace semantic;

void add(std::vector<TrainingDiagnostic>& out,std::string code,std::string message) {
    out.push_back({std::move(code),std::move(message)});
}

const Function* function(const semantic::Module& module,FunctionId id) {
    for (const auto& fn:module.functions) if (fn.id==id) return &fn;
    return nullptr;
}

const Function* function(const semantic::Module& module,const std::string& name) {
    for (const auto& fn:module.functions) if (fn.name==name) return &fn;
    return nullptr;
}

ParameterId parameterId(FunctionId functionId,std::size_t sourceIndex) {
    return {(static_cast<std::uint64_t>(functionId)<<32) | static_cast<std::uint64_t>(sourceIndex+1)};
}

void hashByte(std::uint64_t& hash,std::uint8_t byte) {
    hash^=byte;
    hash*=1099511628211ULL;
}

template<class T> void hashInteger(std::uint64_t& hash,T value) {
    using U=std::make_unsigned_t<T>;
    auto bits=static_cast<U>(value);
    for (std::size_t k=0;k<sizeof(U);++k) hashByte(hash,static_cast<std::uint8_t>(bits>>(k*8)));
}

void hashString(std::uint64_t& hash,const std::string& value) {
    hashInteger(hash,value.size());
    for (unsigned char c:value) hashByte(hash,c);
}

std::uint64_t signature(const TrainingPlan& plan) {
    std::uint64_t hash=1469598103934665603ULL;
    hashString(hash,semantic::dump(plan.source));
    hashInteger(hash,plan.function);
    hashInteger(hash,static_cast<std::uint8_t>(plan.optimizer.kind));
    hashInteger(hash,std::bit_cast<std::uint32_t>(plan.optimizer.learningRate));
    hashInteger(hash,std::bit_cast<std::uint32_t>(plan.optimizer.momentum));
    for (const auto& parameter:plan.parameters) {
        hashInteger(hash,parameter.id.value);
        hashInteger(hash,parameter.sourceParameterIndex);
        hashInteger(hash,parameter.position);
        hashString(hash,parameter.sourceParameterName);
        hashString(hash,typeName(parameter.type));
    }
    return hash;
}

std::vector<TrainingDiagnostic> verifySpec(const OptimizerSpec& spec) {
    std::vector<TrainingDiagnostic> errors;
    if (spec.kind!=OptimizerKind::SGD && spec.kind!=OptimizerKind::SGDMomentum)
        add(errors,"OPT01","unknown optimizer kind");
    if (std::isnan(spec.learningRate)) add(errors,"OPT02","learning rate is NaN");
    else if (std::isinf(spec.learningRate)) add(errors,"OPT03","learning rate is infinite");
    else if (spec.learningRate<0.0f) add(errors,"OPT04","learning rate is negative");
    if (spec.kind==OptimizerKind::SGDMomentum) {
        if (std::isnan(spec.momentum) || spec.momentum<0.0f)
            add(errors,"OPT05","momentum must be finite and at least zero");
        else if (std::isinf(spec.momentum) || spec.momentum>=1.0f)
            add(errors,"OPT06","momentum must be finite and less than one");
    }
    return errors;
}

bool validShape(const std::vector<std::int64_t>& shape,std::size_t& elements) {
    elements=1;
    for (auto extent:shape) {
        if (extent<0) return false;
        auto size=static_cast<std::size_t>(extent);
        if (size && elements>std::numeric_limits<std::size_t>::max()/size) return false;
        elements*=size;
    }
    return true;
}

enum class RuntimeMismatch { None, Type, Shape };
RuntimeMismatch runtimeMismatch(const RuntimeValue& value,const Type& type,
                                const std::vector<std::int64_t>* requiredShape=nullptr) {
    if (type==scalar(TypeKind::F32))
        return !std::holds_alternative<float>(value.data) ? RuntimeMismatch::Type :
            requiredShape && !requiredShape->empty() ? RuntimeMismatch::Shape : RuntimeMismatch::None;
    if (type.kind!=TypeKind::Tensor || type.elements.size()!=1 || type.elements[0]!=scalar(TypeKind::F32))
        return RuntimeMismatch::Type;
    const auto* tensor=std::get_if<RuntimeTensor>(&value.data);
    if (!tensor || tensor->dtype!=TypeKind::F32) return RuntimeMismatch::Type;
    if (tensor->shape.size()!=type.rank) return RuntimeMismatch::Shape;
    if (requiredShape && tensor->shape!=*requiredShape) return RuntimeMismatch::Shape;
    std::size_t elements=0;
    if (!validShape(tensor->shape,elements) || tensor->f32Values.size()!=elements || !tensor->values.empty())
        return RuntimeMismatch::Shape;
    return RuntimeMismatch::None;
}

std::vector<std::int64_t> runtimeShape(const RuntimeValue& value) {
    if (const auto* tensor=std::get_if<RuntimeTensor>(&value.data)) return tensor->shape;
    return {};
}

bool matchesKnownShape(const std::vector<std::int64_t>& concrete,const ShapeFact& expected) {
    if (concrete.size()!=expected.extents.size()) return false;
    for (std::size_t k=0;k<concrete.size();++k)
        if (expected.extents[k] && concrete[k]!=*expected.extents[k]) return false;
    return true;
}

float multiply(float a,float b) { volatile float x=a*b; return x; }
float plus(float a,float b) { volatile float x=a+b; return x; }
float minus(float a,float b) { volatile float x=a-b; return x; }

RuntimeValue zeroLike(const RuntimeValue& reference) {
    if (std::holds_alternative<float>(reference.data)) return RuntimeValue{0.0f};
    const auto& source=std::get<RuntimeTensor>(reference.data);
    return RuntimeValue{RuntimeTensor{TypeKind::F32,source.shape,{},std::vector<float>(source.f32Values.size(),0.0f)}};
}

RuntimeValue scale(const RuntimeValue& value,float factor) {
    if (const auto* scalar=std::get_if<float>(&value.data)) return RuntimeValue{multiply(factor,*scalar)};
    auto result=std::get<RuntimeTensor>(value.data);
    for (auto& element:result.f32Values) element=multiply(factor,element);
    return RuntimeValue{std::move(result)};
}

RuntimeValue addValues(const RuntimeValue& left,const RuntimeValue& right) {
    if (const auto* scalar=std::get_if<float>(&left.data))
        return RuntimeValue{plus(*scalar,std::get<float>(right.data))};
    auto result=std::get<RuntimeTensor>(left.data);
    const auto& other=std::get<RuntimeTensor>(right.data);
    for (std::size_t k=0;k<result.f32Values.size();++k)
        result.f32Values[k]=plus(result.f32Values[k],other.f32Values[k]);
    return RuntimeValue{std::move(result)};
}

RuntimeValue subtractValues(const RuntimeValue& left,const RuntimeValue& right) {
    if (const auto* scalar=std::get_if<float>(&left.data))
        return RuntimeValue{minus(*scalar,std::get<float>(right.data))};
    auto result=std::get<RuntimeTensor>(left.data);
    const auto& other=std::get<RuntimeTensor>(right.data);
    for (std::size_t k=0;k<result.f32Values.size();++k)
        result.f32Values[k]=minus(result.f32Values[k],other.f32Values[k]);
    return RuntimeValue{std::move(result)};
}

std::vector<TrainingDiagnostic> verifyState(const TrainingPlan& plan,const TrainingState& state) {
    std::vector<TrainingDiagnostic> errors;
    if (state.planSignature!=plan.signature) add(errors,"TRN11","state belongs to an incompatible training plan");
    if (state.parameters.size()!=plan.parameters.size()) {
        add(errors,"TRN07","parameter count disagrees with training plan");
        return errors;
    }
    for (std::size_t k=0;k<plan.parameters.size();++k) {
        const auto& descriptor=plan.parameters[k];
        const auto& parameter=state.parameters[k];
        if (parameter.id!=descriptor.id) add(errors,"TRN14","parameter identity/order disagrees with training plan");
        if (parameter.type!=descriptor.type) add(errors,"TRN08","parameter dtype/type disagrees with training plan");
        auto mismatch=runtimeMismatch(parameter.value,descriptor.type,&parameter.shape);
        if (mismatch==RuntimeMismatch::Type) add(errors,"TRN08","parameter runtime dtype/type mismatch");
        if (mismatch==RuntimeMismatch::Shape || !matchesKnownShape(parameter.shape,descriptor.expectedShape))
            add(errors,"TRN09","parameter rank/shape mismatch");
    }
    auto optimizer=verifyOptimizerState(plan,state);
    errors.insert(errors.end(),optimizer.begin(),optimizer.end());
    return errors;
}

std::vector<RuntimeValue> unpackGradients(const RuntimeValue& value,std::size_t count) {
    if (count==1) return {value};
    if (const auto* tuple=std::get_if<RuntimeTuple>(&value.data); tuple && tuple->size()==count) return *tuple;
    return {};
}
} // namespace

PlanResult createTrainingPlan(const semantic::Module& source,const TrainingPlanRequest& request) {
    PlanResult result;
    auto verified=semantic::verify(source);
    if (!verified.ok) { add(result.diagnostics,"TRN-SEMANTIC","semantic verification failed"); return result; }
    const auto* fn=function(source,request.function);
    if (!fn) { add(result.diagnostics,"TRN01","unknown loss function"); return result; }
    if (fn->result!=scalar(TypeKind::F32)) add(result.diagnostics,"TRN04","scalar training requires an f32 scalar loss");
    auto specErrors=verifySpec(request.optimizer);
    result.diagnostics.insert(result.diagnostics.end(),specErrors.begin(),specErrors.end());
    std::set<std::size_t> seen;
    for (auto index:request.trainableParameters) {
        if (index>=fn->parameters.size()) add(result.diagnostics,"TRN01","trainable parameter index is out of range");
        else if (!seen.insert(index).second) add(result.diagnostics,"TRN02","duplicate trainable parameter index");
        else if (!differentiableType(fn->parameters[index].type))
            add(result.diagnostics,"TRN03","trainable parameter is not differentiable in TH-011");
        else if (fn->parameters[index].access!=AccessMode::Read)
            add(result.diagnostics,"TRN-ACCESS","trainable parameter must use Read access");
    }
    if (request.trainableParameters.empty()) add(result.diagnostics,"TRN03","at least one explicit trainable parameter is required");
    auto ownership=analysis::analyze(source);
    if (!ownership.ok() || !analysis::auditFacts(source,ownership).empty())
        add(result.diagnostics,"TRN-OWNERSHIP","TH-006 ownership/effect qualification failed");
    if (!result.diagnostics.empty()) return result;

    TrainingPlan plan;
    plan.source=source;
    plan.ownership=std::move(ownership);
    plan.function=fn->id;
    plan.optimizer=request.optimizer;
    plan.reverseRequest={fn->id,request.trainableParameters};
    for (std::size_t position=0;position<request.trainableParameters.size();++position) {
        auto index=request.trainableParameters[position];
        const auto& parameter=fn->parameters[index];
        plan.parameters.push_back({parameterId(fn->id,index),index,parameter.name,parameter.type,parameter.shape,position});
    }
    for (std::size_t index=0;index<fn->parameters.size();++index)
        if (!seen.contains(index)) plan.runtimeParameterIndices.push_back(index);
    plan.differentiated=autodiff::differentiate(plan.source,plan.ownership,plan.reverseRequest);
    plan.generationCount=1;
    if (!plan.differentiated.ok()) {
        auto code=plan.differentiated.diagnostics.empty()?"TRN15":plan.differentiated.diagnostics.front().code;
        add(result.diagnostics,code,"loss function is not eligible for TH-010 reverse mode");
        return result;
    }
    plan.generatedOwnership=analysis::analyze(*plan.differentiated.module);
    if (!plan.generatedOwnership.ok() ||
        !analysis::auditFacts(*plan.differentiated.module,plan.generatedOwnership).empty()) {
        add(result.diagnostics,"TRN15","generated AD functions failed ownership/effect qualification");
        return result;
    }
    plan.signature=signature(plan);
    auto planErrors=verifyTrainingPlan(plan);
    if (!planErrors.empty()) { result.diagnostics=std::move(planErrors); return result; }
    result.plan=std::move(plan);
    return result;
}

std::vector<TrainingDiagnostic> verifyTrainingPlan(const TrainingPlan& plan) {
    std::vector<TrainingDiagnostic> errors;
    auto verified=semantic::verify(plan.source);
    if (!verified.ok) add(errors,"TRN15","training plan source IR is not verifier-valid");
    if (!plan.ownership.ok() || !analysis::auditFacts(plan.source,plan.ownership).empty())
        add(errors,"TRN15","training plan ownership/effect metadata is missing or corrupted");
    const auto* fn=function(plan.source,plan.function);
    if (!fn) { add(errors,"TRN15","training plan function is missing"); return errors; }
    if (fn->result!=scalar(TypeKind::F32)) add(errors,"TRN04","training plan loss is not scalar f32");
    auto specErrors=verifySpec(plan.optimizer); errors.insert(errors.end(),specErrors.begin(),specErrors.end());
    if (plan.generationCount!=1 || plan.reverseRequest.function!=plan.function ||
        plan.reverseRequest.wrtParameters.size()!=plan.parameters.size())
        add(errors,"TRN15","training plan AD generation metadata is missing or corrupted");
    std::set<std::size_t> indices; std::set<std::uint64_t> ids;
    for (std::size_t k=0;k<plan.parameters.size();++k) {
        const auto& descriptor=plan.parameters[k];
        if (descriptor.sourceParameterIndex>=fn->parameters.size()) add(errors,"TRN01","unknown trainable index in plan");
        else {
            const auto& sourceParameter=fn->parameters[descriptor.sourceParameterIndex];
            if (!differentiableType(sourceParameter.type)) add(errors,"TRN03","non-differentiable trainable parameter in plan");
            if (sourceParameter.access!=AccessMode::Read) add(errors,"TRN-ACCESS","incompatible trainable access mode");
            if (descriptor.type!=sourceParameter.type || descriptor.expectedShape!=sourceParameter.shape ||
                descriptor.sourceParameterName!=sourceParameter.name) add(errors,"TRN15","parameter descriptor metadata is corrupted");
        }
        if (!indices.insert(descriptor.sourceParameterIndex).second) add(errors,"TRN02","duplicate trainable index in plan");
        if (!ids.insert(descriptor.id.value).second) add(errors,"TRN-DUPLICATE-PARAMETER-ID","duplicate ParameterId");
        if (descriptor.position!=k || descriptor.id!=parameterId(plan.function,descriptor.sourceParameterIndex))
            add(errors,"TRN14","parameter identity/order metadata is corrupted");
        if (k>=plan.reverseRequest.wrtParameters.size() ||
            plan.reverseRequest.wrtParameters[k]!=descriptor.sourceParameterIndex)
            add(errors,"TRN15","reverse request order disagrees with parameter order");
    }
    std::vector<std::size_t> expectedRuntime;
    for (std::size_t k=0;k<fn->parameters.size();++k) if (!indices.contains(k)) expectedRuntime.push_back(k);
    if (plan.runtimeParameterIndices!=expectedRuntime) add(errors,"TRN15","runtime parameter binding metadata is corrupted");
    if (!plan.differentiated.ok() || !autodiff::verify(plan.source,plan.differentiated).empty())
        add(errors,"TRN15","plan AD metadata is missing or corrupted");
    else if (!plan.generatedOwnership.ok() ||
        !analysis::auditFacts(*plan.differentiated.module,plan.generatedOwnership).empty())
        add(errors,"TRN15","generated ownership/effect metadata is missing or corrupted");
    if (plan.signature!=signature(plan)) add(errors,"TRN15","training plan signature is corrupted");
    return errors;
}

StateResult initializeTrainingState(const TrainingPlan& plan,const std::vector<RuntimeValue>& orderedParameters) {
    StateResult result;
    auto planErrors=verifyTrainingPlan(plan);
    if (!planErrors.empty()) { result.diagnostics=std::move(planErrors); return result; }
    if (orderedParameters.size()!=plan.parameters.size()) {
        add(result.diagnostics,"TRN07","initial parameter count disagrees with plan"); return result;
    }
    TrainingState state; state.planSignature=plan.signature; state.optimizer.kind=plan.optimizer.kind;
    for (std::size_t k=0;k<orderedParameters.size();++k) {
        const auto& descriptor=plan.parameters[k];
        auto mismatch=runtimeMismatch(orderedParameters[k],descriptor.type);
        if (mismatch==RuntimeMismatch::Type) add(result.diagnostics,"TRN08","initial parameter dtype/type mismatch");
        else if (mismatch==RuntimeMismatch::Shape || !matchesKnownShape(runtimeShape(orderedParameters[k]),descriptor.expectedShape))
            add(result.diagnostics,"TRN09","initial parameter rank/shape mismatch");
        if (!result.diagnostics.empty()) return result;
        auto shape=runtimeShape(orderedParameters[k]);
        state.parameters.push_back({descriptor.id,descriptor.type,shape,orderedParameters[k]});
        if (plan.optimizer.kind==OptimizerKind::SGDMomentum)
            state.optimizer.velocities.push_back({descriptor.id,descriptor.type,shape,zeroLike(orderedParameters[k])});
    }
    result.state=std::move(state);
    return result;
}

std::vector<TrainingDiagnostic> verifyOptimizerState(const TrainingPlan& plan,const TrainingState& state) {
    auto errors=verifySpec(plan.optimizer);
    if (state.optimizer.kind!=OptimizerKind::SGD && state.optimizer.kind!=OptimizerKind::SGDMomentum)
        add(errors,"OPT01","optimizer state contains an unknown kind");
    if (state.optimizer.kind!=plan.optimizer.kind || state.optimizer.step!=state.step)
        add(errors,"OPT10","optimizer kind/step is incompatible with plan or training state");
    const auto expected=plan.optimizer.kind==OptimizerKind::SGDMomentum?plan.parameters.size():0U;
    if (state.optimizer.velocities.size()!=expected) {
        add(errors,"OPT07","momentum velocity count mismatch");
        return errors;
    }
    for (std::size_t k=0;k<state.optimizer.velocities.size();++k) {
        const auto& velocity=state.optimizer.velocities[k];
        if (k>=state.parameters.size() || velocity.id!=plan.parameters[k].id || velocity.id!=state.parameters[k].id)
            add(errors,"OPT10","velocity identity/order is incompatible with parameter state");
        if (velocity.type!=plan.parameters[k].type || runtimeMismatch(velocity.value,velocity.type)==RuntimeMismatch::Type)
            add(errors,"OPT08","velocity dtype/type mismatch");
        if (k<state.parameters.size() && (velocity.shape!=state.parameters[k].shape ||
            runtimeMismatch(velocity.value,velocity.type,&velocity.shape)==RuntimeMismatch::Shape))
            add(errors,"OPT09","velocity shape mismatch");
    }
    return errors;
}

OptimizerResult applyOptimizer(const TrainingPlan& plan,const TrainingState& state,
                               const std::vector<GradientValue>& orderedGradients) {
    OptimizerResult result;
    auto stateErrors=verifyState(plan,state);
    if (!stateErrors.empty()) { result.diagnostics=std::move(stateErrors); return result; }
    if (orderedGradients.size()!=plan.parameters.size()) {
        add(result.diagnostics,"TRN10","gradient count mismatch before optimizer update"); return result;
    }
    for (std::size_t k=0;k<orderedGradients.size();++k) {
        const auto& gradient=orderedGradients[k];
        const auto& parameter=state.parameters[k];
        if (gradient.id!=plan.parameters[k].id) add(result.diagnostics,"TRN10","gradient ParameterId/order mismatch");
        if (gradient.type!=parameter.type || runtimeMismatch(gradient.value,gradient.type)==RuntimeMismatch::Type)
            add(result.diagnostics,"TRN10","gradient dtype/type mismatch before optimizer update");
        if (gradient.shape!=parameter.shape || runtimeMismatch(gradient.value,gradient.type,&gradient.shape)==RuntimeMismatch::Shape)
            add(result.diagnostics,"TRN10","gradient rank/shape mismatch before optimizer update");
    }
    if (!result.diagnostics.empty()) return result;

    TrainingState next=state;
    std::vector<ParameterValue> parameters; parameters.reserve(state.parameters.size());
    std::vector<VelocityValue> velocities; velocities.reserve(state.parameters.size());
    for (std::size_t k=0;k<state.parameters.size();++k) {
        RuntimeValue direction=orderedGradients[k].value;
        if (plan.optimizer.kind==OptimizerKind::SGDMomentum) {
            direction=addValues(scale(state.optimizer.velocities[k].value,plan.optimizer.momentum),direction);
            velocities.push_back({state.parameters[k].id,state.parameters[k].type,state.parameters[k].shape,direction});
        }
        auto updated=subtractValues(state.parameters[k].value,scale(direction,plan.optimizer.learningRate));
        parameters.push_back({state.parameters[k].id,state.parameters[k].type,state.parameters[k].shape,std::move(updated)});
    }
    next.parameters=std::move(parameters);
    next.optimizer.velocities=std::move(velocities);
    ++next.step;
    next.optimizer.step=next.step;
    result.state=std::move(next);
    return result;
}

TrainingStepResult trainingStep(const TrainingPlan& plan,const TrainingState& state,
                                const std::vector<RuntimeInput>& runtimeInputs) {
    TrainingStepResult result;
    auto planErrors=verifyTrainingPlan(plan);
    if (!planErrors.empty()) { result.diagnostics=std::move(planErrors); return result; }
    auto stateErrors=verifyState(plan,state);
    if (!stateErrors.empty()) { result.diagnostics=std::move(stateErrors); return result; }
    const auto* fn=function(plan.source,plan.function);
    std::vector<std::optional<RuntimeValue>> arguments(fn->parameters.size());
    for (std::size_t k=0;k<plan.parameters.size();++k)
        arguments[plan.parameters[k].sourceParameterIndex]=state.parameters[k].value;
    std::set<std::size_t> supplied;
    for (const auto& input:runtimeInputs) {
        if (input.sourceParameterIndex>=fn->parameters.size()) {
            add(result.diagnostics,"TRN05","runtime input index is out of range"); continue;
        }
        if (!supplied.insert(input.sourceParameterIndex).second) {
            add(result.diagnostics,"TRN05","duplicate runtime input"); continue;
        }
        bool trainable=false; for (const auto& parameter:plan.parameters)
            trainable|=parameter.sourceParameterIndex==input.sourceParameterIndex;
        if (trainable) { add(result.diagnostics,"TRN06","runtime input supplied for trainable slot"); continue; }
        auto mismatch=runtimeMismatch(input.value,fn->parameters[input.sourceParameterIndex].type);
        if (mismatch==RuntimeMismatch::Type) add(result.diagnostics,"TRN08","runtime input dtype/type mismatch");
        else if (mismatch==RuntimeMismatch::Shape ||
            !matchesKnownShape(runtimeShape(input.value),fn->parameters[input.sourceParameterIndex].shape))
            add(result.diagnostics,"TRN09","runtime input rank/shape mismatch");
        else arguments[input.sourceParameterIndex]=input.value;
    }
    for (auto index:plan.runtimeParameterIndices) if (!arguments[index])
        add(result.diagnostics,"TRN05","missing non-trainable runtime input");
    if (!result.diagnostics.empty()) return result;
    std::vector<RuntimeValue> fullArguments; fullArguments.reserve(arguments.size());
    for (auto& argument:arguments) fullArguments.push_back(std::move(*argument));

    auto execution=autodiff::executeVjpWithPrimal(plan.differentiated,fullArguments,RuntimeValue{1.0f});
    if (!execution.ok()) { add(result.diagnostics,execution.errorId,"forward/backward reference execution failed"); return result; }
    const auto* loss=std::get_if<float>(&execution.primal->data);
    if (!loss) { add(result.diagnostics,"TRN04","training execution did not produce scalar f32 loss"); return result; }
    auto rawGradients=unpackGradients(*execution.gradients,plan.parameters.size());
    if (rawGradients.size()!=plan.parameters.size()) {
        add(result.diagnostics,"TRN10","generated backward returned wrong gradient count"); return result;
    }
    for (std::size_t k=0;k<rawGradients.size();++k)
        result.gradients.push_back({plan.parameters[k].id,plan.parameters[k].type,runtimeShape(rawGradients[k]),rawGradients[k]});
    auto optimized=applyOptimizer(plan,state,result.gradients);
    if (!optimized.ok()) { result.gradients.clear(); result.diagnostics=std::move(optimized.diagnostics); return result; }
    result.loss=*loss;
    result.nextState=std::move(optimized.state);
    return result;
}

std::string dumpTrainingStep(const TrainingPlan& plan,const TrainingStepResult& result) {
    std::ostringstream out;
    out << "reference-training step=";
    if (result.nextState) out << result.nextState->step-1; else out << "failed";
    out << " loss=";
    if (result.loss) out << std::setprecision(std::numeric_limits<float>::max_digits10) << *result.loss;
    else out << "unavailable";
    out << '\n';
    for (std::size_t k=0;k<plan.parameters.size();++k) {
        const auto& p=plan.parameters[k];
        out << "parameter[" << k << "] id=" << p.id.value << " type=" << typeName(p.type) << " shape=[";
        if (result.nextState && k<result.nextState->parameters.size()) {
            const auto& shape=result.nextState->parameters[k].shape;
            for (std::size_t d=0;d<shape.size();++d) { if (d) out << ','; out << shape[d]; }
        }
        out << "] gradient=";
        if (k<result.gradients.size()) {
            out << typeName(result.gradients[k].type) << " shape=[";
            for (std::size_t d=0;d<result.gradients[k].shape.size();++d) {
                if (d) out << ','; out << result.gradients[k].shape[d];
            }
            out << ']';
        } else out << "unavailable";
        out << '\n';
    }
    return out.str();
}

} // namespace thiran::v0::training
