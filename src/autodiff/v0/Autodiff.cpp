#include "autodiff/v0/Autodiff.hpp"
#include "semantic/v0/Verifier.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace thiran::v0::autodiff {
using namespace semantic;
namespace {
void diagnostic(DifferentiationResult& r,std::string code,std::string message) {
    r.diagnostics.push_back({std::move(code),std::move(message)});
}
const Function* function(const semantic::Module& m,FunctionId id) {
    for (const auto& f:m.functions) if (f.id==id) return &f;
    return nullptr;
}
bool tensor(const Type& t) { return t.kind==TypeKind::Tensor; }
Type dtype(const Type& t) { return tensor(t)?t.elements.at(0):t; }
struct PrimalFact { Type type; ShapeFact shape; SourceSpan span; };
std::map<ValueId,PrimalFact> facts(const Function& f) {
    std::map<ValueId,PrimalFact> r;
    for (const auto& p:f.parameters) r[p.id]={p.type,p.shape,p.span};
    for (const auto& step:f.body.steps) if (const auto* i=std::get_if<Instruction>(&step)) r[i->id]={i->type,i->shape,i->span};
    return r;
}
std::optional<ValueId> returned(const Function& f) {
    for (const auto& step:f.body.steps) if (const auto* flow=std::get_if<Flow>(&step);
        flow && flow->kind==Flow::Kind::Return) return flow->value;
    return f.body.returned;
}
bool eligibleOp(Op op) {
    switch (op) {
        case Op::Integer: case Op::Float: case Op::Boolean: case Op::LoadBinding:
        case Op::Negate: case Op::Add: case Op::Subtract: case Op::Multiply:
        case Op::ElementMultiply: case Op::Matmul: case Op::Transpose: case Op::Sum:
        case Op::StopGradient: return true;
        default: return false;
    }
}
std::string opName(Op op) {
    switch (op) {
        case Op::Copy:return "copy"; case Op::Move:return "move"; case Op::MutableBorrow:return "mutable borrow";
        case Op::Index:return "index"; case Op::Slice:return "slice"; case Op::Call:return "ordinary call";
        case Op::Tuple:return "tuple"; case Op::TensorLiteral:return "tensor literal";
        default:return "unsupported semantic operation";
    }
}
struct Builder {
    Function fn;
    ValueId next=1;
    BindingId binding=1;
    SourceSpan span;
    ValueId emit(Op op,Type type,ShapeFact shape,std::vector<ValueId> operands={},std::uint32_t axis=0,
                 EffectClass effect=EffectClass::Pure,bool view=false) {
        Instruction i; i.id=next++; i.op=op; i.type=std::move(type); i.shape=std::move(shape); i.span=span;
        i.operands=std::move(operands); i.axis=axis; i.effect=effect; i.borrowedView=view;
        fn.body.steps.emplace_back(std::move(i)); return next-1;
    }
    ValueId parameter(std::string name,const Type& type,const ShapeFact& shape) {
        auto id=next++,b=binding++;
        fn.parameters.push_back({id,std::move(name),type,shape,span,b,AccessMode::Read});
        return id;
    }
};
}

DifferentiationResult differentiate(const semantic::Module& source,const analysis::OwnershipAnalysisResult& ownership,
                                    const ReverseModeRequest& request) {
    DifferentiationResult result; result.sourceFunction=request.function; result.wrtParameters=request.wrtParameters;
    auto sourceVerification=semantic::verify(source);
    if (!sourceVerification.ok) { diagnostic(result,"AD-ELIGIBILITY-IR","source semantic IR is not verifier-valid"); return result; }
    if (!ownership.ok() || !analysis::auditFacts(source,ownership).empty()) {
        diagnostic(result,"AD-ELIGIBILITY-OWNERSHIP","TH-006 ownership/effect facts are not valid"); return result;
    }
    const auto* primal=function(source,request.function);
    if (!primal) { diagnostic(result,"ADV01","requested function does not exist"); return result; }
    if (request.wrtParameters.empty()) { diagnostic(result,"ADV10","at least one WRT parameter is required"); return result; }
    std::set<std::size_t> seenWrt;
    for (auto index:request.wrtParameters) {
        if (index>=primal->parameters.size() || !seenWrt.insert(index).second) {
            diagnostic(result,"ADV01","WRT parameter does not uniquely belong to the requested function"); continue;
        }
        if (!differentiableType(primal->parameters[index].type))
            diagnostic(result,"ADV02","WRT parameter is not differentiable in TH-010");
        if (primal->parameters[index].access!=AccessMode::Read)
            diagnostic(result,"AD-ELIGIBILITY-ACCESS","MutableBorrow/Consume parameters are not differentiable in TH-010");
    }
    if (!differentiableType(primal->result)) diagnostic(result,"AD-ELIGIBILITY-RESULT","function result is not a supported differentiable type");
    if (!source.imports.empty()) diagnostic(result,"AD-ELIGIBILITY-IMPORT","import-dependent differentiation is deferred");
    auto effect=ownership.functionEffects.find(primal->id);
    if (effect==ownership.functionEffects.end()) diagnostic(result,"AD-ELIGIBILITY-EFFECT","TH-006 effect summary is missing");
    else {
        constexpr auto rejected=static_cast<analysis::EffectSet>(analysis::EffectKind::Mutates) |
            static_cast<analysis::EffectSet>(analysis::EffectKind::RNG) | static_cast<analysis::EffectSet>(analysis::EffectKind::IO) |
            static_cast<analysis::EffectSet>(analysis::EffectKind::Transfer) | static_cast<analysis::EffectSet>(analysis::EffectKind::Async);
        if (effect->second.kinds & rejected) diagnostic(result,"AD-ELIGIBILITY-EFFECT","function has a rejected mutation/RNG/IO/Transfer/Async effect");
    }
    for (const auto& step:primal->body.steps) {
        if (std::holds_alternative<Structured>(step)) diagnostic(result,"AD-ELIGIBILITY-CONTROL","structured control-flow AD is deferred");
        else if (const auto* w=std::get_if<BindingWrite>(&step); w && !w->declaration)
            diagnostic(result,"AD-ELIGIBILITY-MUTATION","rebinding/mutation AD is deferred");
        else if (const auto* i=std::get_if<Instruction>(&step); i && !eligibleOp(i->op))
            diagnostic(result,"AD-ELIGIBILITY-OP",opName(i->op)+" differentiation is deferred");
    }
    if (!result.diagnostics.empty()) return result;
    const auto pf=facts(*primal);
    const auto out=returned(*primal);
    if (!out || !pf.contains(*out)) { diagnostic(result,"ADV03","source result value is unavailable"); return result; }

    std::map<ValueId,std::set<std::string>> reasons;
    for (auto index:request.wrtParameters) reasons[primal->parameters[index].id].insert("requested input / ZeroLike reference");
    for (const auto& step:primal->body.steps) if (const auto* i=std::get_if<Instruction>(&step)) {
        auto save=[&](ValueId id,const char* why) { if (pf.contains(id) && differentiableType(pf.at(id).type)) reasons[id].insert(why); };
        if ((i->op==Op::Multiply || i->op==Op::ElementMultiply || i->op==Op::Matmul) && i->operands.size()==2) {
            save(i->operands[0],i->op==Op::Matmul?"Matmul derivative operand":"Multiply derivative operand");
            save(i->operands[1],i->op==Op::Matmul?"Matmul derivative operand":"Multiply derivative operand");
        }
        if ((i->op==Op::Add || i->op==Op::Subtract) && i->operands.size()==2 &&
            (tensor(pf.at(i->operands[0]).type) || tensor(pf.at(i->operands[1]).type))) {
            save(i->operands[0],"broadcast reduction shape reference"); save(i->operands[1],"broadcast reduction shape reference");
        }
        if (i->op==Op::Sum && !i->operands.empty()) save(i->operands[0],"Sum broadcast shape reference");
    }
    auto provenance=ownership.valueProvenance.find(primal->id);
    for (const auto& [id,why]:reasons) {
        SavedValue saved; saved.slot=result.saves.size(); saved.primal=id; saved.type=pf.at(id).type; saved.shape=pf.at(id).shape;
        for (const auto& part:why) { if (!saved.reason.empty()) saved.reason += "; "; saved.reason += part; }
        if (provenance!=ownership.valueProvenance.end()) if (auto it=provenance->second.find(id);it!=provenance->second.end()) {
            saved.provenance=it->second.kind; saved.resources=it->second.resources;
        }
        result.saves.push_back(std::move(saved));
    }

    Function forward=*primal; forward.id=1; forward.name=primal->name+".__th010_forward";
    forward.generated=true; forward.sourceFunction=primal->id; forward.generatedRole="forward";
    ValueId next=1; for (const auto& p:forward.parameters) next=std::max(next,static_cast<ValueId>(p.id+1));
    for (const auto& step:forward.body.steps) if (const auto* i=std::get_if<Instruction>(&step)) next=std::max(next,static_cast<ValueId>(i->id+1));
    Type forwardResult{TypeKind::Tuple,{primal->result},0};
    std::vector<ValueId> tupleOperands{*out};
    for (const auto& save:result.saves) { forwardResult.elements.push_back(save.type); tupleOperands.push_back(save.primal); }
    forward.result=forwardResult;
    for (std::size_t k=0;k<forward.body.steps.size();++k) if (auto* flow=std::get_if<Flow>(&forward.body.steps[k]); flow && flow->kind==Flow::Kind::Return) {
        const auto flowSpan=flow->span;
        Instruction tuple; tuple.id=next++; tuple.op=Op::Tuple; tuple.type=forwardResult; tuple.span=flowSpan; tuple.operands=tupleOperands;
        forward.body.steps[k]=tuple; forward.body.steps.insert(forward.body.steps.begin()+static_cast<std::ptrdiff_t>(k+1),
            Flow{Flow::Kind::Return,tuple.id,flowSpan}); forward.body.returned=tuple.id; break;
    }

    Builder b; b.span=primal->span; b.fn.id=2; b.fn.name=primal->name+".__th010_backward";
    b.fn.span=primal->span; b.fn.generated=true; b.fn.sourceFunction=primal->id; b.fn.generatedRole="backward";
    b.fn.body.id=1; b.fn.body.terminated=true;
    std::map<ValueId,ValueId> savedParam;
    for (const auto& save:result.saves) savedParam[save.primal]=b.parameter("save"+std::to_string(save.slot),save.type,save.shape);
    auto seed=b.parameter("output_cotangent",primal->result,pf.at(*out).shape);
    std::map<ValueId,ValueId> cotangent; cotangent[*out]=seed;
    std::map<BindingId,ValueId> bound;
    for (const auto& p:primal->parameters) bound[p.binding]=p.id;
    std::map<ValueId,ValueId> aliasSource;
    for (const auto& step:primal->body.steps) {
        if (const auto* i=std::get_if<Instruction>(&step); i && i->op==Op::LoadBinding && bound.contains(i->binding)) aliasSource[i->id]=bound[i->binding];
        if (const auto* w=std::get_if<BindingWrite>(&step)) bound[w->binding]=w->value;
    }
    auto emit=[&](Op op,const PrimalFact& fact,std::vector<ValueId> operands,EffectClass effect=EffectClass::Pure,bool view=false) {
        return b.emit(op,fact.type,fact.shape,std::move(operands),0,effect,view);
    };
    auto addContribution=[&](ValueId target,ValueId contribution) {
        if (!pf.contains(target) || !differentiableType(pf.at(target).type)) return;
        if (cotangent.contains(target)) cotangent[target]=emit(Op::Add,pf.at(target),{cotangent[target],contribution});
        else cotangent[target]=contribution;
    };
    for (auto it=primal->body.steps.rbegin();it!=primal->body.steps.rend();++it) {
        const auto* i=std::get_if<Instruction>(&*it); if (!i || !cotangent.contains(i->id)) continue;
        auto dz=cotangent.at(i->id);
        switch (i->op) {
            case Op::LoadBinding: if (aliasSource.contains(i->id)) addContribution(aliasSource.at(i->id),dz); break;
            case Op::Negate: addContribution(i->operands[0],emit(Op::Negate,pf.at(i->operands[0]),{dz})); break;
            case Op::Add: case Op::Subtract: {
                for (std::size_t k=0;k<2;++k) {
                    auto contribution=dz;
                    if (tensor(pf.at(i->operands[0]).type) || tensor(pf.at(i->operands[1]).type))
                        contribution=b.emit(Op::ReduceToShape,pf.at(i->operands[k]).type,pf.at(i->operands[k]).shape,
                            {dz,savedParam.at(i->operands[k])},0,EffectClass::CheckedFailure);
                    if (i->op==Op::Subtract && k==1) contribution=emit(Op::Negate,pf.at(i->operands[k]),{contribution});
                    addContribution(i->operands[k],contribution);
                }
                break;
            }
            case Op::Multiply: case Op::ElementMultiply: {
                for (std::size_t k=0;k<2;++k) {
                    auto other=i->operands[1-k];
                    PrimalFact product=pf.at(i->id);
                    auto contribution=b.emit(i->op,product.type,product.shape,{dz,savedParam.at(other)});
                    if (product.type!=pf.at(i->operands[k]).type || tensor(product.type))
                        contribution=b.emit(Op::ReduceToShape,pf.at(i->operands[k]).type,pf.at(i->operands[k]).shape,
                            {contribution,savedParam.at(i->operands[k])},0,EffectClass::CheckedFailure);
                    addContribution(i->operands[k],contribution);
                }
                break;
            }
            case Op::Matmul: {
                auto a=i->operands[0],c=i->operands[1];
                PrimalFact bt{pf.at(c).type,{{pf.at(c).shape.extents[1],pf.at(c).shape.extents[0]}},i->span};
                PrimalFact at{pf.at(a).type,{{pf.at(a).shape.extents[1],pf.at(a).shape.extents[0]}},i->span};
                auto btv=emit(Op::Transpose,bt,{savedParam.at(c)},EffectClass::Pure,true);
                auto atv=emit(Op::Transpose,at,{savedParam.at(a)},EffectClass::Pure,true);
                addContribution(a,b.emit(Op::Matmul,pf.at(a).type,pf.at(a).shape,{dz,btv},0,EffectClass::CheckedFailure));
                addContribution(c,b.emit(Op::Matmul,pf.at(c).type,pf.at(c).shape,{atv,dz},0,EffectClass::CheckedFailure));
                break;
            }
            case Op::Transpose: addContribution(i->operands[0],emit(Op::Transpose,pf.at(i->operands[0]),{dz},EffectClass::Pure,true)); break;
            case Op::Sum: addContribution(i->operands[0],b.emit(Op::BroadcastToShape,pf.at(i->operands[0]).type,pf.at(i->operands[0]).shape,
                {dz,savedParam.at(i->operands[0])},i->axis,EffectClass::CheckedFailure)); break;
            case Op::StopGradient: break;
            default: break;
        }
    }
    std::vector<ValueId> outputs; Type backwardResult;
    for (auto index:request.wrtParameters) {
        auto id=primal->parameters[index].id;
        ValueId gradient=0;
        if (cotangent.contains(id)) gradient=cotangent.at(id);
        else gradient=b.emit(Op::ZeroLike,pf.at(id).type,pf.at(id).shape,{savedParam.at(id)});
        outputs.push_back(gradient); result.gradients.push_back({index,pf.at(id).type});
    }
    if (outputs.size()==1) backwardResult=pf.at(primal->parameters[request.wrtParameters[0]].id).type;
    else {
        backwardResult={TypeKind::Tuple,{},0}; for (auto index:request.wrtParameters) backwardResult.elements.push_back(primal->parameters[index].type);
        PrimalFact tupleFact{backwardResult,{},primal->span}; outputs={emit(Op::Tuple,tupleFact,outputs)};
    }
    b.fn.result=backwardResult; b.fn.body.steps.emplace_back(Flow{Flow::Kind::Return,outputs[0],primal->span});
    semantic::Module generated; generated.source=source.source; generated.span=source.span; generated.functions={std::move(forward),std::move(b.fn)};
    generated.initializer.id=1; generated.initializer.terminated=true;
    result.module=std::move(generated);
    auto audit=verify(source,result); if (!audit.empty()) { result.diagnostics=std::move(audit); result.module.reset(); }
    return result;
}

std::vector<AdDiagnostic> verify(const semantic::Module& source,const DifferentiationResult& result) {
    std::vector<AdDiagnostic> errors;
    auto add=[&](std::string c,std::string m){ errors.push_back({std::move(c),std::move(m)}); };
    const auto* primal=function(source,result.sourceFunction);
    if (!primal) { add("ADV01","source function missing"); return errors; }
    for (auto index:result.wrtParameters) {
        if (index>=primal->parameters.size()) add("ADV01","WRT parameter does not belong to source function");
        else if (!differentiableType(primal->parameters[index].type)) add("ADV02","non-differentiable WRT parameter");
    }
    if (!result.module) { add("ADV06","generated module missing"); return errors; }
    auto ordinary=semantic::verify(*result.module);
    if (!ordinary.ok) for (const auto& e:ordinary.errors) add("ADV07",e);
    const auto pf=facts(*primal);
    std::set<ValueId> savedIds;
    for (std::size_t k=0;k<result.saves.size();++k) {
        const auto& save=result.saves[k];
        if (!pf.contains(save.primal)) add("ADV04","save references unknown primal ValueId");
        if (save.slot!=k || !savedIds.insert(save.primal).second) add("ADV05","duplicate or missing saved slot identity");
        if (save.implicitCopy) add("ADV05","saved value must not be an implicit deep copy");
    }
    const auto* backward=function(*result.module,result.backwardFunction);
    if (!backward || backward->parameters.size()!=result.saves.size()+1) add("ADV06","backward has unavailable/missing save parameters");
    else for (std::size_t k=0;k<result.saves.size();++k)
        if (backward->parameters[k].type!=result.saves[k].type) add("ADV06","backward save type mismatch");
    if (backward) {
        std::map<ValueId,Type> types;
        for (const auto& p:backward->parameters) types[p.id]=p.type;
        for (const auto& step:backward->body.steps) if (const auto* i=std::get_if<Instruction>(&step)) {
            auto operand=[&](std::size_t k)->Type { return k<i->operands.size() && types.contains(i->operands[k])?types.at(i->operands[k]):Type{}; };
            if (i->op==Op::ReduceToShape) {
                auto value=operand(0),target=operand(1);
                if (i->operands.size()!=2 || !differentiableType(value) || !differentiableType(target) || i->type!=target ||
                    dtype(value)!=dtype(target) || (tensor(value)&&tensor(target)&&value.rank<target.rank))
                    add("ADV08","ReduceToShape target type/rank mismatch");
            }
            if (i->op==Op::BroadcastToShape) {
                auto value=operand(0),target=operand(1);
                if (i->operands.size()!=2 || !differentiableType(value) || !differentiableType(target) || i->type!=target ||
                    dtype(value)!=dtype(target) || !tensor(target) || i->axis>=target.rank ||
                    ((!tensor(value) && target.rank!=1) || (tensor(value) && value.rank+1!=target.rank)))
                    add("ADV09","BroadcastToShape target type/rank mismatch");
            }
            types[i->id]=i->type;
        }
    }
    if (result.gradients.size()!=result.wrtParameters.size()) add("ADV10","requested parameter gradient missing");
    for (std::size_t k=0;k<result.gradients.size();++k) {
        auto index=result.gradients[k].parameterIndex;
        if (index>=primal->parameters.size() || result.gradients[k].type!=primal->parameters[index].type)
            add("ADV03","backward output gradient type mismatch");
        if (k<result.wrtParameters.size() && index!=result.wrtParameters[k]) add("ADV10","gradient request order mismatch");
    }
    if (backward && !result.wrtParameters.empty()) {
        Type expected;
        if (result.wrtParameters.size()==1 && result.wrtParameters[0]<primal->parameters.size())
            expected=primal->parameters[result.wrtParameters[0]].type;
        else {
            expected={TypeKind::Tuple,{},0};
            for (auto index:result.wrtParameters) if (index<primal->parameters.size()) expected.elements.push_back(primal->parameters[index].type);
        }
        if (backward->result!=expected) add("ADV03","generated backward result type mismatch");
    }
    return errors;
}

std::string DifferentiationResult::dump() const {
    std::ostringstream out; out << "reverse source=@" << sourceFunction << " forward=@" << forwardFunction << " backward=@" << backwardFunction << '\n';
    out << "wrt"; for (auto x:wrtParameters) out << ' ' << x; out << '\n';
    for (const auto& s:saves) {
        out << "save[" << s.slot << "] primal=%" << s.primal << ':' << typeName(s.type) << " shape=[";
        for (std::size_t d=0;d<s.shape.extents.size();++d) { if (d) out << ','; if (s.shape.extents[d]) out << *s.shape.extents[d]; else out << '?'; }
        out << "] resources="; for (auto id:s.resources) out << id << ','; out << " copy=no reason=" << s.reason << '\n';
    }
    if (module) out << semantic::dump(*module);
    return out.str();
}

Observation executeVjp(const DifferentiationResult& result,const std::vector<RuntimeValue>& arguments,
                       const RuntimeValue& outputCotangent) {
    if (!result.ok()) return {false,{},"TH010-INVALID-AD-RESULT"};
    auto forward=evaluateCall(*result.module,result.module->functions[0].name,arguments);
    if (!forward.ok) return forward;
    const auto* tuple=std::get_if<RuntimeTuple>(&forward.value->data);
    if (!tuple || tuple->size()!=result.saves.size()+1) return {false,{},"TH010-FORWARD-CONTRACT"};
    std::vector<RuntimeValue> backwardArgs;
    for (std::size_t k=1;k<tuple->size();++k) backwardArgs.push_back((*tuple)[k]);
    backwardArgs.push_back(outputCotangent);
    return evaluateCall(*result.module,result.module->functions[1].name,backwardArgs);
}
Observation executeGrad(const DifferentiationResult& result,const std::vector<RuntimeValue>& arguments) {
    if (!result.module || result.module->functions.empty()) return {false,{},"TH010-INVALID-AD-RESULT"};
    const auto* sourceFunction=function(*result.module,result.forwardFunction);
    if (!sourceFunction || sourceFunction->result.kind!=TypeKind::Tuple || sourceFunction->result.elements.empty() ||
        sourceFunction->result.elements[0]!=scalar(TypeKind::F32)) return {false,{},"TH010-GRAD-REQUIRES-F32-SCALAR"};
    return executeVjp(result,arguments,RuntimeValue{1.0f});
}
}
