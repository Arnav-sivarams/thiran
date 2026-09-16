#include "semantic/v0/Verifier.hpp"
#include <algorithm>
#include <map>
#include <set>

namespace thiran::v0::semantic {
namespace {
void err(VerificationResult& r,const char* message) { r.errors.emplace_back(message); }
bool shapeOk(const Type& t,const ShapeFact& s) {
    if (s.extents.size()!=(t.kind==TypeKind::Tensor?t.rank:0)) return false;
    for (auto x:s.extents) if (x && *x<0) return false;
    return true;
}
bool spanOk(const SourceSpan& s) {
    return s.begin.source!=frontend::invalidSourceId && s.begin.source==s.end.source &&
        s.begin.offset<=s.end.offset && s.begin.line>=1 && s.end.line>=1 &&
        s.begin.column>=1 && s.end.column>=1;
}
struct Slot { Type type; bool mutableBinding; };
struct State { ValueId value=1; BindingId binding=1; BlockId block=1; std::set<const Block*> seen; };
using Values=std::map<ValueId,Type>;
using Slots=std::map<BindingId,Slot>;
bool returns(const Block& b) {
    if (b.returned) return true;
    for (const auto& step:b.steps) {
        if (auto* f=std::get_if<Flow>(&step)) return f->kind==Flow::Kind::Return;
        if (auto* s=std::get_if<Structured>(&step); s && s->kind==Structured::Kind::If &&
            s->thenBlock && s->elseBlock && returns(*s->thenBlock) && returns(*s->elseBlock)) return true;
    }
    return false;
}
void inspect(const Module& m,const Block& b,const Function* fn,State& state,VerificationResult& r,
             Values values,Slots slots,BlockId parent,unsigned loopDepth,bool induction=false,BindingId inductionId=0) {
    if (!state.seen.insert(&b).second) { err(r,"duplicate nested block ownership/cycle"); return; }
    if (parent) {
        if (!b.id || b.id!=state.block++ || b.parent!=parent) err(r,"malformed nested block identity/parent");
    } else if (!b.id || b.id!=state.block++ || b.parent) err(r,"malformed root block identity");
    if (!b.terminated) err(r,"missing structural block completion");
    for (const auto& meta:b.bindings) if (!spanOk(meta.span)) err(r,"binding metadata has invalid source provenance");
    std::set<std::string> localNames;
    if (!parent && fn) for (const auto& p:fn->parameters) localNames.insert(p.name);
    if (induction) {
        if (!inductionId || inductionId!=state.binding++) err(r,"malformed loop induction identity");
        slots[inductionId]={scalar(TypeKind::I64),false};
        bool found=false;
        for (const auto& meta:b.bindings) if (meta.id==inductionId && !meta.value &&
            meta.type==scalar(TypeKind::I64) && !meta.mutableBinding) { found=true; localNames.insert(meta.name); }
        if (!found) err(r,"missing loop induction binding metadata");
    }
    auto used=[&](ValueId id)->Type {
        auto it=values.find(id);
        if (it==values.end()) { err(r,"inaccessible or use-before-definition ValueId"); return {}; }
        return it->second;
    };
    auto field=[&](std::optional<ValueId> id) { if (id && used(*id)!=scalar(TypeKind::I64)) err(r,"selector field is not i64"); };
    auto selectors=[&](const std::vector<Selector>& list) {
        for (const auto& s:list) {
            if (s.slice) { if (s.index) err(r,"slice has integer index"); field(s.start); field(s.end); field(s.step); }
            else { if (!s.index || s.start || s.end || s.step) err(r,"malformed integer selector"); field(s.index); }
        }
    };
    // A trapping Check must guard its dependent operation in this very region.
    // This prevents even manually constructed IR from placing a branch Check outside its branch.
    for (std::size_t ci=0;ci<b.steps.size();++ci) if (auto* c=std::get_if<Check>(&b.steps[ci])) {
        bool attached=false;
        for (std::size_t j=ci+1;j<b.steps.size();++j) {
            if (std::holds_alternative<Flow>(b.steps[j]) || std::holds_alternative<Structured>(b.steps[j])) break;
            auto* i=std::get_if<Instruction>(&b.steps[j]); if (!i) continue;
            if (c->kind==CheckKind::Bounds && (i->op==Op::Index || i->op==Op::Slice) &&
                c->operands.size()==2 && i->operands.size()==1 && i->operands[0]==c->operands[0] &&
                c->axis<i->selectors.size() && i->selectors[c->axis].index==c->operands[1]) attached=true;
            if (c->kind==CheckKind::Slice && i->op==Op::Slice && c->operands.size()==1 &&
                i->operands.size()==1 && i->operands[0]==c->operands[0] && c->axis<i->selectors.size() &&
                c->selectors.size()==1 && i->selectors[c->axis].slice &&
                i->selectors[c->axis].start==c->selectors[0].start &&
                i->selectors[c->axis].end==c->selectors[0].end &&
                i->selectors[c->axis].step==c->selectors[0].step) attached=true;
            if (c->kind==CheckKind::Broadcast && (i->op==Op::Add || i->op==Op::Subtract ||
                i->op==Op::Multiply || i->op==Op::ElementMultiply) && i->operands==c->operands) attached=true;
            if (c->kind==CheckKind::MatmulShape && i->op==Op::Matmul && i->operands==c->operands) attached=true;
            if (attached) break;
        }
        if (!attached) err(r,"runtime Check is not attached to dependent operation in its region");
    }
    bool exited=false;
    for (const auto& step:b.steps) {
        if (exited) err(r,"step after unconditional exit");
        if (auto* c=std::get_if<Check>(&step)) {
            if (!spanOk(c->span)) err(r,"Check has invalid source provenance");
            for (auto id:c->operands) used(id);
            selectors(c->selectors);
            auto operand=[&](std::size_t k)->Type { return k<c->operands.size()?used(c->operands[k]):Type{}; };
            if (c->effect!=EffectClass::CheckedFailure || c->failureId.empty()) err(r,"malformed runtime Check effect/ID");
            if (c->kind==CheckKind::Bounds) {
                auto t=operand(0);
                if (c->operands.size()!=2 || t.kind!=TypeKind::Tensor || operand(1)!=scalar(TypeKind::I64) ||
                    c->axis>=t.rank || c->failureId!="TH-SPEC-BOUNDS") err(r,"malformed bounds Check");
            } else if (c->kind==CheckKind::Broadcast || c->kind==CheckKind::MatmulShape) {
                if (c->operands.size()!=2 || operand(0).kind!=TypeKind::Tensor || operand(1).kind!=TypeKind::Tensor ||
                    c->failureId!=(c->kind==CheckKind::Broadcast?"TH-SPEC-BROADCAST":"TH-SPEC-SHAPE")) err(r,"malformed shape Check");
            } else if (c->kind==CheckKind::Slice) {
                auto t=operand(0);
                if (c->operands.size()!=1 || t.kind!=TypeKind::Tensor || c->axis>=t.rank ||
                    c->selectors.size()!=1 || !c->selectors[0].slice || c->failureId!="TH-SPEC-SLICE") err(r,"malformed slice Check");
            } else err(r,"invalid Check kind");
            continue;
        }
        if (auto* w=std::get_if<BindingWrite>(&step)) {
            if (!spanOk(w->span)) err(r,"binding state change has invalid source provenance");
            auto t=used(w->value);
            if (w->declaration) {
                const Binding* meta=nullptr;
                for (const auto& x:b.bindings) if (x.id==w->binding) { meta=&x; break; }
                if (!meta || !w->binding || w->binding!=state.binding++ || meta->value!=w->value ||
                    meta->type!=t || !shapeOk(meta->type,meta->shape) || !localNames.insert(meta->name).second)
                    err(r,"malformed branch binding declaration/join state");
                if (meta) slots[w->binding]={meta->type,meta->mutableBinding};
            } else {
                auto it=slots.find(w->binding);
                if (it==slots.end() || !it->second.mutableBinding || it->second.type!=t)
                    err(r,"malformed loop-carried/outer binding rebind contract");
            }
            continue;
        }
        if (auto* f=std::get_if<Flow>(&step)) {
            if (!spanOk(f->span)) err(r,"control exit has invalid source provenance");
            if (f->kind==Flow::Kind::Return) {
                if (!fn || !f->value || used(f->value.value_or(0))!=fn->result) err(r,"Return type mismatch");
            } else if (!loopDepth || f->value) err(r,"break/continue outside loop or malformed payload");
            exited=true; continue;
        }
        if (auto* s=std::get_if<Structured>(&step)) {
            if (!spanOk(s->span)) err(r,"structured control node has invalid source provenance");
            if (s->kind==Structured::Kind::If) {
                if (used(s->condition)!=scalar(TypeKind::Bool) || !s->thenBlock || s->bodyBlock || s->conditionBlock || s->conditionResult)
                    err(r,"If condition/region contract invalid");
                if (s->thenBlock) inspect(m,*s->thenBlock,fn,state,r,values,slots,b.id,loopDepth);
                if (s->elseBlock) inspect(m,*s->elseBlock,fn,state,r,values,slots,b.id,loopDepth);
            } else if (s->kind==Structured::Kind::ForRange) {
                if (used(s->start)!=scalar(TypeKind::I64) || used(s->end)!=scalar(TypeKind::I64) ||
                    !s->bodyBlock || !s->induction || s->thenBlock || s->elseBlock || s->conditionBlock)
                    err(r,"ForRange bounds/body contract invalid");
                if (s->bodyBlock) inspect(m,*s->bodyBlock,fn,state,r,values,slots,b.id,loopDepth+1,true,s->induction);
            } else if (s->kind==Structured::Kind::While) {
                if (!s->conditionBlock || !s->bodyBlock || !s->conditionResult || s->thenBlock || s->elseBlock)
                    err(r,"While structural regions invalid");
                if (s->conditionBlock) {
                    inspect(m,*s->conditionBlock,fn,state,r,values,slots,b.id,loopDepth);
                    bool found=false;
                    for (const auto& x:s->conditionBlock->steps) if (auto* i=std::get_if<Instruction>(&x);
                        i && i->id==s->conditionResult) { found=i->type==scalar(TypeKind::Bool); break; }
                    if (!found) err(r,"While condition result missing/non-bool");
                }
                if (s->bodyBlock) inspect(m,*s->bodyBlock,fn,state,r,values,slots,b.id,loopDepth+1);
            } else err(r,"invalid structured kind");
            continue;
        }
        const auto& i=std::get<Instruction>(step);
        if (!spanOk(i.span)) err(r,"instruction has invalid source provenance");
        std::vector<Type> operands;
        for (auto id:i.operands) operands.push_back(used(id));
        selectors(i.selectors);
        if (!validType(i.type) || !shapeOk(i.type,i.shape)) err(r,"instruction type/shape invalid");
        if (!i.id || i.id!=state.value++ || values.contains(i.id)) err(r,"duplicate/nonmonotonic ValueId");
        auto require=[&](bool ok) { if (!ok) err(r,"instruction operand/result contract invalid"); };
        switch (i.op) {
            case Op::Integer: require(operands.empty() && i.integer && i.type==scalar(TypeKind::I64)); break;
            case Op::Boolean: require(operands.empty() && i.boolean && i.type==scalar(TypeKind::Bool)); break;
            case Op::LoadBinding: {
                auto it=slots.find(i.binding); require(operands.empty() && it!=slots.end() && it->second.type==i.type); break;
            }
            case Op::TensorLiteral: {
                bool ok=i.type.kind==TypeKind::Tensor && i.type.elements.size()==1 &&
                    i.type.elements[0]==scalar(TypeKind::I64) && (i.type.rank==1 || i.type.rank==2);
                for (auto t:operands) ok &= t==scalar(TypeKind::I64); require(ok); break;
            }
            case Op::Tuple: require(i.type.kind==TypeKind::Tuple && i.type.elements==operands); break;
            case Op::Copy: require(operands.size()==1 && i.type==operands[0] && !i.binding); break;
            case Op::Move: case Op::MutableBorrow: {
                auto it=slots.find(i.binding);
                require(operands.size()==1 && i.type==operands[0] && it!=slots.end() && it->second.type==i.type &&
                    (i.op!=Op::MutableBorrow || (it->second.mutableBinding &&
                        (i.type.kind==TypeKind::Tensor || i.type.kind==TypeKind::Buffer)))); break;
            }
            case Op::Negate: require(operands.size()==1 && i.type==operands[0] && executableType(i.type) && i.type.kind!=TypeKind::Tuple && i.type.kind!=TypeKind::Bool); break;
            case Op::Add: case Op::Subtract: case Op::Multiply: case Op::ElementMultiply: {
                bool ok=operands.size()==2 && executableType(i.type) && i.type.kind!=TypeKind::Tuple && i.type.kind!=TypeKind::Bool;
                if (ok) {
                    bool a=operands[0].kind==TypeKind::Tensor,c=operands[1].kind==TypeKind::Tensor;
                    ok &= (a || operands[0]==scalar(TypeKind::I64)) && (c || operands[1]==scalar(TypeKind::I64));
                    ok &= i.type==(a||c?tensor(scalar(TypeKind::I64),std::max(a?operands[0].rank:0U,c?operands[1].rank:0U)):scalar(TypeKind::I64));
                    if (i.op==Op::Multiply) ok &= !a && !c;
                }
                require(ok); break;
            }
            case Op::Matmul: require(operands.size()==2 && operands[0].kind==TypeKind::Tensor && operands[1].kind==TypeKind::Tensor &&
                operands[0].rank==2 && operands[1].rank==2 && i.type==tensor(scalar(TypeKind::I64),2)); break;
            case Op::Index: case Op::Slice: {
                bool ok=operands.size()==1 && operands[0].kind==TypeKind::Tensor && i.selectors.size()<=operands[0].rank;
                if (ok) {
                    unsigned removed=0; bool sliced=false;
                    for (auto s:i.selectors) { removed+=!s.slice; sliced|=s.slice; }
                    auto rank=operands[0].rank-removed;
                    ok &= i.type==(rank?tensor(scalar(TypeKind::I64),rank):scalar(TypeKind::I64));
                    ok &= (i.op==Op::Slice)==sliced && i.borrowedView==(sliced || rank>0);
                }
                require(ok); break;
            }
            case Op::Transpose: require(operands.size()==1 && operands[0].kind==TypeKind::Tensor && operands[0].rank==2 && i.type==operands[0] && i.borrowedView); break;
            case Op::Sum: require(operands.size()==2 && operands[0].kind==TypeKind::Tensor && operands[1]==scalar(TypeKind::I64) &&
                i.axis<operands[0].rank && i.type==(operands[0].rank==1?scalar(TypeKind::I64):tensor(scalar(TypeKind::I64),operands[0].rank-1))); break;
            case Op::Call: {
                if (!i.callee || i.callee>m.functions.size()) { err(r,"invalid function target"); break; }
                const auto& target=m.functions[i.callee-1];
                bool ok=target.id==i.callee && i.type==target.result && operands.size()==target.parameters.size();
                if (operands.size()==target.parameters.size()) for (std::size_t k=0;k<operands.size();++k) ok &= operands[k]==target.parameters[k].type;
                if (!i.argumentAccess.empty()) {
                    ok &= i.argumentAccess.size()==target.parameters.size();
                    if (i.argumentAccess.size()==target.parameters.size())
                        for (std::size_t k=0;k<i.argumentAccess.size();++k) ok &= i.argumentAccess[k]==target.parameters[k].access;
                } else for (const auto& p:target.parameters) ok &= p.access==AccessMode::Read;
                require(ok); break;
            }
        }
        values.emplace(i.id,i.type);
    }
    if (b.returned && (!fn || used(*b.returned)!=fn->result)) err(r,"block Return type mismatch");
}
}
VerificationResult verify(const Module& m) {
    VerificationResult r; std::set<std::string> names;
    for (std::size_t k=0;k<m.functions.size();++k) {
        const auto& f=m.functions[k];
        if (f.id!=k+1 || !names.insert(f.name).second) err(r,"FunctionId/name invalid");
        if (!validType(f.result)) err(r,"function result type invalid");
        State state; Values values; Slots slots;
        for (const auto& p:f.parameters) {
            if (!spanOk(p.span)) err(r,"parameter has invalid source provenance");
            if (p.id!=state.value++ || p.binding!=state.binding++ || values.contains(p.id) || slots.contains(p.binding))
                err(r,"parameter ID invalid");
            if (!validType(p.type) || !shapeOk(p.type,p.shape)) err(r,"parameter type/shape invalid");
            if (p.access!=AccessMode::Read && p.access!=AccessMode::MutableBorrow && p.access!=AccessMode::Consume)
                err(r,"invalid parameter access mode");
            if (p.access==AccessMode::MutableBorrow && p.type.kind!=TypeKind::Tensor && p.type.kind!=TypeKind::Buffer)
                err(r,"mutable parameter access mode requires resource type");
            values[p.id]=p.type; slots[p.binding]={p.type,p.access==AccessMode::MutableBorrow};
        }
        inspect(m,f.body,&f,state,r,values,slots,0,0);
        if (!returns(f.body)) err(r,"function lacks return on some path");
    }
    State init; inspect(m,m.initializer,nullptr,init,r,{}, {},0,0);
    r.ok=r.errors.empty(); return r;
}
}
