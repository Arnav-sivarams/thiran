#include "analysis/v0/Ownership.hpp"
#include "semantic/v0/Verifier.hpp"
#include <algorithm>
#include <functional>
#include <sstream>
#include <tuple>

namespace thiran::v0::analysis {
namespace {
using namespace semantic;
using Live = std::set<BindingId>;
using State = std::map<BindingId,BindingFact>;
using Values = std::map<ValueId,ResourceValue>;
constexpr EffectSet bit(EffectKind kind) { return static_cast<EffectSet>(kind); }
bool resourceType(const Type& type) {
    if (type.kind==TypeKind::Tensor || type.kind==TypeKind::Buffer) return true;
    if (type.kind==TypeKind::Tuple) for (const auto& t:type.elements) if (resourceType(t)) return true;
    return false;
}
Live unite(Live a,const Live& b) { a.insert(b.begin(),b.end()); return a; }
ResourceValue combine(ResourceValue a,const ResourceValue& b) {
    if (a==b) return a;
    a.resources.insert(b.resources.begin(),b.resources.end());
    a.copySources.insert(b.copySources.begin(),b.copySources.end());
    a.viewRoots.insert(b.viewRoots.begin(),b.viewRoots.end());
    a.sourceBinding=a.sourceBinding==b.sourceBinding?a.sourceBinding:0;
    a.view=a.view==b.view?a.view:0;
    a.kind=a.kind==b.kind?a.kind:ProvenanceKind::PossibleAlias;
    if (a.elements.size()==b.elements.size())
        for (std::size_t i=0;i<a.elements.size();++i) a.elements[i]=combine(a.elements[i],b.elements[i]);
    else a.elements.clear();
    return a;
}
BindingFact join(const BindingFact& a,const BindingFact& b) {
    BindingFact r=a;
    r.value=combine(a.value,b.value);
    if (a.availability!=b.availability) r.availability=BindingFact::Availability::MaybeUnavailable;
    if (a.moveSite.begin.offset!=b.moveSite.begin.offset)
        r.moveSite=a.moveSite.begin.offset<=b.moveSite.begin.offset?a.moveSite:b.moveSite;
    return r;
}
State merge(State a,const State& b,const State& scope) {
    for (auto it=a.begin();it!=a.end();) if (!scope.contains(it->first)) it=a.erase(it); else ++it;
    for (const auto& [id,f]:b) if (scope.contains(id)) {
        if (a.contains(id)) a[id]=join(a[id],f); else a[id]=f;
    }
    return a;
}
std::set<ResourceId> allResources(const ResourceValue& value) {
    auto r=value.resources;
    for (const auto& e:value.elements) {
        auto inner=allResources(e); r.insert(inner.begin(),inner.end());
    }
    return r;
}
std::set<BindingId> allRoots(const ResourceValue& value) {
    auto r=value.viewRoots;
    for (const auto& e:value.elements) {
        auto inner=allRoots(e); r.insert(inner.begin(),inner.end());
    }
    return r;
}
bool intersects(const std::set<ResourceId>& a,const std::set<ResourceId>& b) {
    for (auto x:a) if (b.contains(x)) return true;
    return false;
}
ResourceId site(FunctionId fn,ValueId value) { return (static_cast<ResourceId>(fn)<<48)|(static_cast<ResourceId>(value)<<16); }
class Liveness {
public:
    std::map<ValueId,Live> after;
    std::map<const BindingWrite*,Live> afterWrite;
    Live universe;
    Live block(const Block& b,Live exit,Live breakExit={},Live continueExit={},bool loop=false) {
        for (auto it=b.steps.rbegin();it!=b.steps.rend();++it) {
            if (const auto* i=std::get_if<Instruction>(&*it)) {
                after[i->id]=exit;
                if (i->op==Op::LoadBinding) exit.insert(i->binding);
                if (i->op==Op::Move) exit.erase(i->binding);
            } else if (const auto* w=std::get_if<BindingWrite>(&*it)) {
                afterWrite[w]=exit; exit.erase(w->binding);
            }
            else if (const auto* f=std::get_if<Flow>(&*it)) {
                exit=f->kind==Flow::Kind::Return?Live{}:
                    f->kind==Flow::Kind::Break?breakExit:continueExit;
            } else if (const auto* s=std::get_if<Structured>(&*it)) {
                if (s->kind==Structured::Kind::If) {
                    auto yes=block(*s->thenBlock,exit,breakExit,continueExit,loop);
                    auto no=s->elseBlock?block(*s->elseBlock,exit,breakExit,continueExit,loop):exit;
                    exit=unite(std::move(yes),no);
                } else {
                    Live head=exit;
                    bool converged=false;
                    for (std::size_t n=0;n<=universe.size()+1;++n) {
                        auto bodyIn=block(*s->bodyBlock,head,exit,head,true);
                        Live next=unite(exit,bodyIn);
                        if (s->kind==Structured::Kind::While)
                            next=unite(next,block(*s->conditionBlock,unite(exit,bodyIn)));
                        if (next==head) { converged=true; break; }
                        head=std::move(next);
                    }
                    if (!converged) {
                        head=unite(head,universe);
                        block(*s->bodyBlock,head,exit,head,true);
                        if (s->kind==Structured::Kind::While) block(*s->conditionBlock,head);
                    }
                    exit=head;
                }
            }
        }
        for (const auto& meta:b.bindings) exit.erase(meta.id);
        return exit;
    }
};
struct Paths { std::vector<State> normal, breaks, continues, returns; };
class Checker {
public:
    Checker(const semantic::Module& m,OwnershipAnalysisResult& result):m_(m),r_(result) {}
    void run(FunctionId id,const Function& fn) {
        fn_=id; values_.clear(); metadata_.clear(); bindingBlock_.clear(); parents_.clear(); readParameterResources_.clear();
        live_.after.clear(); live_.afterWrite.clear(); live_.universe.clear();
        borrowSources_.clear(); moveInputs_.clear();
        collect(fn.body);
        for (const auto& p:fn.parameters) live_.universe.insert(p.binding);
        live_.block(fn.body,{});
        r_.liveAfter[id]=live_.after;
        State state;
        for (const auto& p:fn.parameters) {
            bindingBlock_[p.binding]=fn.body.id;
            ResourceValue value=typedFresh(p.id,p.type,ProvenanceKind::Fresh);
            if (resourceType(p.type)) value.sourceBinding=p.binding;
            values_[p.id]=value;
            state[p.binding]={BindingFact::Availability::DefinitelyAvailable,value,
                p.access==AccessMode::MutableBorrow,p.span,{}};
            if (p.access==AccessMode::Read) {
                auto rs=allResources(value); readParameterResources_.insert(rs.begin(),rs.end());
            }
        }
        auto paths=block(fn.body,state);
        State scope=state;
        for (const auto& meta:fn.body.bindings) scope[meta.id]={};
        State end;
        bool hasPath=false;
        for (const auto& path:paths.normal) { end=hasPath?merge(end,path,scope):path; hasPath=true; }
        for (const auto& path:paths.returns) { end=hasPath?merge(end,path,scope):path; hasPath=true; }
        if (!hasPath) end=state;
        for (auto it=end.begin();it!=end.end();) if (!scope.contains(it->first)) it=end.erase(it); else ++it;
        r_.bindingAvailability[id]=end;
        r_.valueProvenance[id]=values_;
    }
    void runInitializer(const Block& b) {
        fn_=0; values_.clear(); metadata_.clear(); bindingBlock_.clear(); parents_.clear(); readParameterResources_.clear();
        live_.after.clear(); live_.afterWrite.clear(); live_.universe.clear();
        borrowSources_.clear(); moveInputs_.clear();
        collect(b); live_.block(b,{}); r_.liveAfter[0]=live_.after;
        auto paths=block(b,{});
        if (!paths.normal.empty()) r_.bindingAvailability[0]=paths.normal.front();
        r_.valueProvenance[0]=values_;
    }
private:
    const semantic::Module& m_; OwnershipAnalysisResult& r_;
    FunctionId fn_=0;
    Values values_;
    std::map<BindingId,Binding> metadata_;
    std::map<BindingId,BlockId> bindingBlock_;
    std::map<BlockId,BlockId> parents_;
    std::set<ResourceId> readParameterResources_;
    Liveness live_;
    std::set<std::tuple<FunctionId,std::string,std::size_t,std::size_t>> reported_;
    std::set<ValueId> moveInputs_;
    void collect(const Block& b) {
        parents_[b.id]=b.parent;
        for (const auto& meta:b.bindings) {
            metadata_[meta.id]=meta; bindingBlock_[meta.id]=b.id; live_.universe.insert(meta.id);
        }
        for (const auto& step:b.steps) if (const auto* i=std::get_if<Instruction>(&step);
            i && i->op==Op::Move && i->operands.size()==1) moveInputs_.insert(i->operands[0]);
        for (const auto& step:b.steps) if (const auto* s=std::get_if<Structured>(&step)) {
            if (s->thenBlock) collect(*s->thenBlock);
            if (s->elseBlock) collect(*s->elseBlock);
            if (s->conditionBlock) collect(*s->conditionBlock);
            if (s->bodyBlock) collect(*s->bodyBlock);
        }
    }
    void diagnostic(SourceSpan span,std::string category,std::string message) {
        auto key=std::tuple{fn_,category,span.begin.offset,span.end.offset};
        if (reported_.insert(key).second)
            r_.diagnostics.push_back({m_.source,std::move(category),std::move(message),span});
    }
    ResourceValue get(ValueId id) const { auto it=values_.find(id); return it==values_.end()?ResourceValue{}:it->second; }
    void remember(ValueId id,ResourceValue value) {
        auto [it,inserted]=values_.emplace(id,value);
        if (!inserted) it->second=combine(it->second,value);
    }
    ResourceValue fresh(ValueId id,ProvenanceKind kind) {
        return newAt(id,0,kind);
    }
    ResourceValue newAt(ValueId id,std::uint32_t path,ProvenanceKind kind) {
        ResourceValue v; v.kind=kind;
        auto resource=site(fn_,id)|static_cast<ResourceId>(path);
        v.resources.insert(resource);
        r_.createdResources[resource]=kind;
        return v;
    }
    ResourceValue typedFresh(ValueId id,const Type& type,ProvenanceKind kind,std::uint32_t path=0) {
        if (!resourceType(type)) return {};
        if (type.kind!=TypeKind::Tuple) return newAt(id,path,kind);
        ResourceValue result; result.kind=kind;
        for (std::size_t k=0;k<type.elements.size();++k)
            result.elements.push_back(typedFresh(id,type.elements[k],kind,path*64+static_cast<std::uint32_t>(k+1)));
        return result;
    }
    ResourceValue copyOf(const ResourceValue& source,ValueId id,const Type& type,std::uint32_t path=0) {
        if (!resourceType(type)) return {};
        if (type.kind==TypeKind::Tuple) {
            ResourceValue result; result.kind=ProvenanceKind::IndependentCopyOf;
            for (std::size_t k=0;k<type.elements.size();++k)
                result.elements.push_back(copyOf(k<source.elements.size()?source.elements[k]:ResourceValue{},
                    id,type.elements[k],path*64+static_cast<std::uint32_t>(k+1)));
            return result;
        }
        auto result=newAt(id,path,ProvenanceKind::IndependentCopyOf);
        result.copySources=allResources(source);
        return result;
    }
    ResourceValue callResult(ValueId id,const Type& type,const Instruction& call,std::uint32_t path=0) {
        if (!resourceType(type)) return {};
        if (type.kind==TypeKind::Tuple) {
            ResourceValue result; result.kind=ProvenanceKind::PossibleAlias;
            for (std::size_t k=0;k<type.elements.size();++k)
                result.elements.push_back(callResult(id,type.elements[k],call,
                    path*64+static_cast<std::uint32_t>(k+1)));
            return result;
        }
        auto value=newAt(id,path,ProvenanceKind::Fresh);
        value.kind=ProvenanceKind::PossibleAlias;
        for (auto operand:call.operands) {
            auto argument=get(operand);
            auto rs=allResources(argument); value.resources.insert(rs.begin(),rs.end());
            auto roots=allRoots(argument); value.viewRoots.insert(roots.begin(),roots.end());
        }
        return value;
    }
    const BindingFact* binding(const State& state,BindingId id) const {
        auto it=state.find(id); return it==state.end()?nullptr:&it->second;
    }
    void requireAvailable(const State& state,BindingId id,SourceSpan span,bool secondMove=false) {
        auto* f=binding(state,id);
        if (!f || f->availability==BindingFact::Availability::DefinitelyAvailable) return;
        const auto category=f->availability==BindingFact::Availability::MaybeUnavailable?
            "TH006-MAYBE-MOVED":secondMove?"TH006-DOUBLE-MOVE":"TH006-USE-AFTER-MOVE";
        std::string name=metadata_.contains(id)?metadata_.at(id).name:"parameter";
        diagnostic(span,category,"binding '"+name+"' is unavailable after move at "+
            std::to_string(f->moveSite.begin.line)+":"+std::to_string(f->moveSite.begin.column));
    }
    void instruction(const Instruction& i,State& state) {
        ResourceValue value;
        switch (i.op) {
            case Op::Integer: case Op::Float: case Op::Boolean: break;
            case Op::LoadBinding: {
                if (!moveInputs_.contains(i.id)) requireAvailable(state,i.binding,i.span);
                if (auto* f=binding(state,i.binding)) { value=f->value; value.sourceBinding=i.binding; }
                value.kind=value.resources.empty() && value.elements.empty()?ProvenanceKind::NoResource:ProvenanceKind::AliasOf;
                break;
            }
            case Op::TensorLiteral: value=fresh(i.id,ProvenanceKind::Fresh); break;
            case Op::Tuple: {
                value.kind=ProvenanceKind::NoResource;
                std::function<ResourceValue(ResourceValue)> tupleAlias=[&](ResourceValue item) {
                    if (!item.resources.empty()) item.kind=ProvenanceKind::AliasOf;
                    for (auto& child:item.elements) child=tupleAlias(std::move(child));
                    return item;
                };
                for (auto op:i.operands) value.elements.push_back(tupleAlias(get(op)));
                break;
            }
            case Op::Copy: value=copyOf(get(i.operands[0]),i.id,i.type); break;
            case Op::StopGradient: {
                value=get(i.operands[0]);
                if (!value.resources.empty() || !value.elements.empty()) value.kind=ProvenanceKind::AliasOf;
                break;
            }
            case Op::Move: {
                requireAvailable(state,i.binding,i.span,true);
                value=get(i.operands[0]);
                if (auto it=state.find(i.binding);it!=state.end()) {
                    auto after=live_.after[i.id];
                    for (auto other:after) if (other!=i.binding) {
                        auto* f=binding(state,other);
                        if (f && f->availability!=BindingFact::Availability::DefinitelyMoved && allRoots(f->value).contains(i.binding)) {
                            diagnostic(i.span,"TH006-VIEW-ROOT-MOVED","move invalidates a live view rooted in this binding"); break;
                        }
                    }
                    it->second.availability=BindingFact::Availability::DefinitelyMoved;
                    it->second.moveSite=i.span;
                    bool recorded=false;
                    for (const auto& move:r_.moves)
                        recorded|=move.function==fn_ && move.instruction==i.id && move.binding==i.binding;
                    if (!recorded) r_.moves.push_back({fn_,i.id,i.binding,it->second.availability,i.span});
                }
                break;
            }
            case Op::MutableBorrow: {
                requireAvailable(state,i.binding,i.span);
                value=get(i.operands[0]);
                if (!allRoots(value).empty()) diagnostic(i.span,"TH006-LIVE-VIEW-CONFLICT","a read view cannot grant mutable access");
                break;
            }
            case Op::Slice: case Op::Index: case Op::Transpose: {
                if (i.borrowedView && resourceType(i.type)) {
                    auto base=get(i.operands[0]); value.kind=ProvenanceKind::ReadViewOf;
                    value.resources=allResources(base);
                    value.viewRoots=allRoots(base);
                    if (value.viewRoots.empty() && base.sourceBinding) value.viewRoots.insert(base.sourceBinding);
                    if (value.viewRoots.empty())
                        diagnostic(i.span,"TH006-VIEW-ROOT-UNKNOWN","view source has no provable live root handle");
                    value.view=site(fn_,i.id);
                    auto& view=r_.views[value.view]; view={value.view,value.resources,value.viewRoots,i.span};
                }
                break;
            }
            case Op::Call: {
                value=callResult(i.id,i.type,i);
                auto after=live_.after[i.id];
                std::set<ResourceId> borrowed;
                for (std::size_t k=0;k<i.operands.size();++k) {
                    if (m_.functions[i.callee-1].parameters[k].access!=AccessMode::MutableBorrow) continue;
                    auto arg=get(i.operands[k]); auto resources=allResources(arg);
                    if (intersects(resources,readParameterResources_))
                        diagnostic(i.span,"TH006-ACCESS-MODE","read-only parameter resource cannot be mutably borrowed through an alias");
                    if (intersects(resources,borrowed)) diagnostic(i.span,"TH006-MUT-BORROW-CONFLICT","overlapping mutable call arguments");
                    borrowed.insert(resources.begin(),resources.end());
                    BindingId source=borrowSources_.contains(i.operands[k])?borrowSources_.at(i.operands[k]):0;
                    if (!source) { diagnostic(i.span,"TH006-RESOURCE-INVARIANT","mutable argument lacks source binding"); continue; }
                    for (auto other:after) if (other!=source) {
                        auto* f=binding(state,other);
                        if (!f || f->availability==BindingFact::Availability::DefinitelyMoved || !intersects(resources,allResources(f->value))) continue;
                        diagnostic(i.span,!allRoots(f->value).empty()?"TH006-LIVE-VIEW-CONFLICT":"TH006-MUT-BORROW-CONFLICT",
                            "live binding '"+(metadata_.contains(other)?metadata_.at(other).name:"parameter")+"' conflicts with mutable borrow");
                    }
                    for (std::size_t j=0;j<i.operands.size();++j) if (j!=k &&
                        intersects(resources,allResources(get(i.operands[j]))))
                        diagnostic(i.span,"TH006-MUT-BORROW-CONFLICT","another call argument reads the mutable resource");
                    bool recorded=false;
                    for (const auto& b:r_.mutableBorrows) recorded|=b.function==fn_ && b.call==i.id && b.binding==source;
                    if (!recorded) r_.mutableBorrows.push_back({fn_,i.id,source,resources,{},i.span});
                }
                break;
            }
            default: if (resourceType(i.type)) value=fresh(i.id,ProvenanceKind::Fresh); break;
        }
        if (i.op==Op::MutableBorrow) borrowSources_[i.id]=i.binding;
        remember(i.id,value);
    }
    std::map<ValueId,BindingId> borrowSources_;
    bool ancestor(BlockId older,BlockId younger) const {
        for (auto current=younger;current && parents_.contains(current);current=parents_.at(current))
            if (current==older) return true;
        return false;
    }
    void checkViewWrite(const BindingWrite& write,const State& state,const ResourceValue& value) {
        auto roots=allRoots(value);
        for (auto root:roots) {
            if (!bindingBlock_.contains(root) || !bindingBlock_.contains(write.binding)) continue;
            if (root==write.binding || (bindingBlock_[root]!=bindingBlock_[write.binding] &&
                ancestor(bindingBlock_[write.binding],bindingBlock_[root])))
                diagnostic(write.span,"TH006-VIEW-ESCAPE","view rooted in a shorter-lived binding cannot escape into this binding");
        }
        if (write.declaration) return;
        for (auto other:live_.afterWrite[&write]) if (other!=write.binding) {
            auto* f=binding(state,other);
            if (f && f->availability!=BindingFact::Availability::DefinitelyMoved &&
                allRoots(f->value).contains(write.binding)) {
                diagnostic(write.span,"TH006-VIEW-ROOT-REBOUND","rebind invalidates a live view rooted in this binding");
                break;
            }
        }
    }
    Paths block(const Block& b,State start) {
        for (const auto& meta:b.bindings) if (!meta.value && !start.contains(meta.id))
            start[meta.id]={BindingFact::Availability::DefinitelyAvailable,{},false,meta.span,{}};
        Paths paths; paths.normal.push_back(std::move(start));
        for (const auto& step:b.steps) {
            std::vector<State> next;
            for (auto state:paths.normal) {
                if (const auto* i=std::get_if<Instruction>(&step)) { instruction(*i,state); next.push_back(std::move(state)); }
                else if (const auto* w=std::get_if<BindingWrite>(&step)) {
                    auto v=get(w->value);
                    checkViewWrite(*w,state,v);
                    if (w->declaration) {
                        auto meta=metadata_.at(w->binding);
                        state[w->binding]={BindingFact::Availability::DefinitelyAvailable,v,meta.mutableBinding,meta.span,{}};
                    } else if (auto it=state.find(w->binding);it!=state.end()) {
                        it->second.value=v;
                        it->second.availability=BindingFact::Availability::DefinitelyAvailable;
                    }
                    next.push_back(std::move(state));
                } else if (const auto* f=std::get_if<Flow>(&step)) {
                    if (f->kind==Flow::Kind::Return) {
                    if (f->value && !allRoots(get(*f->value)).empty())
                            diagnostic(f->span,"TH006-VIEW-RETURN-DEFERRED","view return requires an explicit caller-root lifetime signature");
                        if (fn_ && f->value && resourceType(m_.functions[fn_-1].result)) {
                            auto returned=get(*f->value);
                            for (const auto& p:m_.functions[fn_-1].parameters)
                                if (p.access==AccessMode::MutableBorrow &&
                                    intersects(allResources(returned),allResources(state[p.binding].value)))
                                    diagnostic(f->span,"TH006-ACCESS-MODE","mutable borrow cannot escape through a return value");
                        }
                        paths.returns.push_back(std::move(state));
                    } else if (f->kind==Flow::Kind::Break) paths.breaks.push_back(std::move(state));
                    else paths.continues.push_back(std::move(state));
                } else if (const auto* s=std::get_if<Structured>(&step)) {
                    if (s->kind==Structured::Kind::If) {
                        auto yes=block(*s->thenBlock,state);
                        auto no=s->elseBlock?block(*s->elseBlock,state):Paths{{state},{},{},{}};
                        for (auto p:yes.normal) next.push_back(merge(p,p,state));
                        for (auto p:no.normal) next.push_back(merge(p,p,state));
                        paths.breaks.insert(paths.breaks.end(),yes.breaks.begin(),yes.breaks.end());
                        paths.breaks.insert(paths.breaks.end(),no.breaks.begin(),no.breaks.end());
                        paths.continues.insert(paths.continues.end(),yes.continues.begin(),yes.continues.end());
                        paths.continues.insert(paths.continues.end(),no.continues.begin(),no.continues.end());
                        paths.returns.insert(paths.returns.end(),yes.returns.begin(),yes.returns.end());
                        paths.returns.insert(paths.returns.end(),no.returns.begin(),no.returns.end());
                    } else {
                        State head=state, exit=state;
                        bool converged=false;
                        for (unsigned count=0;count<512;++count) {
                            State bodyStart=head;
                            if (s->kind==Structured::Kind::While) {
                                auto condition=block(*s->conditionBlock,head);
                                if (!condition.normal.empty()) {
                                    bodyStart=condition.normal.front();
                                    exit=merge(exit,bodyStart,state);
                                }
                            }
                            auto body=block(*s->bodyBlock,bodyStart);
                            State back=head;
                            for (const auto& p:body.normal) back=merge(back,p,state);
                            for (const auto& p:body.continues) back=merge(back,p,state);
                            for (const auto& p:body.breaks) exit=merge(exit,p,state);
                            paths.returns.insert(paths.returns.end(),body.returns.begin(),body.returns.end());
                            if (back==head) { converged=true; break; }
                            head=std::move(back);
                        }
                        if (!converged)
                            diagnostic(s->span,"TH006-ANALYSIS-INCONCLUSIVE","loop ownership state did not reach a bounded fixed point");
                        next.push_back(merge(exit,head,state));
                    }
                } else next.push_back(std::move(state));
            }
            if (next.size()>1) {
                State scope=paths.normal.empty()?State{}:paths.normal.front();
                State joined=next.front();
                for (std::size_t k=1;k<next.size();++k) joined=merge(joined,next[k],scope);
                paths.normal={std::move(joined)};
            } else paths.normal=std::move(next);
        }
        return paths;
    }
};
void effects(const semantic::Module& m,OwnershipAnalysisResult& r) {
    for (const auto& f:m.functions) {
        auto& summary=r.functionEffects[f.id];
        for (std::size_t k=0;k<f.parameters.size();++k)
            if (f.parameters[k].access==AccessMode::MutableBorrow) {
                summary.kinds|=bit(EffectKind::Mutates);
                summary.mutatedParameters.insert(k);
            }
    }
    std::function<void(const Block&,FunctionEffects&,std::set<FunctionId>&)> scan =
        [&](const Block& b,FunctionEffects& out,std::set<FunctionId>& calls) {
            for (const auto& step:b.steps) {
                if (std::holds_alternative<Check>(step)) out.kinds|=bit(EffectKind::MayTrap);
                if (const auto* i=std::get_if<Instruction>(&step)) {
                    if (i->op==Op::Call) calls.insert(i->callee);
                    if (i->op==Op::MutableBorrow) out.kinds|=bit(EffectKind::Mutates);
                    if (i->effect==EffectClass::CheckedFailure && i->op!=Op::Call)
                        out.kinds|=bit(EffectKind::MayTrap);
                }
                if (const auto* s=std::get_if<Structured>(&step)) {
                    if (s->thenBlock) scan(*s->thenBlock,out,calls);
                    if (s->elseBlock) scan(*s->elseBlock,out,calls);
                    if (s->conditionBlock) scan(*s->conditionBlock,out,calls);
                    if (s->bodyBlock) scan(*s->bodyBlock,out,calls);
                }
            }
        };
    std::map<FunctionId,std::set<FunctionId>> calls;
    for (const auto& f:m.functions) scan(f.body,r.functionEffects[f.id],calls[f.id]);
    for (std::size_t n=0;n<m.functions.size()+1;++n) {
        bool changed=false;
        for (const auto& f:m.functions) for (auto callee:calls[f.id]) {
            auto& summary=r.functionEffects[f.id]; auto old=summary.kinds;
            summary.kinds|=r.functionEffects[callee].kinds;
            changed|=old!=summary.kinds;
        }
        if (!changed) break;
    }
    for (auto& [id,summary]:r.functionEffects)
        summary.pureTensorCandidate=(summary.kinds==0);
}
std::vector<semantic::SemanticDiagnostic> ownershipIRIssues(const semantic::Module& m) {
    std::vector<semantic::SemanticDiagnostic> errors;
    for (const auto& f:m.functions) {
        std::map<ValueId,Op> produced;
        std::map<ValueId,SourceSpan> borrowSpans;
        std::map<ValueId,unsigned> borrowUses;
        std::function<void(const Block&)> collect=[&](const Block& b) {
            for (const auto& step:b.steps) {
                if (const auto* i=std::get_if<Instruction>(&step)) {
                    produced[i->id]=i->op;
                    if (i->op==Op::MutableBorrow) borrowSpans[i->id]=i->span;
                }
                if (const auto* s=std::get_if<Structured>(&step)) {
                    if (s->thenBlock) collect(*s->thenBlock);
                    if (s->elseBlock) collect(*s->elseBlock);
                    if (s->conditionBlock) collect(*s->conditionBlock);
                    if (s->bodyBlock) collect(*s->bodyBlock);
                }
            }
        };
        collect(f.body);
        std::function<void(const Block&)> inspect=[&](const Block& b) {
            for (const auto& step:b.steps) {
                if (const auto* i=std::get_if<Instruction>(&step)) {
                    for (auto operand:i->operands) if (borrowSpans.contains(operand)) ++borrowUses[operand];
                    if (i->op==Op::Call && i->callee && i->callee<=m.functions.size()) {
                        const auto& target=m.functions[i->callee-1];
                        for (std::size_t k=0;k<i->operands.size() && k<target.parameters.size();++k) {
                            auto op=produced.contains(i->operands[k])?produced.at(i->operands[k]):Op::Integer;
                            auto mode=target.parameters[k].access;
                            if ((mode==AccessMode::MutableBorrow && op!=Op::MutableBorrow) ||
                                (mode==AccessMode::Consume && op!=Op::Move) ||
                                (mode!=AccessMode::MutableBorrow && op==Op::MutableBorrow))
                                errors.push_back({m.source,"TH006-ACCESS-MODE",
                                    "call operand does not match declared ownership access",i->span});
                        }
                    }
                } else if (const auto* w=std::get_if<BindingWrite>(&step)) {
                    if (borrowSpans.contains(w->value)) ++borrowUses[w->value];
                } else if (const auto* flow=std::get_if<Flow>(&step)) {
                    if (flow->value && borrowSpans.contains(*flow->value)) ++borrowUses[*flow->value];
                } else if (const auto* c=std::get_if<Check>(&step)) {
                    for (auto operand:c->operands) if (borrowSpans.contains(operand)) ++borrowUses[operand];
                } else if (const auto* s=std::get_if<Structured>(&step)) {
                    if (s->thenBlock) inspect(*s->thenBlock);
                    if (s->elseBlock) inspect(*s->elseBlock);
                    if (s->conditionBlock) inspect(*s->conditionBlock);
                    if (s->bodyBlock) inspect(*s->bodyBlock);
                }
            }
        };
        inspect(f.body);
        for (const auto& [id,span]:borrowSpans) if (borrowUses[id]!=1)
            errors.push_back({m.source,"TH006-ACCESS-MODE",
                "mutable borrow must be used exactly once as a call argument",span});
    }
    return errors;
}
}
OwnershipAnalysisResult analyze(const semantic::Module& module) {
    OwnershipAnalysisResult result;
    auto verified=semantic::verify(module);
    if (!verified.ok) {
        result.diagnostics.push_back({module.source,"TH006-RESOURCE-INVARIANT",
            "semantic IR failed verification before ownership analysis",module.span});
        return result;
    }
    result.diagnostics=ownershipIRIssues(module);
    if (!result.diagnostics.empty()) return result;
    Checker checker(module,result);
    for (const auto& f:module.functions) checker.run(f.id,f);
    checker.runInitializer(module.initializer);
    effects(module,result);
    if (!result.diagnostics.empty())
        for (auto& [id,summary]:result.functionEffects) { (void)id; summary.pureTensorCandidate=false; }
    auto issues=auditFacts(module,result);
    if (!issues.empty())
        result.diagnostics.push_back({module.source,"TH006-RESOURCE-INVARIANT",issues.front(),module.span});
    return result;
}
std::vector<std::string> auditFacts(const semantic::Module& m,const OwnershipAnalysisResult& r) {
    std::vector<std::string> errors;
    std::map<FunctionId,std::set<BindingId>> knownRoots;
    std::function<void(FunctionId,const semantic::Block&)> roots=[&](FunctionId fn,const semantic::Block& block) {
        for (const auto& meta:block.bindings) knownRoots[fn].insert(meta.id);
        for (const auto& step:block.steps) if (const auto* s=std::get_if<semantic::Structured>(&step)) {
            if (s->thenBlock) roots(fn,*s->thenBlock);
            if (s->elseBlock) roots(fn,*s->elseBlock);
            if (s->conditionBlock) roots(fn,*s->conditionBlock);
            if (s->bodyBlock) roots(fn,*s->bodyBlock);
        }
    };
    for (const auto& f:m.functions) {
        for (const auto& p:f.parameters) knownRoots[f.id].insert(p.binding);
        roots(f.id,f.body);
    }
    roots(0,m.initializer);
    for (const auto& [fn,values]:r.valueProvenance) for (const auto& [id,v]:values) {
        std::function<void(const ResourceValue&)> createdAt=[&](const ResourceValue& item) {
            if (item.kind==ProvenanceKind::Fresh || item.kind==ProvenanceKind::IndependentCopyOf)
                for (auto resource:item.resources) if ((resource>>16)!=(site(fn,id)>>16))
                    errors.push_back("duplicate or inconsistent fresh ResourceId provenance");
            for (const auto& e:item.elements) createdAt(e);
        };
        createdAt(v);
        for (auto resource:allResources(v)) if (!r.createdResources.contains(resource))
            errors.push_back("alias points to unknown ResourceId");
        std::function<void(const ResourceValue&)> copySources=[&](const ResourceValue& item) {
            for (auto source:item.copySources) if (!r.createdResources.contains(source))
                errors.push_back("copy source points to unknown ResourceId");
            if (item.kind==ProvenanceKind::IndependentCopyOf &&
                intersects(item.resources,item.copySources))
                errors.push_back("copy shares source ResourceId");
            for (const auto& e:item.elements) copySources(e);
        };
        copySources(v);
        for (auto root:allRoots(v)) if (!knownRoots[fn].contains(root))
            errors.push_back("view points to unknown root BindingId");
    }
    for (const auto& [id,view]:r.views) {
        if (id!=view.id || view.roots.empty()) errors.push_back("view points to unknown root");
        auto fn=static_cast<FunctionId>(id>>48);
        for (auto root:view.roots) if (!knownRoots[fn].contains(root))
            errors.push_back("view points to unknown root BindingId");
        for (auto resource:view.resources) if (!r.createdResources.contains(resource)) errors.push_back("view points to unknown ResourceId");
    }
    for (const auto& borrow:r.mutableBorrows) {
        if (borrow.resources.empty()) errors.push_back("mutation references non-resource scalar");
        if (!borrow.liveConflicts.empty()) errors.push_back("mutable borrow coexists with live alias");
        for (auto resource:borrow.resources) if (!r.createdResources.contains(resource)) errors.push_back("borrow points to unknown ResourceId");
    }
    for (const auto& move:r.moves) if (move.availabilityAfter!=BindingFact::Availability::DefinitelyMoved)
        errors.push_back("moved binding marked available");
    std::function<void(FunctionId,const semantic::Block&)> copies=[&](FunctionId fn,const semantic::Block& block) {
        for (const auto& step:block.steps) {
            if (const auto* i=std::get_if<semantic::Instruction>(&step);
                i && i->op==semantic::Op::Copy && i->operands.size()==1 &&
                r.valueProvenance.contains(fn) && r.valueProvenance.at(fn).contains(i->id) &&
                r.valueProvenance.at(fn).contains(i->operands[0]) &&
                intersects(allResources(r.valueProvenance.at(fn).at(i->id)),
                    allResources(r.valueProvenance.at(fn).at(i->operands[0]))))
                errors.push_back("copy shares source ResourceId");
            if (const auto* s=std::get_if<semantic::Structured>(&step)) {
                if (s->thenBlock) copies(fn,*s->thenBlock);
                if (s->elseBlock) copies(fn,*s->elseBlock);
                if (s->conditionBlock) copies(fn,*s->conditionBlock);
                if (s->bodyBlock) copies(fn,*s->bodyBlock);
            }
        }
    };
    for (const auto& f:m.functions) copies(f.id,f.body);
    copies(0,m.initializer);
    for (const auto& f:m.functions) {
        for (const auto& p:f.parameters) if (p.access!=semantic::AccessMode::Read &&
            p.access!=semantic::AccessMode::MutableBorrow && p.access!=semantic::AccessMode::Consume)
            errors.push_back("impossible parameter mode");
        auto it=r.functionEffects.find(f.id);
        if (it==r.functionEffects.end()) errors.push_back("missing effect summary");
        else if (it->second.pureTensorCandidate && it->second.kinds!=0) errors.push_back("inconsistent effect summary");
        if (it!=r.functionEffects.end()) {
            if (it->second.kinds & ~static_cast<EffectSet>(63)) errors.push_back("unknown effect kind");
            for (std::size_t k=0;k<f.parameters.size();++k) if (f.parameters[k].access==semantic::AccessMode::MutableBorrow &&
                (!(it->second.kinds & bit(EffectKind::Mutates)) || !it->second.mutatedParameters.contains(k)))
                errors.push_back("inconsistent parameter mutation summary");
            std::function<void(const semantic::Block&)> auditEffects=[&](const semantic::Block& block) {
                for (const auto& step:block.steps) {
                    if (std::holds_alternative<semantic::Check>(step) &&
                        !(it->second.kinds & bit(EffectKind::MayTrap)))
                        errors.push_back("runtime Check missing MayTrap summary");
                    if (const auto* i=std::get_if<semantic::Instruction>(&step)) {
                        if (i->effect==semantic::EffectClass::CheckedFailure && i->op!=semantic::Op::Call &&
                            !(it->second.kinds & bit(EffectKind::MayTrap)))
                            errors.push_back("checked instruction missing MayTrap summary");
                        if (i->op==semantic::Op::MutableBorrow && !(it->second.kinds & bit(EffectKind::Mutates)))
                            errors.push_back("mutable borrow missing Mutates summary");
                        if (i->op==semantic::Op::Call && r.functionEffects.contains(i->callee) &&
                            (r.functionEffects.at(i->callee).kinds & ~it->second.kinds))
                            errors.push_back("callee effect missing from caller summary");
                    }
                    if (const auto* s=std::get_if<semantic::Structured>(&step)) {
                        if (s->thenBlock) auditEffects(*s->thenBlock);
                        if (s->elseBlock) auditEffects(*s->elseBlock);
                        if (s->conditionBlock) auditEffects(*s->conditionBlock);
                        if (s->bodyBlock) auditEffects(*s->bodyBlock);
                    }
                }
            };
            auditEffects(f.body);
        }
    }
    return errors;
}
std::string OwnershipAnalysisResult::dump() const {
    std::ostringstream out;
    for (const auto& [fn,values]:valueProvenance) {
        out << "function " << fn << '\n';
        for (const auto& [id,v]:values) {
            out << '%' << id << " kind=" << static_cast<int>(v.kind);
            for (auto resource:allResources(v)) out << " R" << resource;
            for (auto source:v.copySources) out << " copiedFrom=R" << source;
            for (auto root:allRoots(v)) out << " root=$" << root;
            out << '\n';
        }
    }
    for (const auto& [fn,bindings]:bindingAvailability) {
        out << "bindings " << fn << '\n';
        for (const auto& [id,f]:bindings) {
            out << '$' << id << " availability=" << static_cast<int>(f.availability);
            for (auto resource:allResources(f.value)) out << " R" << resource;
            for (auto root:allRoots(f.value)) out << " root=$" << root;
            out << '\n';
        }
    }
    for (const auto& [id,v]:views) {
        out << "view " << id;
        for (auto resource:v.resources) out << " R" << resource;
        for (auto root:v.roots) out << " root=$" << root;
        out << '\n';
    }
    for (const auto& borrow:mutableBorrows) {
        out << "borrow " << borrow.function << " %" << borrow.call << " $" << borrow.binding;
        for (auto resource:borrow.resources) out << " R" << resource;
        out << '\n';
    }
    for (const auto& [fn,after]:liveAfter) for (const auto& [id,bindings]:after) {
        out << "live " << fn << " after %" << id;
        for (auto binding:bindings) out << " $" << binding;
        out << '\n';
    }
    for (const auto& [fn,effects]:functionEffects)
        out << "effects " << fn << " kinds=" << effects.kinds << " candidate=" << effects.pureTensorCandidate << '\n';
    return out.str();
}
}
