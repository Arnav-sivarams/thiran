#include "analysis/v0/Ownership.hpp"
#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"
#include <iostream>
#include <stdexcept>

using namespace thiran::v0;
namespace s=thiran::v0::semantic;
namespace a=thiran::v0::analysis;
namespace {
void expect(bool yes,const std::string& why) { if (!yes) throw std::runtime_error(why); }
constexpr const char* helpers=R"(fn touch(x: borrow mut Tensor<i64,2>) -> i64 { return 0 }
fn use(x: Tensor<i64,2>) -> i64 { return x[0,0] }
)";
struct Program { s::Module module; a::OwnershipAnalysisResult facts; };
Program compile(const std::string& source) {
    auto parsed=parse(source,"ownership.th");
    expect(parsed.module.has_value(),"parse: "+(parsed.diagnostics.empty()?std::string{}:parsed.diagnostics[0].format()));
    auto sem=s::analyze(*parsed.module);
    expect(sem.module.has_value(),"semantic: "+(sem.diagnostics.empty()?std::string{}:sem.diagnostics[0].format()));
    expect(s::verify(*sem.module).ok,"semantic verifier");
    auto facts=a::analyze(*sem.module);
    return {std::move(*sem.module),std::move(facts)};
}
std::string category(const std::string& source) {
    auto parsed=parse(source,"ownership.th");
    expect(parsed.module.has_value(),"parse before diagnostic: "+(parsed.diagnostics.empty()?std::string{}:parsed.diagnostics[0].format()));
    auto sem=s::analyze(*parsed.module);
    if (!sem.diagnostics.empty()) return sem.diagnostics[0].category;
    expect(sem.module.has_value(),"missing semantic result");
    auto facts=a::analyze(*sem.module);
    if (facts.diagnostics.empty()) return {};
    expect(facts.diagnostics[0].span.begin.line>=1 && facts.diagnostics[0].span.begin.column>=1,"unlocated ownership diagnostic");
    return facts.diagnostics[0].category;
}
std::set<a::ResourceId> resources(const a::ResourceValue& v) {
    auto r=v.resources;
    for (const auto& child:v.elements) { auto x=resources(child); r.insert(x.begin(),x.end()); }
    return r;
}
s::ValueId bound(const s::Function& f,const std::string& name) {
    for (const auto& b:f.body.bindings) if (b.name==name) return b.value;
    throw std::runtime_error("missing fixture binding "+name);
}
void sourceMatrix() {
    struct Fixture { const char* id; const char* source; const char* expected; };
    const Fixture cases[]={
        {"OWN01",R"(fn f()->i64 { let A=[1,2;3,4]; let B=A; return A[0,0]+B[0,0] })",""},
        {"OWN02",R"(fn f()->i64 { let A=[1,2;3,4]; let B=copy(A); return A[0,0]+B[0,0] })",""},
        {"OWN03",R"(fn f()->i64 { let A=[1,2;3,4]; let B=move(A); return B[0,0] })",""},
        {"OWN04",R"(fn f()->i64 { let A=[1,2;3,4]; let B=A; let C=move(A); return B[0,0]+C[0,0] })",""},
        {"OWN05",R"(fn f()->i64 { let A=[1,2;3,4]; let B=move(A); return A[0,0] })","TH006-USE-AFTER-MOVE"},
        {"OWN06",R"(fn f()->i64 { let A=[1,2;3,4]; let B=move(A); let C=move(A); return B[0,0] })","TH006-DOUBLE-MOVE"},
        {"OWN07",R"(fn f()->i64 { let mut A=[1,2;3,4]; let B=move(A); A=[5,6;7,8]; return A[0,0]+B[0,0] })",""},
        {"OWN08",R"(fn f()->i64 { let mut A=[1,2;3,4]; let B=A; A=[5,6;7,8]; return B[0,0] })",""},
        {"OWN09",R"(fn f()->i64 { let A=[1,2;3,4]; let row=A[0,:]; return row[0] })",""},
        {"OWN10",R"(fn f()->i64 { let A=[1,2;3,4]; let T=A.T; return T[0,0] })",""},
        {"OWN11",R"(fn f()->i64 { let A=[1,2;3,4]; let row=A[0]; return row[0] })",""},
        {"OWN12",R"(fn f()->i64 { let mut A=[1,2;3,4]; let row=A[0,:]; touch(borrow mut A); return row[0] })","TH006-LIVE-VIEW-CONFLICT"},
        {"OWN13",R"(fn f()->i64 { let mut A=[1,2;3,4]; let row=A[0,:]; let x=row[0]; touch(borrow mut A); return A[0,0] })",""},
        {"OWN14",R"(fn f()->i64 { let mut A=[1,2;3,4]; let B=A; touch(borrow mut A); return use(B) })","TH006-MUT-BORROW-CONFLICT"},
        {"OWN15",R"(fn f()->i64 { let mut A=[1,2;3,4]; let B=A; use(B); touch(borrow mut A); return A[0,0] })",""},
        {"OWN16",R"(fn f()->i64 { let A=[1,2;3,4]; let mut C=copy(A); touch(borrow mut C); return A[0,0] })",""},
        {"OWN17",R"(fn f(flag: bool)->i64 { let mut A=[1,2;3,4]; if flag { let B=A; use(B) }; touch(borrow mut A); return 0 })",""},
        {"OWN18",R"(fn f(flag: bool)->i64 { let A=[1,2;3,4]; if flag { let X=move(A) }; return A[0,0] })","TH006-MAYBE-MOVED"},
        {"OWN19",R"(fn f(flag: bool)->i64 { let A=[1,2;3,4]; if flag { let X=move(A) } else { let Y=move(A) }; return A[0,0] })","TH006-USE-AFTER-MOVE"},
        {"OWN20",R"(fn f(flag: bool)->i64 { let mut A=[1,2;3,4]; if flag { let X=move(A); A=[5,6;7,8] } else { let Y=move(A); A=[9,10;11,12] }; return A[0,0] })",""},
        {"OWN21",R"(fn f(n: i64)->i64 { let mut A=[1,2;3,4]; let B=A; for i in 0:n { touch(borrow mut A); use(B) }; return 0 })","TH006-MUT-BORROW-CONFLICT"},
        {"OWN22",R"(fn f(n: i64)->i64 { let A=[1,2;3,4]; for i in 0:n { let X=move(A) }; return 0 })","TH006-MAYBE-MOVED"},
        {"OWN23",R"(fn twice(x: Tensor<i64,2>)->Tensor<i64,2> { return x+x }
fn f()->i64 { let A=[1,2;3,4]; let B=twice(A); return A[0,0]+B[0,0] })",""},
        {"OWN24",R"(fn twice(x: Tensor<i64,2>)->Tensor<i64,2> { return x+x }
fn f()->i64 { let A=[1,2;3,4]; let B=twice(A); let C=twice(A); return B[0,0]+C[0,0] })",""},
        {"OWN25",R"(fn f()->i64 { let mut A=[1,2;3,4]; touch(A); return 0 })","TH006-ACCESS-MODE"},
        {"OWN26",R"(fn f()->i64 { let A=[1,2;3,4]; touch(borrow mut A); return 0 })","TH006-MUT-BORROW-IMMUTABLE"},
        {"OWN27",R"(fn f()->i64 { let mut A=[1,2;3,4]; touch(borrow mut A); return A[0,0] })",""},
        {"OWN28",R"(fn f()->Tensor<i64,1> { let A=[1,2;3,4]; return A[0,:] })","TH006-VIEW-RETURN-DEFERRED"},
        {"OWN29",R"(fn f()->i64 { let A=[1,2;3,4]; let Alias=A; let row=Alias[0,:]; let B=move(A); return row[0]+B[0,0] })",""},
        {"OWN30",R"(fn f()->i64 { let A=[1,2;3,4]; let Alias=A; let row=Alias[0,:]; let B=move(Alias); return row[0] })","TH006-VIEW-ROOT-MOVED"},
        {"OWN31",R"(fn f()->i64 { let A=[1,2;3,4]; let pair=(A,1); return A[0,0] })",""},
        {"OWN32",R"(fn f()->i64 { let A=[1,2;3,4]; if false { let B=move(A); use(A) }; return A[0,0] })","TH006-USE-AFTER-MOVE"},
        {"OWN33",R"(fn f(flag: bool)->i64 { let mut A=[1,2;3,4]; let B=A; if flag { while flag { touch(borrow mut A); use(B); break } }; return 0 })","TH006-MUT-BORROW-CONFLICT"},
        {"OWN34",R"(fn f(flag: bool)->i64 { let mut A=[1,2;3,4]; let B=A; if flag { touch(borrow mut A); return 0 }; use(B); return 0 })",""},
        {"OWN35",R"(fn f(flag: bool,n: i64)->i64 { let mut A=[1,2;3,4]; let B=A; for i in 0:n { if flag { continue }; touch(borrow mut A); use(B); break }; return 0 })","TH006-MUT-BORROW-CONFLICT"}
    };
    for (const auto& c:cases) {
        std::string src=helpers; src+=c.source;
        auto got=category(src);
        expect(got==c.expected,std::string(c.id)+": expected "+c.expected+" got "+got);
        if (!got.empty()) continue;
        auto p=compile(src);
        auto audit=a::auditFacts(p.module,p.facts);
        expect(audit.empty(),std::string(c.id)+": audit "+(audit.empty()?std::string{}:audit[0]));
        expect(p.facts.dump()==compile(src).facts.dump(),std::string(c.id)+": nondeterministic facts");
    }
    auto alias=compile("fn f()->i64 { let A=[1,2;3,4]; let B=A; return A[0,0]+B[0,0] }");
    auto fn=alias.module.functions[0];
    auto rs=resources(alias.facts.valueProvenance.at(fn.id).at(bound(fn,"A")));
    expect(rs.size()==1 && rs==resources(alias.facts.valueProvenance.at(fn.id).at(bound(fn,"B"))),"OWN01 ResourceId alias");
    expect(s::dump(alias.module).find("copy")==std::string::npos,"hidden copy for ordinary alias");
    auto copy=compile("fn f()->i64 { let A=[1,2;3,4]; let B=copy(A); return B[0,0] }");
    fn=copy.module.functions[0];
    expect(resources(copy.facts.valueProvenance.at(fn.id).at(bound(fn,"A"))) !=
        resources(copy.facts.valueProvenance.at(fn.id).at(bound(fn,"B"))),"OWN02 copy resource independence");
    expect(copy.facts.valueProvenance.at(fn.id).at(bound(fn,"B")).copySources==
        resources(copy.facts.valueProvenance.at(fn.id).at(bound(fn,"A"))),
        "OWN02 copy lost independent-copy source provenance");
    expect(s::evaluateCall(copy.module,"f",{}).ok,"copy reference observation");
    auto moved=compile("fn f()->i64 { let A=[1,2;3,4]; let B=A; let C=move(A); return B[0,0]+C[0,0] }");
    const auto& mf=moved.module.functions[0];
    auto findBinding=[&](const std::string& name) {
        for (const auto& b:mf.body.bindings) if (b.name==name) return b.id;
        throw std::runtime_error("missing move fixture binding");
    };
    expect(moved.facts.bindingAvailability.at(mf.id).at(findBinding("A")).availability==
        a::BindingFact::Availability::DefinitelyMoved,"move did not consume source binding");
    expect(moved.facts.bindingAvailability.at(mf.id).at(findBinding("B")).availability==
        a::BindingFact::Availability::DefinitelyAvailable,"move consumed surviving alias");
    auto movedObservation=s::evaluateCall(moved.module,"f",{});
    expect(movedObservation.ok && movedObservation.format().find("\"value\":2")!=std::string::npos,
        "move/surviving-alias reference observation failed");
    auto reinit=compile("fn f()->i64 { let mut A=[1,2;3,4]; let B=move(A); A=[5,6;7,8]; return A[0,0] }");
    expect(reinit.facts.bindingAvailability.at(1).at(reinit.module.functions[0].body.bindings[0].id).availability==
        a::BindingFact::Availability::DefinitelyAvailable,"mutable reinitialization did not restore availability");
    auto rebind=compile("fn f()->i64 { let mut A=[1,2;3,4]; let B=A; A=[5,6;7,8]; return B[0,0] }");
    auto rebindObservation=s::evaluateCall(rebind.module,"f",{});
    expect(rebindObservation.ok && rebindObservation.format().find("\"value\":1")!=std::string::npos,
        "rebind mutated the old alias resource in reference observation");
    auto tuple=compile("fn f()->i64 { let A=[1,2;3,4]; let pair=(A,1); return 0 }");
    fn=tuple.module.functions[0];
    expect(resources(tuple.facts.valueProvenance.at(fn.id).at(bound(fn,"A")))==
        resources(tuple.facts.valueProvenance.at(fn.id).at(bound(fn,"pair"))),"OWN31 tuple resource");
    auto copiedTuple=compile("fn f()->i64 { let A=[1,2;3,4]; let pair=(A,1); let dup=copy(pair); return 0 }");
    fn=copiedTuple.module.functions[0];
    expect(resources(copiedTuple.facts.valueProvenance.at(fn.id).at(bound(fn,"pair")))!=
        resources(copiedTuple.facts.valueProvenance.at(fn.id).at(bound(fn,"dup"))),
        "tuple copy lost independent resource semantics");
    auto view=compile("fn f()->i64 { let A=[1,2;3,4]; let row=A[0,:]; return row[0] }");
    fn=view.module.functions[0];
    expect(resources(view.facts.valueProvenance.at(fn.id).at(bound(fn,"A")))==
        resources(view.facts.valueProvenance.at(fn.id).at(bound(fn,"row"))) && !view.facts.views.empty(),"OWN09 view provenance");
    struct Extra { const char* id; const char* source; const char* expected; };
    const Extra extra[]={
        {"ROOT-REBOUND",R"(fn f()->i64 { let mut A=[1,2;3,4]; let row=A[0,:]; A=[5,6;7,8]; return row[0] })","TH006-VIEW-ROOT-REBOUND"},
        {"ROOT-REBOUND-DEAD",R"(fn f()->i64 { let mut A=[1,2;3,4]; let row=A[0,:]; let x=row[0]; A=[5,6;7,8]; return A[0,0] })",""},
        {"OUTER-VIEW-ESCAPE",R"(fn f(flag: bool)->i64 { let mut outer=[0,0]; if flag { let A=[1,2;3,4]; outer=A[0,:] }; return outer[0] })","TH006-VIEW-ESCAPE"},
        {"READ-MODE-LAUNDER",R"(fn f(x: Tensor<i64,2>)->i64 { let mut B=x; touch(borrow mut B); return 0 })","TH006-ACCESS-MODE"},
        {"READ-MODE-COPY",R"(fn f(x: Tensor<i64,2>)->i64 { let mut B=copy(x); touch(borrow mut B); return 0 })",""},
        {"LOOP-REINIT",R"(fn f(n: i64)->i64 { let mut A=[1,2;3,4]; for i in 0:n { let X=move(A); A=[5,6;7,8]; use(A) }; return A[0,0] })",""},
        {"TUPLE-VIEW-RETURN",R"(fn f()->(Tensor<i64,1>,i64) { let A=[1,2;3,4]; return (A[0,:],1) })","TH006-VIEW-RETURN-DEFERRED"},
        {"MUT-RETURN-ESCAPE",R"(fn f(x: borrow mut Tensor<i64,2>)->Tensor<i64,2> { return x })","TH006-ACCESS-MODE"},
        {"VIEW-TEMP-ROOT",R"(fn f()->i64 { let row=([1,2;3,4]+[5,6;7,8])[0,:]; return row[0] })","TH006-VIEW-ROOT-UNKNOWN"}
    };
    for (const auto& c:extra) {
        std::string src=helpers; src+=c.source;
        auto got=category(src);
        expect(got==c.expected,std::string(c.id)+": expected "+c.expected+" got "+got);
    }
    const char* whileMove=R"(fn pred(x: Tensor<i64,2>)->bool { return false }
fn f()->i64 { let A=[1,2;3,4]; while pred(move(A)) { let x=0 }; return A[0,0] })";
    expect(category(whileMove)=="TH006-MAYBE-MOVED","while-condition move lost at loop successor");
    const char* whileRepeated=R"(fn pred(x: Tensor<i64,2>)->bool { return false }
fn f()->i64 { let A=[1,2;3,4]; while pred(move(A)) { let x=0 }; return 0 })";
    expect(category(whileRepeated)=="TH006-MAYBE-MOVED","while-condition move lost across back-edge");
    const char* whileAlias=R"(fn pred(x: Tensor<i64,2>)->bool { return true }
fn touch(x: borrow mut Tensor<i64,2>)->i64 { return 0 }
fn f()->i64 { let mut A=[1,2;3,4]; let B=A; while pred(B) { touch(borrow mut A) }; return 0 })";
    expect(category(whileAlias)=="TH006-MUT-BORROW-CONFLICT","while-condition future alias use lost");
    expect(category(R"(fn id(x: Tensor<i64,1>)->Tensor<i64,1>{return x}
fn f()->Tensor<i64,1>{let A=[1,2;3,4];let row=A[0,:];return id(row)})")==
        "TH006-VIEW-RETURN-DEFERRED","call alias result dropped caller view root on return");
    expect(category(R"(fn id(x: Tensor<i64,1>)->Tensor<i64,1>{return x}
fn touch(x: borrow mut Tensor<i64,2>)->i64{return 0}
fn f()->i64{let mut A=[1,2;3,4];let row=A[0,:];let row2=id(row);touch(borrow mut A);return row2[0]})")==
        "TH006-LIVE-VIEW-CONFLICT","call alias result dropped caller view root at mutable call");
    expect(category(R"(fn id(x: Tensor<i64,1>)->Tensor<i64,1>{return x}
fn f()->i64{let A=[1,2;3,4];let Alias=A;let row=Alias[0,:];let row2=id(row);let B=move(A);return row2[0]})").empty(),
        "call alias result incorrectly invalidated by different alias move");
    expect(category(R"(fn touch(x: borrow mut Tensor<i64,2>)->i64{return 0}
fn f()->(Tensor<i64,2>,i64){let mut A=[1,2;3,4];let pair=(A,1);touch(borrow mut A);return pair})")==
        "TH006-MUT-BORROW-CONFLICT","live tuple alias did not block mutable borrow");
    expect(category(R"(fn touch(x: borrow mut Tensor<i64,2>)->i64{return 0}
fn f()->i64{let mut A=[1,2;3,4];let pair=(A,1);touch(borrow mut A);return 0})").empty(),
        "dead tuple alias incorrectly blocked mutable borrow");
    expect(category("fn f(n: i64)->i64 { for i in 0:n { let x=move(i); let y=i }; return 0 }")==
        "TH006-USE-AFTER-MOVE","loop induction binding move was not tracked");
    expect(category("fn f(x: borrow mut i64)->i64 { return 0 }")=="TH006-ACCESS-MODE",
        "scalar mutable parameter accepted without resource contract");
    expect(category(R"(fn take(x: move Tensor<i64,2>)->i64 { return x[0,0] }
fn f()->i64 { let A=[1,2;3,4]; return take(A) })")=="TH006-ACCESS-MODE",
        "Consume parameter accepted an implicit move");
    expect(category(R"(fn take(x: move Tensor<i64,2>)->i64 { return x[0,0] }
fn f()->i64 { let A=[1,2;3,4]; return take(move(A)) })").empty(),
        "explicit consuming parameter call rejected");
    auto readReturn=compile(R"(fn id(x: Tensor<i64,2>)->Tensor<i64,2> { return x }
fn touch(x: borrow mut Tensor<i64,2>)->i64 { return 0 }
fn f()->i64 { let mut A=[1,2;3,4]; let B=id(A); touch(borrow mut A); return B[0,0] })");
    expect(!readReturn.facts.ok() && readReturn.facts.diagnostics[0].category=="TH006-MUT-BORROW-CONFLICT",
        "read-only alias-valued call result lost provenance");
    auto noHiddenCopy=compile(R"(fn touch(x: borrow mut Tensor<i64,2>)->i64 { return 0 }
fn f()->i64 { let mut A=[1,2;3,4]; let B=A; touch(borrow mut A); return B[0,0] })");
    expect(!noHiddenCopy.facts.ok() && s::dump(noHiddenCopy.module).find("copy")==std::string::npos,
        "illegal mutable access was rescued by hidden copy");
    auto legalBorrow=compile(R"(fn touch(x: borrow mut Tensor<i64,2>)->i64 { return 0 }
fn f()->i64 { let mut A=[1,2;3,4]; touch(borrow mut A); return A[0,0] })");
    expect(legalBorrow.facts.ok() && s::dump(legalBorrow.module).find("copy")==std::string::npos,
        "mutable borrow introduced hidden copy");
    auto borrowObservation=s::evaluateCall(legalBorrow.module,"f",{});
    expect(borrowObservation.ok && borrowObservation.format().find("\"value\":1")!=std::string::npos,
        "mutable-borrow call did not preserve reference observation");
    auto tupleReturn=compile(R"(fn pair(x: Tensor<i64,2>)->(Tensor<i64,2>,i64) { return (x,1) }
fn f()->i64 { let A=[1,2;3,4]; let out=pair(A); return 0 })");
    const auto& caller=tupleReturn.module.functions[1];
    auto relation=tupleReturn.facts.valueProvenance.at(caller.id).at(bound(caller,"out"));
    expect(relation.elements.size()==2 && resources(relation.elements[0]).size()>=1 &&
        resources(relation.elements[1]).empty(),"tuple call result lost per-element provenance");
}
void effectMatrix() {
    struct Fixture { const char* id; const char* source; const char* name; a::EffectSet expected; };
    const Fixture cases[]={
        {"EFF01","fn f()->i64 { return 1+2 }","f",0},
        {"EFF02","fn f(x: Tensor<i64,2>,i: i64)->i64 { return x[i,0] }","f",1},
        {"EFF03",R"(fn trap(x: Tensor<i64,2>,i: i64)->i64 { return x[i,0] }
fn f(x: Tensor<i64,2>,i: i64)->i64 { return trap(x,i) })","f",1},
        {"EFF04","fn f(x: borrow mut Tensor<i64,2>)->i64 { return 0 }","f",2},
        {"EFF05",R"(fn touch(x: borrow mut Tensor<i64,2>)->i64 { return 0 }
fn f()->i64 { let mut A=[1,2;3,4]; touch(borrow mut A); return 0 })","f",2},
        {"EFF06","fn f(x: Tensor<i64,2>)->i64 { return 0 }","f",0},
        {"EFF07",R"(fn touch(x: borrow mut Tensor<i64,2>)->i64 { return 0 }
fn f(flag: bool)->i64 { let mut A=[1,2;3,4]; if flag { touch(borrow mut A) }; return 0 })","f",2},
        {"EFF08",R"(fn touch(x: borrow mut Tensor<i64,2>)->i64 { return 0 }
fn f(n: i64)->i64 { let mut A=[1,2;3,4]; for i in 0:n { touch(borrow mut A) }; return 0 })","f",2},
        {"EFF09","fn f()->i64 { let A=[1,2;3,4]; let B=copy(A); let C=move(A); return 0 }","f",0}
    };
    for (const auto& c:cases) {
        auto p=compile(c.source);
        expect(p.facts.ok(),std::string(c.id)+": ownership rejection");
        const s::Function* fn=nullptr;
        for (const auto& f:p.module.functions) if (f.name==c.name) fn=&f;
        expect(fn!=nullptr,std::string(c.id)+": missing function");
        auto got=p.facts.functionEffects.at(fn->id).kinds;
        expect((got&c.expected)==c.expected,std::string(c.id)+": missing effect");
        if (!c.expected) expect(got==0,std::string(c.id)+": pure function gained effect");
        if (c.expected==2) expect(!p.facts.functionEffects.at(fn->id).pureTensorCandidate,
            std::string(c.id)+": mutation marked region candidate");
    }
    expect(static_cast<a::EffectSet>(a::EffectKind::RNG)==4 &&
        static_cast<a::EffectSet>(a::EffectKind::IO)==8 &&
        static_cast<a::EffectSet>(a::EffectKind::Transfer)==16 &&
        static_cast<a::EffectSet>(a::EffectKind::Async)==32,"EFF10 future effect kinds");
    auto recursion=compile(R"(fn f(x: Tensor<i64,2>,i: i64)->i64 { return g(x,i) }
fn g(x: Tensor<i64,2>,i: i64)->i64 { let y=x[i,0]; return f(x,i) })");
    expect(recursion.facts.functionEffects.at(1).kinds & static_cast<a::EffectSet>(a::EffectKind::MayTrap),
        "recursive MayTrap fixed point did not propagate to caller");
    auto controlled=compile("fn f(x: Tensor<i64,2>)->i64 { if false { return x[100,0] }; return 0 }");
    expect(controlled.facts.functionEffects.at(1).kinds & static_cast<a::EffectSet>(a::EffectKind::MayTrap),
        "branch MayTrap omitted from function summary");
    s::RuntimeTensor tensor{s::TypeKind::I64,{2,2},{1,2,3,4}};
    auto observation=s::evaluateCall(controlled.module,"f",{s::RuntimeValue{tensor}});
    expect(observation.ok && observation.format().find("\"value\":0")!=std::string::npos,
        "untaken branch Check executed or effect position was changed");
}
void adversarial() {
    auto p=compile("fn f()->i64 { let A=[1,2;3,4]; let B=copy(A); return 0 }");
    expect(a::auditFacts(p.module,p.facts).empty(),"valid facts rejected");
    auto x=p.facts;
    x.valueProvenance.at(1).at(bound(p.module.functions[0],"A")).resources={999999};
    expect(!a::auditFacts(p.module,x).empty(),"unknown resource accepted");
    x=p.facts;
    x.valueProvenance.at(1).at(bound(p.module.functions[0],"B")).resources=
        resources(x.valueProvenance.at(1).at(bound(p.module.functions[0],"A")));
    expect(!a::auditFacts(p.module,x).empty(),"copy sharing source accepted");
    auto duplicate=compile("fn f()->i64 { let A=[1,2;3,4]; let B=[5,6;7,8]; return 0 }");
    x=duplicate.facts;
    x.valueProvenance.at(1).at(bound(duplicate.module.functions[0],"B")).resources=
        resources(x.valueProvenance.at(1).at(bound(duplicate.module.functions[0],"A")));
    expect(!a::auditFacts(duplicate.module,x).empty(),"duplicate fresh ResourceId accepted");
    auto view=compile("fn f()->i64 { let A=[1,2;3,4]; let row=A[0,:]; return 0 }");
    x=view.facts; x.views.begin()->second.roots={999};
    expect(!a::auditFacts(view.module,x).empty(),"unknown view root accepted");
    auto moved=compile("fn f()->i64 { let A=[1,2;3,4]; let B=move(A); return 0 }");
    x=moved.facts; x.moves[0].availabilityAfter=a::BindingFact::Availability::DefinitelyAvailable;
    expect(!a::auditFacts(moved.module,x).empty(),"moved binding marked available accepted");
    auto borrowed=compile(R"(fn touch(x: borrow mut Tensor<i64,2>)->i64{return 0}
fn f()->i64{let mut A=[1,2;3,4];touch(borrow mut A);return 0})");
    x=borrowed.facts; x.mutableBorrows[0].liveConflicts.insert(999);
    expect(!a::auditFacts(borrowed.module,x).empty(),"borrow with live alias accepted");
    x=borrowed.facts; x.mutableBorrows[0].resources.clear();
    expect(!a::auditFacts(borrowed.module,x).empty(),"scalar mutation accepted");
    x=borrowed.facts; x.functionEffects.at(1).kinds=0;
    expect(!a::auditFacts(borrowed.module,x).empty(),"inconsistent effect summary accepted");
    auto malformed=borrowed.module; malformed.functions[0].parameters[0].access=static_cast<s::AccessMode>(99);
    expect(!s::verify(malformed).ok,"impossible access mode accepted");
    auto escape=compile(R"(fn touch(x: borrow mut Tensor<i64,2>)->i64{return 0}
fn f()->Tensor<i64,2>{let mut A=[1,2;3,4];touch(borrow mut A);let B=[5,6;7,8];return B})");
    auto forged=escape.module;
    s::ValueId borrow=0;
    for (const auto& step:forged.functions[1].body.steps)
        if (const auto* i=std::get_if<s::Instruction>(&step); i && i->op==s::Op::MutableBorrow) borrow=i->id;
    expect(borrow!=0,"missing forged-borrow seed");
    for (auto& step:forged.functions[1].body.steps)
        if (auto* f=std::get_if<s::Flow>(&step); f && f->kind==s::Flow::Kind::Return) f->value=borrow;
    forged.functions[1].body.returned=borrow;
    expect(s::verify(forged).ok,"forged escaping borrow did not preserve semantic IR contracts");
    auto rejected=a::analyze(forged);
    expect(!rejected.ok() && rejected.diagnostics[0].category=="TH006-ACCESS-MODE",
        "escaping mutable borrow survived ownership IR audit");
}
}
int main() {
    try { sourceMatrix(); effectMatrix(); adversarial(); std::cout<<"PASS TH-006 ownership/effects/adversarial\n"; }
    catch (const std::exception& e) { std::cerr<<"FAIL TH-006: "<<e.what()<<'\n'; return 1; }
    return 0;
}
