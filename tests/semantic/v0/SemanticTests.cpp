#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Verifier.hpp"
#include "semantic/v0/Evaluator.hpp"
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace thiran::v0;
using namespace thiran::v0::semantic;
using SemanticModule = thiran::v0::semantic::Module;
namespace {
void require(bool condition,const std::string& message) { if (!condition) throw std::runtime_error(message); }
SemanticModule valid(const std::string& id,const std::string& source) {
    auto syntax=parse(source,id+".th");
    require(syntax.module.has_value() && syntax.diagnostics.empty(),id+": syntax failed");
    auto result=analyze(*syntax.module);
    require(result.module.has_value() && result.diagnostics.empty(),id+": semantic failed " + (result.diagnostics.empty()?"":result.diagnostics[0].format()));
    auto verification=verify(*result.module);
    require(verification.ok,id+": verification failed " + (verification.errors.empty()?"":verification.errors[0]));
    auto repeat=analyze(*syntax.module);
    require(repeat.module && dump(*repeat.module)==dump(*result.module),id+": nondeterministic semantic dump");
    return std::move(*result.module);
}
void value(const std::string& id,const std::string& source,const std::string& binding,const std::string& expected) {
    auto module=valid(id,source);
    auto observation=evaluateBinding(module,binding);
    require(observation.format()==expected,id+": unexpected observation " + observation.format());
}
void staticError(const std::string& id,const std::string& source,const std::string& category) {
    auto syntax=parse(source,id+".th");
    require(syntax.module && syntax.diagnostics.empty(),id+": syntax unexpectedly failed");
    auto result=analyze(*syntax.module);
    require(!result.module && !result.diagnostics.empty(),id+": semantic error missing");
    require(result.diagnostics[0].category==category,id+": wrong category " + result.diagnostics[0].format());
    require(result.diagnostics[0].source==id+".th" && result.diagnostics[0].span.begin.line>=1 &&
        result.diagnostics[0].span.begin.column>=1,id+": missing provenance");
}
void runtime(const std::string& id,const std::string& source,const std::string& function,
             std::vector<RuntimeValue> args,const std::string& expected,const std::string& checkText) {
    auto module=valid(id,source);
    require(dump(module).find(checkText)!=std::string::npos,id+": explicit runtime check absent");
    auto result=evaluateCall(module,function,args);
    require(result.format()==expected,id+": wrong runtime result " + result.format());
}
RuntimeValue matrix(std::vector<std::int64_t> shape,std::vector<std::int64_t> values) {
    return RuntimeValue{RuntimeTensor{TypeKind::I64,std::move(shape),std::move(values)}};
}
Instruction integerInstruction(ValueId id,std::int64_t value) {
    Instruction i; i.id=id; i.op=Op::Integer; i.type=scalar(TypeKind::I64); i.integer=value;
    i.span={{0,0,1,1},{0,0,1,1}}; return i;
}
SemanticModule malformedBase() {
    SemanticModule m; m.source="manual"; m.initializer.id=1; m.initializer.terminated=true;
    Function f; f.id=1; f.name="manual"; f.result=scalar(TypeKind::I64);
    f.body.id=1;
    f.body.steps.push_back(integerInstruction(1,7)); f.body.returned=1; f.body.terminated=true;
    m.functions.push_back(std::move(f)); return m;
}
void malformed(const std::string& id,const std::function<void(SemanticModule&)>& mutate) {
    auto m=malformedBase(); mutate(m);
    require(!verify(m).ok,id+": verifier accepted malformed manual IR");
}
void verifierTests() {
    require(verify(malformedBase()).ok,"manual verifier baseline invalid");
    malformed("VIR01",[](SemanticModule& m){ m.functions[0].body.steps.push_back(integerInstruction(1,8)); });
    malformed("VIR02",[](SemanticModule& m){ Instruction i; i.id=2; i.op=Op::Add; i.type=scalar(TypeKind::I64); i.operands={1,99}; m.functions[0].body.steps.push_back(i); });
    malformed("VIR03",[](SemanticModule& m){ Instruction i; i.id=2; i.op=Op::Add; i.type=scalar(TypeKind::I64); i.operands={1,3}; m.functions[0].body.steps.push_back(i); m.functions[0].body.steps.push_back(integerInstruction(3,8)); });
    malformed("VIR04",[](SemanticModule& m){ std::get<Instruction>(m.functions[0].body.steps[0]).type=scalar(TypeKind::Bool); });
    malformed("VIR05",[](SemanticModule& m){ auto& i=std::get<Instruction>(m.functions[0].body.steps[0]); i.op=Op::TensorLiteral; i.type=tensor(scalar(TypeKind::I64),2); i.shape.extents={2}; });
    malformed("VIR06",[](SemanticModule& m){ auto& i=std::get<Instruction>(m.functions[0].body.steps[0]); i.op=Op::Call; i.callee=42; i.integer.reset(); });
    malformed("VIR07",[](SemanticModule& m){ auto& i=std::get<Instruction>(m.functions[0].body.steps[0]); i.op=Op::Call; i.callee=1; i.integer.reset(); i.type=scalar(TypeKind::Bool); });
    malformed("VIR08",[](SemanticModule& m){ m.functions[0].result=scalar(TypeKind::Bool); });
    malformed("VIR09",[](SemanticModule& m){ m.functions[0].body.terminated=false; });
    malformed("VIR10",[](SemanticModule& m){ Check c; c.kind=CheckKind::Bounds; c.operands={1,1}; c.failureId="TH-SPEC-BOUNDS"; m.functions[0].body.steps.insert(m.functions[0].body.steps.begin(),c); });
}
SemanticModule controlManual() {
    auto m=malformedBase(); auto& b=m.functions[0].body; b.id=1;
    Instruction condition; condition.id=2; condition.op=Op::Boolean; condition.boolean=true; condition.type=scalar(TypeKind::Bool);
    condition.span={{0,0,1,1},{0,0,1,1}};
    b.steps.push_back(condition);
    Structured s; s.kind=Structured::Kind::If; s.condition=2; s.span={{0,0,1,1},{0,0,1,1}};
    s.thenBlock=std::make_shared<Block>(); s.thenBlock->id=2; s.thenBlock->parent=1; s.thenBlock->terminated=true;
    s.elseBlock=std::make_shared<Block>(); s.elseBlock->id=3; s.elseBlock->parent=1; s.elseBlock->terminated=true;
    b.steps.push_back(s); return m;
}
void controlVerifierTests() {
    require(verify(controlManual()).ok,"manual structured verifier base invalid");
    auto bad=[](const std::string& id,const std::function<void(SemanticModule&)>& change) {
        auto m=controlManual(); change(m); require(!verify(m).ok,id+": malformed structured IR accepted");
    };
    bad("CFV01",[](SemanticModule& m){ std::get<Structured>(m.functions[0].body.steps[2]).condition=1; });
    bad("CFV02",[](SemanticModule& m){ m.functions[0].body.steps.push_back(Flow{Flow::Kind::Break,{},{}}); });
    bad("CFV03",[](SemanticModule& m){ m.functions[0].body.steps.push_back(Flow{Flow::Kind::Continue,{},{}}); });
    bad("CFV04",[](SemanticModule& m){ auto& s=std::get<Structured>(m.functions[0].body.steps[2]); s.thenBlock->steps.push_back(integerInstruction(3,8)); Instruction i; i.id=4; i.op=Op::Add; i.type=scalar(TypeKind::I64); i.operands={1,3}; m.functions[0].body.steps.push_back(i); });
    bad("CFV05",[](SemanticModule& m){ auto& s=std::get<Structured>(m.functions[0].body.steps[2]); s.thenBlock->steps.push_back(BindingWrite{1,1,true,{}}); });
    bad("CFV06",[](SemanticModule& m){ Structured s; s.kind=Structured::Kind::ForRange; s.start=1; s.end=1; s.induction=1; s.bodyBlock=std::make_shared<Block>(); s.bodyBlock->id=4; s.bodyBlock->parent=1; s.bodyBlock->terminated=true; s.bodyBlock->steps.push_back(BindingWrite{77,1,false,{}}); m.functions[0].body.steps.push_back(s); });
    bad("CFV07",[](SemanticModule& m){ auto& s=std::get<Structured>(m.functions[0].body.steps[2]); s.thenBlock->steps.push_back(Flow{Flow::Kind::Return,2,{}}); });
    bad("CFV08",[](SemanticModule& m){ auto& s=std::get<Structured>(m.functions[0].body.steps[2]); Check c; c.kind=CheckKind::Bounds; c.operands={1,1}; c.failureId="TH-SPEC-BOUNDS"; s.thenBlock->steps.push_back(c); });
    bad("CFV09",[](SemanticModule& m){ Structured s; s.kind=Structured::Kind::ForRange; s.start=1; s.end=1; s.induction=1; m.functions[0].body.steps.push_back(s); });
    bad("CFV10",[](SemanticModule& m){ auto& s=std::get<Structured>(m.functions[0].body.steps[2]); s.elseBlock->id=2; });
}
void scalarCall(const std::string& id,const std::string& source,const std::string& name,
                std::vector<RuntimeValue> args,std::int64_t expected) {
    auto m=valid(id,source); auto result=evaluateCall(m,name,args);
    require(result.format()=="{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":"+std::to_string(expected)+"}",
        id+": scalar observation mismatch "+result.format());
}
void controlFlowTests() {
    const std::string choose="fn choose(flag:bool)->i64{let mut x=0; if flag { x=10 } else { x=20 }; return x}";
    scalarCall("CF01-T",choose,"choose",{RuntimeValue{true}},10);
    scalarCall("CF01-F",choose,"choose",{RuntimeValue{false}},20);
    const std::string partial="fn partial(flag:bool)->i64{let mut x=0; if flag { x=10 }; return x}";
    scalarCall("CF02-T",partial,"partial",{RuntimeValue{true}},10);
    scalarCall("CF02-F",partial,"partial",{RuntimeValue{false}},0);
    staticError("CF03","fn f(flag:bool)->i64{if flag { let x=1 }; return x}","TH005-UNDEFINED-NAME");
    scalarCall("CF04-T",choose,"choose",{RuntimeValue{true}},10);
    scalarCall("CF04-F",choose,"choose",{RuntimeValue{false}},20);
    const std::string nested="fn nested(a:bool,b:bool)->i64{if a { if b { return 1 } else { return 2 } } else { return 3 }}";
    scalarCall("CF05-A",nested,"nested",{RuntimeValue{true},RuntimeValue{true}},1);
    scalarCall("CF05-B",nested,"nested",{RuntimeValue{true},RuntimeValue{false}},2);
    scalarCall("CF05-C",nested,"nested",{RuntimeValue{false},RuntimeValue{true}},3);
    const std::string sum="fn sum_to(n:i64)->i64{let mut total=0; for i in 0:n { total=total+i }; return total}";
    scalarCall("CF06",sum,"sum_to",{RuntimeValue{std::int64_t{5}}},10);
    scalarCall("CF07",sum,"sum_to",{RuntimeValue{std::int64_t{0}}},0);
    scalarCall("CF08",sum,"sum_to",{RuntimeValue{std::int64_t{4}}},6);
    scalarCall("CF-R",sum,"sum_to",{RuntimeValue{std::int64_t{-5}}},0);
    const std::string once="fn once(flag:bool)->i64{let mut running=flag; let mut value=0; while running { value=7; running=false }; return value}";
    scalarCall("CF09-T",once,"once",{RuntimeValue{true}},7);
    scalarCall("CF09-F",once,"once",{RuntimeValue{false}},0);
    scalarCall("CF10","fn f(n:i64)->i64{let mut x=0; for i in 0:n { x=x+1; break }; return x}","f",{RuntimeValue{std::int64_t{5}}},1);
    scalarCall("CF11","fn f(n:i64)->i64{let mut x=0; for i in 0:n { if true { continue }; x=x+1 }; return x}","f",{RuntimeValue{std::int64_t{5}}},0);
    scalarCall("CF12","fn f(n:i64)->i64{let mut x=0; for i in 0:n { for j in 0:3 { break }; x=x+1 }; return x}","f",{RuntimeValue{std::int64_t{4}}},4);
    scalarCall("CF13-T","fn f(flag:bool)->i64{if flag { return 1 }; return 2}","f",{RuntimeValue{true}},1);
    scalarCall("CF13-F","fn f(flag:bool)->i64{if flag { return 1 }; return 2}","f",{RuntimeValue{false}},2);
    scalarCall("CF14","fn f(n:i64)->i64{for i in 0:n { return i }; return 99}","f",{RuntimeValue{std::int64_t{5}}},0);
    const std::string guarded="fn safe(flag:bool,x:Tensor<i64,2>)->i64{if flag { return x[999,0] }; return 0}";
    auto safe=valid("CF15",guarded);
    require(dump(safe).find("if %")<dump(safe).find("check bounds"),"CF15 Check escaped branch region");
    require(evaluateCall(safe,"safe",{RuntimeValue{false},matrix({1,1},{7})}).format()==
        "{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":0}","CF15 untaken Check trapped");
    require(evaluateCall(safe,"safe",{RuntimeValue{true},matrix({1,1},{7})}).errorId=="TH-SPEC-BOUNDS","CF16 taken Check did not trap");
    const std::string tensor="fn f(flag:bool,A:Tensor<i64,2>)->Tensor<i64,2>{if flag { return A+A } else { return A*A }}";
    auto tm=valid("CF17",tensor);
    require(evaluateCall(tm,"f",{RuntimeValue{true},matrix({2,2},{1,2,3,4})}).format()==
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[2,4,6,8]}","CF17 then tensor wrong");
    require(evaluateCall(tm,"f",{RuntimeValue{false},matrix({2,2},{1,2,3,4})}).format()==
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[7,10,15,22]}","CF17 else tensor wrong");
    scalarCall("CF18","fn f(A:Tensor<i64,1>,n:i64)->i64{let mut x=0; for i in 0:n { x=x+A[i] }; return x}","f",
        {matrix({3},{2,3,4}),RuntimeValue{std::int64_t{3}}},9);
    auto loopOverflow=valid("CF18-OVERFLOW","fn f(A:Tensor<i64,1>)->i64{let mut x=0; for i in 0:2 { x=x+A[i] }; return x}");
    require(evaluateCall(loopOverflow,"f",{matrix({2},{std::numeric_limits<std::int64_t>::max(),1})}).errorId=="TH-SPEC-I64-OVERFLOW",
        "loop-carried accumulation lost checked i64 overflow");
    auto det=valid("CF19",sum); require(dump(det)==dump(valid("CF19-repeat",sum)),"CF19 structured dump differs");
    staticError("CF-D01","fn f(x:i64)->i64{if x { return 1 }; return 0}","TH005C-CONDITION-TYPE");
    staticError("CF-D02","fn f(x:i64)->i64{while x { break }; return 0}","TH005C-CONDITION-TYPE");
    staticError("CF-D03","fn f(x:bool)->i64{for i in x:4 { break }; return 0}","TH005C-RANGE-TYPE");
    staticError("CF-D04","fn f(x:bool)->i64{for i in 0:x { break }; return 0}","TH005C-RANGE-TYPE");
    staticError("CF-D05","fn f()->i64{break; return 0}","TH005C-LOOP-CONTROL");
    staticError("CF-D06","fn f()->i64{continue; return 0}","TH005C-LOOP-CONTROL");
    staticError("CF-D07","fn f()->i64{if true { let x=1; let x=2 }; return 0}","TH005-DUPLICATE-BINDING");
    staticError("CF-D08","fn f()->i64{let x=0; if true { x=1 }; return x}","TH005-IMMUTABLE-REBIND");
    staticError("CF-D09","fn f()->i64{let mut x=0; for i in 0:1 { x=true }; return x}","TH005-REBIND-TYPE");
    staticError("CF-D10","fn f()->i64{if true { let x=y }; return 0}","TH005-UNDEFINED-NAME");
    staticError("CF-D11","fn f(A:Tensor<i64,2>)->i64{for row in A { break }; return 0}","TH005C-ITERABLE-FOR-DEFERRED");
    staticError("CF-D12","fn f()->i64{for i in 0:2 { break; let x=1 }; return 0}","TH005-AFTER-RETURN");
    staticError("CF-D13","fn f()->i64{return 1; return 2}","TH005-AFTER-RETURN");
    scalarCall("CF-SHADOW","fn f(flag:bool)->i64{let x=1; if flag { let x=2; return x }; return x}","f",{RuntimeValue{false}},1);
    scalarCall("CF-ALIAS","fn f(flag:bool)->i64{let mut x=1; let y=x; if flag { x=2 }; return y}","f",{RuntimeValue{true}},1);
    auto infinite=valid("CF-LIMIT","fn f()->i64{while true { }; return 0}");
    require(evaluateCall(infinite,"f",{}).errorId=="TH005C-STEP-LIMIT","infinite while did not stop at evaluator guard");
}
void sourceTests() {
    value("S01","let x = 4 + 5","x","{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":9}");
    value("S02","let A = [1,2;3,4]","A","{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[1,2,3,4]}");
    value("S03","let A = [1,2;3,4]\nlet B = [5,6;7,8]\nlet C = A+B","C","{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[6,8,10,12]}");
    value("S04","let A = [1,2;3,4]\nlet C = A+10","C","{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[11,12,13,14]}");
    value("S05","let A = [1;2]\nlet B = [10,20,30]\nlet C = A+B","C","{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,3],\"values\":[11,21,31,12,22,32]}");
    staticError("S06","let A=[1,2]\nlet B=[3,4,5]\nlet C=A+B","TH-SPEC-BROADCAST");
    value("S07","let A=[1,2;3,4]\nlet x=A[1,0]","x","{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":3}");
    staticError("S08","let A=[1,2;3,4]\nlet x=A[2,0]","TH-SPEC-BOUNDS");
    value("S09","let A=[1,2,3,4]\nlet B=A[1:4:2]","B","{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2],\"values\":[2,4]}");
    staticError("S10","let A=[1,2,3]\nlet B=A[0:3:0]","TH-SPEC-SLICE");
    value("S11","let A=[1,2;3,4]\nlet B=[5,6;7,8]\nlet C=A*B","C","{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[19,22,43,50]}");
    staticError("S12","let A=[1,2;3,4]\nlet B=[5,6,7;8,9,10]\nlet C=B*A","TH-SPEC-SHAPE");
    value("S13","let A=[1,2;3,4]\nlet C=sum(A,0)","C","{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2],\"values\":[4,6]}");
    value("S14","let A=[1,2;3,4]\nlet C=sum(A,1)","C","{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2],\"values\":[3,7]}");
    auto overflow=valid("S15","let A=[9223372036854775807,1]\nlet C=sum(A,0)");
    require(evaluateBinding(overflow,"C").errorId=="TH-SPEC-I64-OVERFLOW","S15 reduction overflow missing");
    runtime("S16","fn first(x: Tensor<i64,2>, i: i64) -> i64 { return x[i,0] }","first",
        {matrix({2,2},{1,2,3,4}),RuntimeValue{std::int64_t{1}}},"{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":3}","check bounds");
    auto first=valid("S16-invalid","fn first(x: Tensor<i64,2>, i: i64) -> i64 { return x[i,0] }");
    require(evaluateCall(first,"first",{matrix({2,2},{1,2,3,4}),RuntimeValue{std::int64_t{2}}}).errorId=="TH-SPEC-BOUNDS","S16 runtime bounds failure");
    runtime("S17-b","fn add(a: Tensor<i64,2>, b: Tensor<i64,2>) -> Tensor<i64,2> { return a+b }","add",
        {matrix({2,1},{1,2}),matrix({1,3},{10,20,30})},
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,3],\"values\":[11,21,31,12,22,32]}","check broadcast");
    auto add=valid("S17-bad","fn add(a: Tensor<i64,2>, b: Tensor<i64,2>) -> Tensor<i64,2> { return a+b }");
    require(evaluateCall(add,"add",{matrix({2,2},{1,2,3,4}),matrix({1,3},{1,2,3})}).errorId=="TH-SPEC-BROADCAST","runtime broadcast failure");
    auto mm=valid("S17-mm","fn mm(a: Tensor<i64,2>, b: Tensor<i64,2>) -> Tensor<i64,2> { return a*b }");
    require(dump(mm).find("check matmul_shape")!=std::string::npos,"runtime matmul check missing");
    require(evaluateCall(mm,"mm",{matrix({2,2},{1,2,3,4}),matrix({3,1},{1,2,3})}).errorId=="TH-SPEC-SHAPE","runtime matmul shape failure");
    value("S18","fn caller(x: i64) -> i64 { return later(x) }\nfn later(y: i64) { return y+1 }\nlet z=caller(4)","z",
        "{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":5}");
    auto alias=valid("S19","let A=[1,2;3,4]\nlet B=A\nlet C=A+B");
    require(dump(alias).find("bind A=%")!=std::string::npos && dump(alias).find("bind B=%")!=std::string::npos &&
        dump(alias).find("copy")==std::string::npos && dump(alias).find("move")==std::string::npos,"immutable alias modeled as copy/move");
    require(evaluateBinding(alias,"C").ok,"immutable alias invalidated source");
    value("S20","fn pair(x: i64) -> (i64,i64) { let y=x+1; return (y,x) }\nlet p=pair(3)","p",
        "{\"status\":\"ok\",\"kind\":\"tuple\",\"values\":[{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":4},{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":3}]}");
}
void diagnosticTests() {
    auto absoluteSyntax=parse("let x=1","/tmp/machine-specific/private.th");
    require(absoluteSyntax.module && absoluteSyntax.diagnostics.empty(),"absolute-source syntax probe failed");
    auto absoluteIr=analyze(*absoluteSyntax.module);
    require(absoluteIr.module && dump(*absoluteIr.module).find("/tmp/machine-specific")==std::string::npos,
        "developer IR dump leaked absolute machine path");
    auto represented=valid("TYPE-TABLE","fn bytes(x:Buffer<u8>)->Buffer<u8>{return x}");
    require(represented.functions[0].parameters[0].type.kind==TypeKind::Buffer &&
        !executableType(represented.functions[0].result),"recognized Buffer type confused with executable type");
    staticError("D01","let x=y","TH005-UNDEFINED-NAME");
    staticError("D02","let x=1\nlet x=2","TH005-DUPLICATE-BINDING");
    staticError("D03","fn f() -> i64 { let x=1; x=2; return x }","TH005-IMMUTABLE-REBIND");
    staticError("D04","fn f() -> i64 { let mut x=1; x=true; return x }","TH005-REBIND-TYPE");
    staticError("D05","fn f(x: State) -> i64 { return 1 }","TH005-UNKNOWN-TYPE");
    staticError("D06","fn f(x: Tensor<i64,999999999999999999999>) -> i64 { return 1 }","TH005-TENSOR-RANK");
    staticError("D07","let x=9223372036854775808","TH005-INTEGER-RANGE");
    staticError("D08","let x=true+1","TH005-OPERAND-TYPE");
    staticError("D09","let A=[1,2;3,4]\nlet B=[1,2,3;4,5,6]\nlet x=B*A","TH-SPEC-SHAPE");
    staticError("D10","let A=[1,2]\nlet x=A[-1]","TH-SPEC-BOUNDS");
    staticError("D11","let A=[1,2]\nlet x=A[0:2:-1]","TH-SPEC-SLICE");
    staticError("D12","let x=f(1)","TH005-UNKNOWN-FUNCTION");
    staticError("D13","fn f(x:i64)->i64{return x}\nlet y=f()","TH005-ARITY");
    staticError("D14","fn f(x:i64)->i64{return x}\nlet y=f(true)","TH005-ARGUMENT-TYPE");
    staticError("D15","fn f()->bool{return 1}","TH005-RETURN-TYPE");
    staticError("D16","fn f()->i64{let x=1}","TH005-MISSING-RETURN");
    staticError("D17","import \"x.th\" as x","TH005-IMPORT-STAGE");
    staticError("D18","let x=4/2","TH005-DIVISION-UNSPECIFIED");
    staticError("D19","fn f(x: Tensor<i64,1>, axis:i64)->i64{return sum(x,axis)}","TH005-DYNAMIC-AXIS");
    staticError("D20","fn f(x:i64){return f(x)}","TH005-INFERENCE-CYCLE");
    value("I64MIN","let x=-9223372036854775808","x","{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":-9223372036854775808}");
    value("REBIND","fn f()->i64{let mut x=1; x=2; return x}\nlet y=f()","y",
        "{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":2}");
}
void adversarialTests() {
    auto add=valid("OVERFLOW","fn add(x:i64,y:i64)->i64{return x+y}");
    require(evaluateCall(add,"add",{RuntimeValue{std::numeric_limits<std::int64_t>::max()},RuntimeValue{std::int64_t{1}}}).errorId=="TH-SPEC-I64-OVERFLOW","max add overflow");
    auto sub=valid("UNDERFLOW","fn sub(x:i64,y:i64)->i64{return x-y}");
    require(evaluateCall(sub,"sub",{RuntimeValue{std::numeric_limits<std::int64_t>::min()},RuntimeValue{std::int64_t{1}}}).errorId=="TH-SPEC-I64-OVERFLOW","min sub overflow");
    auto mul=valid("MULOVERFLOW","fn mul(x:i64,y:i64)->i64{return x*y}");
    require(evaluateCall(mul,"mul",{RuntimeValue{std::numeric_limits<std::int64_t>::max()},RuntimeValue{std::int64_t{2}}}).errorId=="TH-SPEC-I64-OVERFLOW","multiplication overflow");
    auto mm=valid("MATOVERFLOW","fn mm(x:Tensor<i64,2>,y:Tensor<i64,2>)->Tensor<i64,2>{return x*y}");
    require(evaluateCall(mm,"mm",{matrix({1,1},{std::numeric_limits<std::int64_t>::max()}),matrix({1,1},{2})}).errorId=="TH-SPEC-I64-OVERFLOW","matmul product overflow");
    require(evaluateCall(mm,"mm",{matrix({1,2},{std::numeric_limits<std::int64_t>::max(),1}),matrix({2,1},{1,1})}).errorId=="TH-SPEC-I64-OVERFLOW","matmul accumulation overflow");
    auto reduction=valid("EMPTY","fn reduce(x:Tensor<i64,2>)->Tensor<i64,1>{return sum(x,0)}");
    require(evaluateCall(reduction,"reduce",{matrix({0,3},{})}).format()==
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[3],\"values\":[0,0,0]}","zero-sized reduction incorrect");
    auto slice=valid("SLICE-RUNTIME","fn take(x:Tensor<i64,1>,end:i64)->Tensor<i64,1>{return x[0:end]}");
    require(dump(slice).find("check slice")!=std::string::npos,"runtime slice check missing");
    require(evaluateCall(slice,"take",{matrix({2},{1,2}),RuntimeValue{std::int64_t{3}}}).errorId=="TH-SPEC-SLICE","runtime slice failure");
    auto transpose=valid("TRANSPOSE","let A=[1,2,3;4,5,6]\nlet B=A.T");
    require(evaluateBinding(transpose,"B").format()==
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[3,2],\"values\":[1,4,2,5,3,6]}","transpose incorrect");
    value("SUBTRACT","let A=[5,6;7,8]\nlet B=A-2","B",
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[3,4,5,6]}");
    value("ELEMENT-MUL","let A=[1,2;3,4]\nlet B=A.*A","B",
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2,2],\"values\":[1,4,9,16]}");
    value("SCALE","let A=[1,2]\nlet B=3*A","B",
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2],\"values\":[3,6]}");
    value("MIXED-SLICE","let A=[1,2;3,4]\nlet B=A[:,1]","B",
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2],\"values\":[2,4]}");
    auto partial=valid("PARTIAL-INDEX","let A=[1,2;3,4]\nlet B=A[1]");
    require(dump(partial).find("index:Tensor<i64,1>")!=std::string::npos &&
        dump(partial).find("borrowed_view")!=std::string::npos,"partial indexing lost logical view fact");
    require(evaluateBinding(partial,"B").format()==
        "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[2],\"values\":[3,4]}","partial indexing value incorrect");
    auto recursive=valid("RECURSION-GUARD","fn forever(x:i64)->i64{return forever(x)}");
    require(evaluateCall(recursive,"forever",{RuntimeValue{std::int64_t{1}}}).errorId=="TH005-CALL-DEPTH",
        "bounded call-depth guard failed");
    const std::vector<std::string> invalid={"let x=foo", "let x=[true,1]", "fn f(x:Tensor<i64,2>)->i64{return x[0,0,0]}",
        "let x=99999999999999999999999999999999", "fn f(x:i64)->i64{return x.T}", "fn f(x:i64)->i64{return sum(x,0)}"};
    for (std::size_t k=0;k<invalid.size();++k) {
        auto syntax=parse(invalid[k],"bounded.th");
        require(syntax.module && syntax.diagnostics.empty(),"bounded invalid source did not parse");
        auto result=analyze(*syntax.module);
        require(!result.module && !result.diagnostics.empty(),"bounded invalid source crashed or passed");
    }
}
}
int main() {
    try { verifierTests(); controlVerifierTests(); sourceTests(); controlFlowTests(); diagnosticTests(); adversarialTests();
        std::cout << "PASS V0 semantic source, verifier, evaluator and adversarial matrix\n"; return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL V0 semantic: " << error.what() << '\n'; return 1; }
}
