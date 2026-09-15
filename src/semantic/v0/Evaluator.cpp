#include "semantic/v0/Evaluator.hpp"
#include "semantic/v0/Verifier.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <sstream>

namespace thiran::v0::semantic {
namespace {
struct RuntimeFailure { std::string id; };
[[noreturn]] void fail(std::string id) { throw RuntimeFailure{std::move(id)}; }
std::int64_t checked(char op,std::int64_t a,std::int64_t b) {
    std::int64_t result=0;
    bool overflow=op=='+' ? __builtin_add_overflow(a,b,&result) :
                  op=='-' ? __builtin_sub_overflow(a,b,&result) : __builtin_mul_overflow(a,b,&result);
    if (overflow) fail("TH-SPEC-I64-OVERFLOW");
    return result;
}
std::int64_t neg(std::int64_t a) {
    if (a==std::numeric_limits<std::int64_t>::min()) fail("TH-SPEC-I64-OVERFLOW");
    return -a;
}
std::size_t count(const std::vector<std::int64_t>& shape) {
    std::size_t size=1;
    for (auto extent:shape) {
        if (extent<0) fail("TH-SPEC-SHAPE");
        if (static_cast<std::uint64_t>(extent)>1000000 || size>1000000/static_cast<std::size_t>(extent ? extent : 1))
            fail("TH005-RESOURCE-LIMIT");
        size*=static_cast<std::size_t>(extent);
    }
    return size;
}
const RuntimeTensor& tensorValue(const RuntimeValue& v) {
    if (auto* t=std::get_if<RuntimeTensor>(&v.data)) return *t;
    fail("TH005-RUNTIME-TYPE");
}
std::int64_t integerValue(const RuntimeValue& v) {
    if (auto* x=std::get_if<std::int64_t>(&v.data)) return *x;
    fail("TH005-RUNTIME-TYPE");
}
std::vector<std::int64_t> coordinates(std::size_t flat,const std::vector<std::int64_t>& shape) {
    std::vector<std::int64_t> result(shape.size());
    for (std::size_t d=shape.size();d-- > 0;) {
        if (shape[d]==0) return result;
        result[d]=static_cast<std::int64_t>(flat%static_cast<std::size_t>(shape[d]));
        flat/=static_cast<std::size_t>(shape[d]);
    }
    return result;
}
std::size_t offset(const std::vector<std::int64_t>& coordinate,const std::vector<std::int64_t>& shape) {
    std::size_t flat=0;
    for (std::size_t d=0;d<shape.size();++d) flat=flat*static_cast<std::size_t>(shape[d])+static_cast<std::size_t>(coordinate[d]);
    return flat;
}
std::vector<std::int64_t> broadcastShape(const RuntimeValue& a,const RuntimeValue& b) {
    auto* at=std::get_if<RuntimeTensor>(&a.data), *bt=std::get_if<RuntimeTensor>(&b.data);
    std::size_t rank=std::max(at?at->shape.size():0U,bt?bt->shape.size():0U);
    std::vector<std::int64_t> shape(rank);
    for (std::size_t d=0;d<rank;++d) {
        auto extent=[&](const RuntimeTensor* t) {
            return !t || d<rank-t->shape.size() ? std::int64_t{1} : t->shape[d-(rank-t->shape.size())];
        };
        auto x=extent(at),y=extent(bt);
        if (x!=y && x!=1 && y!=1) fail("TH-SPEC-BROADCAST");
        shape[d]=x==1?y:y==1?x:x;
    }
    return shape;
}
std::int64_t elementAt(const RuntimeValue& v,const std::vector<std::int64_t>& outputCoordinate,std::size_t outputRank) {
    if (auto* x=std::get_if<std::int64_t>(&v.data)) return *x;
    const auto& t=tensorValue(v);
    std::vector<std::int64_t> coordinate(t.shape.size());
    for (std::size_t d=0;d<t.shape.size();++d)
        coordinate[d]=t.shape[d]==1?0:outputCoordinate[outputRank-t.shape.size()+d];
    return t.values.at(offset(coordinate,t.shape));
}
RuntimeValue elementwise(const RuntimeValue& a,const RuntimeValue& b,char op) {
    auto shape=broadcastShape(a,b);
    if (shape.empty()) return RuntimeValue{checked(op,integerValue(a),integerValue(b))};
    RuntimeTensor result{TypeKind::I64,shape,{}};
    auto size=count(shape); result.values.reserve(size);
    for (std::size_t flat=0;flat<size;++flat) {
        auto coordinate=coordinates(flat,shape);
        result.values.push_back(checked(op,elementAt(a,coordinate,shape.size()),elementAt(b,coordinate,shape.size())));
    }
    return RuntimeValue{std::move(result)};
}
RuntimeValue matmul(const RuntimeValue& a,const RuntimeValue& b) {
    const auto& x=tensorValue(a); const auto& y=tensorValue(b);
    if (x.shape.size()!=2 || y.shape.size()!=2 || x.shape[1]!=y.shape[0]) fail("TH-SPEC-SHAPE");
    RuntimeTensor result{TypeKind::I64,{x.shape[0],y.shape[1]},std::vector<std::int64_t>(count({x.shape[0],y.shape[1]}),0)};
    for (std::int64_t i=0;i<x.shape[0];++i) for (std::int64_t j=0;j<y.shape[1];++j) {
        std::int64_t accumulator=0;
        for (std::int64_t k=0;k<x.shape[1];++k) {
            auto product=checked('*',x.values.at(static_cast<std::size_t>(i*x.shape[1]+k)),
                                      y.values.at(static_cast<std::size_t>(k*y.shape[1]+j)));
            accumulator=checked('+',accumulator,product);
        }
        result.values[static_cast<std::size_t>(i*y.shape[1]+j)]=accumulator;
    }
    return RuntimeValue{std::move(result)};
}
struct ResolvedSelector { bool slice; std::int64_t start,end,step; };
ResolvedSelector resolve(const Selector& selector,const RuntimeTensor& t,std::uint32_t axis,
                         const std::map<ValueId,RuntimeValue>& values) {
    auto get=[&](std::optional<ValueId> id,std::int64_t otherwise) {
        return id?integerValue(values.at(*id)):otherwise;
    };
    auto extent=t.shape.at(axis);
    if (!selector.slice) {
        auto index=get(selector.index,0);
        if (index<0 || index>=extent) fail("TH-SPEC-BOUNDS");
        return {false,index,index+1,1};
    }
    auto start=get(selector.start,0), end=get(selector.end,extent), step=get(selector.step,1);
    if (step<=0 || start<0 || end<0 || start>end || end>extent) fail("TH-SPEC-SLICE");
    return {true,start,end,step};
}
RuntimeValue indexed(const RuntimeValue& value,const Instruction& instruction,const std::map<ValueId,RuntimeValue>& values) {
    const auto& t=tensorValue(value);
    std::vector<ResolvedSelector> selectors;
    std::vector<std::int64_t> shape;
    for (std::size_t d=0;d<t.shape.size();++d) {
        ResolvedSelector s = d<instruction.selectors.size() ? resolve(instruction.selectors[d],t,static_cast<std::uint32_t>(d),values)
            : ResolvedSelector{true,0,t.shape[d],1};
        selectors.push_back(s);
        if (s.slice) shape.push_back(s.end==s.start?0:1+(s.end-s.start-1)/s.step);
    }
    if (shape.empty()) {
        std::vector<std::int64_t> coordinate;
        for (auto s:selectors) coordinate.push_back(s.start);
        return RuntimeValue{t.values.at(offset(coordinate,t.shape))};
    }
    RuntimeTensor result{TypeKind::I64,shape,{}};
    auto size=count(shape); result.values.reserve(size);
    for (std::size_t flat=0;flat<size;++flat) {
        auto out=coordinates(flat,shape);
        std::vector<std::int64_t> source; std::size_t axis=0;
        for (auto s:selectors) source.push_back(s.start+(s.slice?out[axis++]*s.step:0));
        result.values.push_back(t.values.at(offset(source,t.shape)));
    }
    return RuntimeValue{std::move(result)};
}
RuntimeValue transpose(const RuntimeValue& value) {
    const auto& t=tensorValue(value);
    RuntimeTensor result{t.dtype,{t.shape[1],t.shape[0]},std::vector<std::int64_t>(t.values.size())};
    for (std::int64_t i=0;i<t.shape[0];++i) for (std::int64_t j=0;j<t.shape[1];++j)
        result.values[static_cast<std::size_t>(j*t.shape[0]+i)]=t.values.at(static_cast<std::size_t>(i*t.shape[1]+j));
    return RuntimeValue{std::move(result)};
}
RuntimeValue sum(const RuntimeValue& value,std::uint32_t axis) {
    const auto& t=tensorValue(value);
    if (axis>=t.shape.size()) fail("TH-SPEC-AXIS");
    std::vector<std::int64_t> shape=t.shape; shape.erase(shape.begin()+axis);
    auto outputSize=count(shape);
    if (shape.empty()) {
        std::int64_t acc=0;
        for (auto v:t.values) acc=checked('+',acc,v);
        return RuntimeValue{acc};
    }
    RuntimeTensor result{TypeKind::I64,shape,std::vector<std::int64_t>(outputSize,0)};
    for (std::size_t flat=0;flat<outputSize;++flat) {
        auto out=coordinates(flat,shape); std::int64_t acc=0;
        for (std::int64_t k=0;k<t.shape[axis];++k) {
            auto source=out; source.insert(source.begin()+axis,k);
            acc=checked('+',acc,t.values.at(offset(source,t.shape)));
        }
        result.values[flat]=acc;
    }
    return RuntimeValue{std::move(result)};
}
bool runtimeType(const RuntimeValue& v,const Type& type) {
    if (type==scalar(TypeKind::I64)) return std::holds_alternative<std::int64_t>(v.data);
    if (type==scalar(TypeKind::Bool)) return std::holds_alternative<bool>(v.data);
    if (type.kind==TypeKind::Tensor) {
        auto* t=std::get_if<RuntimeTensor>(&v.data);
        return t && t->dtype==type.elements[0].kind && t->shape.size()==type.rank && count(t->shape)==t->values.size();
    }
    if (type.kind==TypeKind::Tuple) {
        auto* tuple=std::get_if<RuntimeTuple>(&v.data);
        if (!tuple || tuple->size()!=type.elements.size()) return false;
        for (std::size_t i=0;i<tuple->size();++i) if (!runtimeType((*tuple)[i],type.elements[i])) return false;
        return true;
    }
    return false;
}
class Evaluator {
public:
    explicit Evaluator(const Module& module):module_(module) {}
    RuntimeValue call(FunctionId id,const std::vector<RuntimeValue>& args) {
        if (++depth_>64) fail("TH005-CALL-DEPTH");
        const auto& f=module_.functions.at(id-1);
        if (args.size()!=f.parameters.size()) fail("TH005-RUNTIME-ARITY");
        std::map<ValueId,RuntimeValue> values;
        std::map<BindingId,RuntimeValue> bindings;
        for (std::size_t i=0;i<args.size();++i) {
            if (!runtimeType(args[i],f.parameters[i].type)) fail("TH005-RUNTIME-TYPE");
            values[f.parameters[i].id]=args[i];
            bindings[f.parameters[i].binding]=args[i];
        }
        auto signal=execute(f.body,values,bindings);
        if (signal.kind!=Signal::Kind::Return || !signal.value) fail("TH005-INVALID-IR");
        auto result=*signal.value;
        --depth_; return result;
    }
    std::map<BindingId,RuntimeValue> initializer() {
        std::map<ValueId,RuntimeValue> values;
        std::map<BindingId,RuntimeValue> bindings;
        execute(module_.initializer,values,bindings);
        return bindings;
    }
private:
    const Module& module_;
    std::uint32_t depth_=0;
    std::uint64_t steps_=0;
    std::uint64_t iterations_=0;
    struct Signal {
        enum class Kind { Normal, Return, Break, Continue }; Kind kind=Kind::Normal;
        std::optional<RuntimeValue> value;
    };
    void guardIteration() { if (++iterations_>100000) fail("TH005C-STEP-LIMIT"); }
    Signal execute(const Block& block,std::map<ValueId,RuntimeValue>& values,std::map<BindingId,RuntimeValue>& bindings) {
        for (const auto& step:block.steps) {
            if (++steps_>1000000) fail("TH005-RESOURCE-LIMIT");
            if (auto* c=std::get_if<Check>(&step)) {
                if (c->kind==CheckKind::Bounds) {
                    const auto& t=tensorValue(values.at(c->operands[0]));
                    auto index=integerValue(values.at(c->operands[1]));
                    if (index<0 || index>=t.shape.at(c->axis)) fail(c->failureId);
                } else if (c->kind==CheckKind::Broadcast) {
                    try { broadcastShape(values.at(c->operands[0]),values.at(c->operands[1])); }
                    catch (const RuntimeFailure&) { fail(c->failureId); }
                } else if (c->kind==CheckKind::MatmulShape) {
                    const auto& a=tensorValue(values.at(c->operands[0]));
                    const auto& b=tensorValue(values.at(c->operands[1]));
                    if (a.shape[1]!=b.shape[0]) fail(c->failureId);
                } else if (c->kind==CheckKind::Slice) {
                    try { resolve(c->selectors.at(0),tensorValue(values.at(c->operands[0])),c->axis,values); }
                    catch (const RuntimeFailure&) { fail(c->failureId); }
                }
                continue;
            }
            if (auto* write=std::get_if<BindingWrite>(&step)) {
                bindings[write->binding]=values.at(write->value); continue;
            }
            if (auto* flow=std::get_if<Flow>(&step)) {
                if (flow->kind==Flow::Kind::Return) return {Signal::Kind::Return,values.at(*flow->value)};
                return {flow->kind==Flow::Kind::Break?Signal::Kind::Break:Signal::Kind::Continue,{}};
            }
            if (auto* s=std::get_if<Structured>(&step)) {
                if (s->kind==Structured::Kind::If) {
                    auto condition=std::get<bool>(values.at(s->condition).data);
                    auto chosen=condition?s->thenBlock:s->elseBlock;
                    if (chosen) { auto signal=execute(*chosen,values,bindings); if (signal.kind!=Signal::Kind::Normal) return signal; }
                } else if (s->kind==Structured::Kind::ForRange) {
                    auto start=integerValue(values.at(s->start)),end=integerValue(values.at(s->end));
                    // Checked signed induction: never increment after the final iteration.
                    for (auto i=start;i<end;) {
                        guardIteration(); bindings[s->induction]=RuntimeValue{i};
                        auto signal=execute(*s->bodyBlock,values,bindings);
                        if (signal.kind==Signal::Kind::Return) return signal;
                        if (signal.kind==Signal::Kind::Break) break;
                        if (i==std::numeric_limits<std::int64_t>::max()) break;
                        ++i;
                    }
                    bindings.erase(s->induction);
                } else {
                    while (true) {
                        guardIteration();
                        auto conditionSignal=execute(*s->conditionBlock,values,bindings);
                        if (conditionSignal.kind!=Signal::Kind::Normal) fail("TH005-INVALID-IR");
                        if (!std::get<bool>(values.at(*s->conditionResult).data)) break;
                        auto signal=execute(*s->bodyBlock,values,bindings);
                        if (signal.kind==Signal::Kind::Return) return signal;
                        if (signal.kind==Signal::Kind::Break) break;
                    }
                }
                continue;
            }
            const auto& i=std::get<Instruction>(step);
            auto operand=[&](std::size_t index)->const RuntimeValue& { return values.at(i.operands.at(index)); };
            RuntimeValue result;
            switch (i.op) {
                case Op::Integer: result.data=*i.integer; break;
                case Op::Boolean: result.data=*i.boolean; break;
                case Op::LoadBinding: result=bindings.at(i.binding); break;
                case Op::TensorLiteral: {
                    RuntimeTensor tensor{TypeKind::I64,{},{}};
                    for (auto extent:i.shape.extents) tensor.shape.push_back(*extent);
                    for (auto id:i.operands) tensor.values.push_back(integerValue(values.at(id)));
                    result.data=std::move(tensor); break;
                }
                case Op::Tuple: { RuntimeTuple tuple; for (auto id:i.operands) tuple.push_back(values.at(id)); result.data=std::move(tuple); break; }
                case Op::Negate: {
                    if (auto* x=std::get_if<std::int64_t>(&operand(0).data)) result.data=neg(*x);
                    else { auto t=tensorValue(operand(0)); for (auto& v:t.values) v=neg(v); result.data=std::move(t); }
                    break;
                }
                case Op::Add: result=elementwise(operand(0),operand(1),'+'); break;
                case Op::Subtract: result=elementwise(operand(0),operand(1),'-'); break;
                case Op::Multiply: case Op::ElementMultiply: result=elementwise(operand(0),operand(1),'*'); break;
                case Op::Matmul: result=matmul(operand(0),operand(1)); break;
                case Op::Index: case Op::Slice: result=indexed(operand(0),i,values); break;
                case Op::Transpose: result=transpose(operand(0)); break;
                case Op::Sum: result=sum(operand(0),i.axis); break;
                case Op::Call: { std::vector<RuntimeValue> args; for (auto id:i.operands) args.push_back(values.at(id)); result=call(i.callee,args); break; }
            }
            if (!runtimeType(result,i.type)) fail("TH005-RUNTIME-TYPE");
            values[i.id]=std::move(result);
        }
        return {};
    }
};
Observation guarded(const Module& module,const std::function<RuntimeValue(Evaluator&)>& action) {
    try {
        if (!verify(module).ok) return {false,{},"TH005-INVALID-IR"};
        Evaluator evaluator(module);
        return {true,action(evaluator),{}};
    } catch (const RuntimeFailure& failure) { return {false,{},failure.id}; }
      catch (const std::exception&) { return {false,{},"TH005-EVALUATOR-INTERNAL"}; }
}
std::string valueFormat(const RuntimeValue& v) {
    std::ostringstream out;
    if (auto* x=std::get_if<std::int64_t>(&v.data)) out << "{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":" << *x << '}';
    else if (auto* b=std::get_if<bool>(&v.data)) out << "{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"bool\",\"value\":" << (*b?"true":"false") << '}';
    else if (auto* t=std::get_if<RuntimeTensor>(&v.data)) {
        out << "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\"i64\",\"shape\":[";
        for (std::size_t i=0;i<t->shape.size();++i) { if (i) out << ','; out << t->shape[i]; }
        out << "],\"values\":[";
        for (std::size_t i=0;i<t->values.size();++i) { if (i) out << ','; out << t->values[i]; }
        out << "]}";
    } else {
        out << "{\"status\":\"ok\",\"kind\":\"tuple\",\"values\":[";
        const auto& tuple=std::get<RuntimeTuple>(v.data);
        for (std::size_t i=0;i<tuple.size();++i) { if (i) out << ','; out << valueFormat(tuple[i]); }
        out << "]}";
    }
    return out.str();
}
}
std::string Observation::format() const {
    if (!ok) return "{\"status\":\"error\",\"error_id\":\"" + errorId + "\"}";
    return valueFormat(*value);
}
Observation evaluateBinding(const Module& module,const std::string& name) {
    return guarded(module,[&](Evaluator& e) {
        auto values=e.initializer();
        for (auto it=module.initializer.bindings.rbegin();it!=module.initializer.bindings.rend();++it)
            if (it->name==name) return values.at(it->id);
        fail("TH005-UNKNOWN-BINDING");
    });
}
Observation evaluateCall(const Module& module,const std::string& name,const std::vector<RuntimeValue>& args) {
    return guarded(module,[&](Evaluator& e) {
        e.initializer();
        for (const auto& f:module.functions) if (f.name==name) return e.call(f.id,args);
        fail("TH005-UNKNOWN-FUNCTION");
    });
}
}
