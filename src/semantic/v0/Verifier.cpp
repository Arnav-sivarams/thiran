#include "semantic/v0/Verifier.hpp"
#include <map>
#include <set>

namespace thiran::v0::semantic {
namespace {
void error(VerificationResult& r, std::string message) { r.errors.push_back(std::move(message)); }
bool shapeValid(const Type& t, const ShapeFact& s) {
    return s.extents.size() == (t.kind == TypeKind::Tensor ? t.rank : 0);
}
bool sameElement(const Type& a, const Type& b) {
    return a.kind==TypeKind::Tensor && b.kind==TypeKind::Tensor && a.elements.size()==1 && b.elements.size()==1 && a.elements[0]==b.elements[0];
}
void block(const Module& m, const Block& b, const Function* function, VerificationResult& r) {
    std::map<ValueId,Type> available;
    ValueId expected=1;
    if (function) for (const auto& p : function->parameters) {
        if (p.id!=expected++) error(r,"parameter ValueIds must be unique and monotonic");
        if (!validType(p.type) || !shapeValid(p.type,p.shape)) error(r,"invalid parameter type/shape");
        if (!available.emplace(p.id,p.type).second) error(r,"duplicate ValueId");
    }
    auto used=[&](ValueId id) -> Type {
        auto it=available.find(id);
        if (it==available.end()) { error(r,"unknown or use-before-definition ValueId"); return {}; }
        return it->second;
    };
    auto selectorUses=[&](const std::vector<Selector>& selectors) {
        for (const auto& s : selectors) {
            if (s.slice) {
                if (s.index) error(r,"slice selector has integer index");
                for (auto v : {s.start,s.end,s.step}) if (v && used(*v)!=scalar(TypeKind::I64)) error(r,"slice field must be i64");
            } else {
                if (!s.index || s.start || s.end || s.step) error(r,"malformed integer selector");
                if (s.index && used(*s.index)!=scalar(TypeKind::I64)) error(r,"index selector must be i64");
            }
        }
    };
    for (const auto& step : b.steps) {
        if (const auto* c=std::get_if<Check>(&step)) {
            for (auto id:c->operands) used(id);
            selectorUses(c->selectors);
            if (c->failureId.empty() || c->effect!=EffectClass::CheckedFailure) error(r,"malformed runtime check");
            if (c->kind==CheckKind::Bounds) {
                if (c->operands.size()!=2 || used(c->operands[0]).kind!=TypeKind::Tensor ||
                    used(c->operands[1])!=scalar(TypeKind::I64) || c->axis>=used(c->operands[0]).rank ||
                    c->failureId!="TH-SPEC-BOUNDS") error(r,"malformed bounds check");
            } else if (c->kind==CheckKind::Broadcast || c->kind==CheckKind::MatmulShape) {
                if (c->operands.size()!=2 || used(c->operands[0]).kind!=TypeKind::Tensor ||
                    used(c->operands[1]).kind!=TypeKind::Tensor ||
                    c->failureId!=(c->kind==CheckKind::Broadcast ? "TH-SPEC-BROADCAST":"TH-SPEC-SHAPE")) error(r,"malformed shape check");
            } else if (c->kind==CheckKind::Slice) {
                if (c->operands.size()!=1 || used(c->operands[0]).kind!=TypeKind::Tensor ||
                    c->axis>=used(c->operands[0]).rank || c->selectors.size()!=1 || !c->selectors[0].slice ||
                    c->failureId!="TH-SPEC-SLICE") error(r,"malformed slice check");
            } else error(r,"invalid check kind");
            continue;
        }
        const auto& i=std::get<Instruction>(step);
        std::vector<Type> operands;
        for (auto id:i.operands) operands.push_back(used(id));
        selectorUses(i.selectors);
        if (!validType(i.type) || !shapeValid(i.type,i.shape)) error(r,"invalid instruction type/shape");
        if (i.id!=expected++) error(r,"instruction ValueIds must be unique and monotonic");
        if (available.contains(i.id)) error(r,"duplicate ValueId");
        auto require=[&](bool condition) { if (!condition) error(r,"wrong instruction result type or operand contract"); };
        switch (i.op) {
            case Op::Integer: require(operands.empty() && i.integer && i.type==scalar(TypeKind::I64)); break;
            case Op::Boolean: require(operands.empty() && i.boolean && i.type==scalar(TypeKind::Bool)); break;
            case Op::TensorLiteral: {
                bool correct=i.type.kind==TypeKind::Tensor && i.type.elements.size()==1 &&
                    i.type.elements[0]==scalar(TypeKind::I64) && (i.type.rank==1 || i.type.rank==2);
                for (auto t:operands) correct &= t==scalar(TypeKind::I64);
                require(correct); break;
            }
            case Op::Tuple: {
                bool correct=i.type.kind==TypeKind::Tuple && i.type.elements==operands;
                require(correct); break;
            }
            case Op::Negate: require(operands.size()==1 && i.type==operands[0] && executableType(i.type) && i.type.kind!=TypeKind::Tuple && i.type.kind!=TypeKind::Bool); break;
            case Op::Add: case Op::Subtract: case Op::ElementMultiply: case Op::Multiply: {
                bool correct=operands.size()==2 && executableType(i.type) && i.type.kind!=TypeKind::Tuple && i.type.kind!=TypeKind::Bool;
                if (correct) {
                    bool a=operands[0].kind==TypeKind::Tensor,bv=operands[1].kind==TypeKind::Tensor;
                    correct &= (a || operands[0]==scalar(TypeKind::I64)) && (bv || operands[1]==scalar(TypeKind::I64));
                    correct &= i.type==(a || bv ? tensor(scalar(TypeKind::I64),std::max(a?operands[0].rank:0U,bv?operands[1].rank:0U)) : scalar(TypeKind::I64));
                    if (i.op==Op::Multiply) correct &= !a && !bv;
                }
                require(correct); break;
            }
            case Op::Matmul: require(operands.size()==2 && sameElement(operands[0],operands[1]) && operands[0].rank==2 && operands[1].rank==2 && i.type==tensor(scalar(TypeKind::I64),2)); break;
            case Op::Index: case Op::Slice: {
                bool correct=operands.size()==1 && operands[0].kind==TypeKind::Tensor && i.selectors.size()<=operands[0].rank;
                if (correct) {
                    std::uint32_t removed=0; bool hasSlice=false;
                    for (auto s:i.selectors) { removed+=!s.slice; hasSlice|=s.slice; }
                    std::uint32_t rank=operands[0].rank-removed;
                    correct &= i.type==(rank==0 ? scalar(TypeKind::I64) : tensor(scalar(TypeKind::I64),rank));
                    correct &= (i.op==Op::Slice)==hasSlice && i.borrowedView==(hasSlice || rank>0);
                }
                require(correct); break;
            }
            case Op::Transpose: require(operands.size()==1 && operands[0].kind==TypeKind::Tensor && operands[0].rank==2 && i.type==operands[0] && i.borrowedView); break;
            case Op::Sum: require(operands.size()==2 && operands[0].kind==TypeKind::Tensor && operands[1]==scalar(TypeKind::I64) && i.axis<operands[0].rank && i.type==(operands[0].rank==1 ? scalar(TypeKind::I64) : tensor(scalar(TypeKind::I64),operands[0].rank-1))); break;
            case Op::Call: {
                if (i.callee==0 || i.callee>m.functions.size()) { error(r,"invalid function call target"); break; }
                const auto& target=m.functions[i.callee-1];
                bool correct=target.id==i.callee && i.type==target.result && operands.size()==target.parameters.size();
                if (operands.size()==target.parameters.size()) for (std::size_t k=0;k<operands.size();++k) correct &= operands[k]==target.parameters[k].type;
                require(correct); break;
            }
        }
        available.emplace(i.id,i.type);
    }
    for (const auto& binding:b.bindings) used(binding.value);
    if (!b.terminated || (function && !b.returned) || (!function && b.returned)) error(r,"missing/invalid block terminator");
    if (b.returned && function && used(*b.returned)!=function->result) error(r,"return type mismatch");
}
}
VerificationResult verify(const Module& m) {
    VerificationResult result;
    std::set<std::string> names;
    for (std::size_t k=0;k<m.functions.size();++k) {
        const auto& f=m.functions[k];
        if (f.id!=k+1 || !names.insert(f.name).second) error(result,"FunctionIds/names must be deterministic and unique");
        if (!validType(f.result)) error(result,"invalid function result type");
        block(m,f.body,&f,result);
    }
    block(m,m.initializer,nullptr,result);
    result.ok=result.errors.empty();
    return result;
}
}
