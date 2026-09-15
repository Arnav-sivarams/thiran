#include "semantic/v0/Analyzer.hpp"
#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace thiran::v0::semantic {
std::string SemanticDiagnostic::format() const {
    return source + ':' + std::to_string(span.begin.line) + ':' + std::to_string(span.begin.column) +
        ": [" + category + "] " + message;
}
namespace {
struct Failure { SemanticDiagnostic diagnostic; };
struct Fact { Type type; ShapeFact shape; std::optional<std::int64_t> constant; };
struct Located { ValueId id; Fact fact; };
struct Local { Located value; bool mutableBinding; };
bool isTensor(const Type& t) { return t.kind == TypeKind::Tensor; }
bool i64(const Type& t) { return t == scalar(TypeKind::I64); }
bool numeric(const Type& t) { return i64(t) || (isTensor(t) && t.elements.size()==1 &&
    t.elements[0]==scalar(TypeKind::I64) && (t.rank==1 || t.rank==2)); }
std::optional<std::int64_t> checked(char op, std::int64_t a, std::int64_t b) {
    std::int64_t result;
    bool overflow = op == '+' ? __builtin_add_overflow(a,b,&result) :
                    op == '-' ? __builtin_sub_overflow(a,b,&result) : __builtin_mul_overflow(a,b,&result);
    if (overflow) return std::nullopt;
    return result;
}
std::optional<std::int64_t> literal(const std::string& spelling, bool negative = false) {
    std::uint64_t magnitude = 0;
    constexpr std::uint64_t max = 9223372036854775807ULL;
    std::uint64_t limit = negative ? max + 1 : max;
    for (char c : spelling) {
        if (c < '0' || c > '9') return std::nullopt;
        auto digit = static_cast<std::uint64_t>(c - '0');
        if (magnitude > (limit - digit) / 10) return std::nullopt;
        magnitude = magnitude * 10 + digit;
    }
    if (negative && magnitude == max + 1) return std::numeric_limits<std::int64_t>::min();
    auto value = static_cast<std::int64_t>(magnitude);
    return negative ? -value : value;
}
ShapeFact unknownShape(const Type& t) {
    ShapeFact result;
    if (isTensor(t)) result.extents.resize(t.rank);
    return result;
}
class Analyzer {
public:
    explicit Analyzer(const thiran::v0::Module& syntax) : syntax_(syntax) {
        result_.source = syntax.source; result_.span = syntax.span;
    }
    Module run() {
        for (const auto& item : syntax_.items) {
            if (auto* i = std::get_if<ImportDecl>(&item)) {
                result_.imports.push_back({i->path, i->alias, i->span});
                fail(i->span, "TH005-IMPORT-STAGE", "module linking is not supported in TH-005");
            }
        }
        for (const auto& item : syntax_.items) if (auto* decl = std::get_if<FunctionDecl>(&item)) {
            if (names_.contains(decl->name)) fail(decl->span, "TH005-DUPLICATE-FUNCTION", "duplicate function " + decl->name);
            Function f; f.id = static_cast<FunctionId>(result_.functions.size() + 1);
            f.name = decl->name; f.span = decl->span; f.exported = decl->exported;
            if (decl->resultType) f.result = resolve(*decl->resultType);
            else if (decl->exported) fail(decl->span, "TH005-PUBLIC-RESULT", "exported function requires result type");
            std::set<std::string> params;
            for (const auto& p : decl->parameters) {
                if (!params.insert(p.name).second) fail(p.span, "TH005-DUPLICATE-BINDING", "duplicate parameter " + p.name);
                auto type = resolve(p.type);
                f.parameters.push_back({static_cast<ValueId>(f.parameters.size() + 1), p.name, type, unknownShape(type), p.span});
            }
            names_[f.name] = f.id;
            declarations_.push_back(decl);
            result_.functions.push_back(std::move(f));
            states_.push_back(0);
        }
        for (std::size_t i = 0; i < result_.functions.size(); ++i) ensure(static_cast<FunctionId>(i + 1));
        BlockContext init{result_.initializer, {}, {}, 1, nullptr};
        for (const auto& item : syntax_.items) if (auto* let = std::get_if<LetStmt>(&item)) statement(*let, init);
        result_.initializer.terminated = true;
        return std::move(result_);
    }
private:
    struct BlockContext {
        Block& block;
        std::map<std::string, Local> locals;
        std::map<ValueId, Fact> facts;
        ValueId next;
        Function* function;
    };
    const thiran::v0::Module& syntax_;
    Module result_;
    std::map<std::string, FunctionId> names_;
    std::vector<const FunctionDecl*> declarations_;
    std::vector<int> states_;
    [[noreturn]] void fail(SourceSpan span, std::string id, std::string message) const {
        throw Failure{{syntax_.source, std::move(id), std::move(message), span}};
    }
    Type resolve(const TypeSyntax& t) {
        if (t.name.empty()) {
            Type result{TypeKind::Tuple, {}, 0};
            for (const auto& e : t.elements) result.elements.push_back(resolve(e));
            if (!validType(result)) fail(t.span, "TH005-UNKNOWN-TYPE", "invalid tuple type");
            return result;
        }
        static const std::map<std::string, TypeKind> scalarKinds = {
            {"bool",TypeKind::Bool},{"u8",TypeKind::U8},{"i32",TypeKind::I32},{"i64",TypeKind::I64},
            {"u32",TypeKind::U32},{"u64",TypeKind::U64},{"f32",TypeKind::F32},{"f64",TypeKind::F64}
        };
        if (auto it = scalarKinds.find(t.name); it != scalarKinds.end()) return scalar(it->second);
        if (t.name == "Tensor") {
            if (t.elements.size() != 1 || !t.rank) fail(t.span,"TH005-TENSOR-RANK","Tensor requires element and rank");
            std::uint32_t rank = 0;
            auto [end, error] = std::from_chars(t.rank->data(), t.rank->data()+t.rank->size(), rank);
            if (error != std::errc{} || end != t.rank->data()+t.rank->size() || rank > 32)
                fail(t.span,"TH005-TENSOR-RANK","Tensor rank must be an integer from 0 to 32");
            auto result = tensor(resolve(t.elements[0]), rank);
            if (!validType(result)) fail(t.span,"TH005-TENSOR-ELEMENT","Tensor element must be a scalar dtype");
            return result;
        }
        if (t.name == "Buffer") {
            Type result{TypeKind::Buffer, {resolve(t.elements.at(0))}, 0};
            if (!validType(result)) fail(t.span,"TH005-UNKNOWN-TYPE","invalid Buffer type");
            return result;
        }
        fail(t.span,"TH005-UNKNOWN-TYPE","unknown semantic type " + t.name);
    }
    void ensure(FunctionId id) {
        auto index = static_cast<std::size_t>(id - 1);
        if (states_[index] == 2) return;
        if (states_[index] == 1) {
            if (result_.functions[index].result.kind == TypeKind::Invalid)
                fail(result_.functions[index].span,"TH005-INFERENCE-CYCLE","recursive/inference cycle requires explicit result types");
            return;
        }
        states_[index] = 1;
        auto& function = result_.functions[index];
        BlockContext context{function.body, {}, {}, static_cast<ValueId>(function.parameters.size() + 1), &function};
        for (const auto& p : function.parameters) {
            Fact fact{p.type,p.shape,{}};
            context.locals[p.name] = {{p.id,fact},false};
            context.facts[p.id] = fact;
        }
        for (const auto& s : declarations_[index]->body) {
            if (function.body.terminated) fail(std::visit([](const auto& n){return n.span;},s),"TH005-AFTER-RETURN","statement after return");
            std::visit([&](const auto& n) { statement(n, context); }, s);
        }
        if (!function.body.terminated) fail(function.span,"TH005-MISSING-RETURN","function requires a return expression");
        states_[index] = 2;
    }
    Located emit(BlockContext& c, Instruction i, Fact fact) {
        if (i.op==Op::Negate || i.op==Op::Add || i.op==Op::Subtract || i.op==Op::Multiply ||
            i.op==Op::ElementMultiply || i.op==Op::Matmul || i.op==Op::Index || i.op==Op::Slice ||
            i.op==Op::Sum || i.op==Op::Call) i.effect=EffectClass::CheckedFailure;
        i.id = c.next++;
        i.type = fact.type; i.shape = fact.shape;
        c.block.steps.emplace_back(i);
        c.facts[i.id] = fact;
        return {i.id, fact};
    }
    void check(BlockContext& c, CheckKind kind, std::vector<ValueId> operands, SourceSpan span,
               std::string failureId, std::vector<Selector> selectors = {}, std::uint32_t axis = 0) {
        c.block.steps.emplace_back(Check{kind,std::move(operands),std::move(selectors),span,std::move(failureId),axis});
    }
    Located expr(const Expr& e, BlockContext& c) {
        return std::visit([&](const auto& n) -> Located {
            using N = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<N, IdentifierExpr>) {
                auto it = c.locals.find(n.name);
                if (it == c.locals.end()) fail(e.span,"TH005-UNDEFINED-NAME","undefined identifier " + n.name);
                return it->second.value;
            } else if constexpr (std::is_same_v<N, IntegerLiteralExpr>) {
                auto value = literal(n.spelling);
                if (!value) fail(e.span,"TH005-INTEGER-RANGE","integer literal is outside exact i64 range");
                Instruction i; i.op=Op::Integer; i.span=e.span; i.integer=*value;
                return emit(c,std::move(i),{scalar(TypeKind::I64),{},*value});
            } else if constexpr (std::is_same_v<N, BooleanLiteralExpr>) {
                Instruction i; i.op=Op::Boolean; i.span=e.span; i.boolean=n.value;
                return emit(c,std::move(i),{scalar(TypeKind::Bool),{}, {}});
            } else if constexpr (std::is_same_v<N, TensorLiteralExpr>) {
                Instruction i; i.op=Op::TensorLiteral; i.span=e.span;
                for (const auto& row : n.rows) for (const auto& element : row) {
                    auto v=expr(*element,c);
                    if (!i64(v.fact.type)) fail(element->span,"TH005-TENSOR-ELEMENT","TH-005 tensor literals require i64 elements");
                    i.operands.push_back(v.id);
                }
                auto rank = n.rows.size() == 1 ? 1U : 2U;
                ShapeFact shape; if (rank == 1) shape.extents={static_cast<std::int64_t>(n.rows[0].size())};
                else shape.extents={static_cast<std::int64_t>(n.rows.size()),static_cast<std::int64_t>(n.rows[0].size())};
                return emit(c,std::move(i),{tensor(scalar(TypeKind::I64),rank),shape,{}});
            } else if constexpr (std::is_same_v<N, TupleExpr>) {
                Instruction i; i.op=Op::Tuple; i.span=e.span;
                Type type{TypeKind::Tuple,{},0};
                for (const auto& element : n.elements) { auto v=expr(*element,c); i.operands.push_back(v.id); type.elements.push_back(v.fact.type); }
                return emit(c,std::move(i),{type,{}, {}});
            } else if constexpr (std::is_same_v<N, UnaryExpr>) {
                if (auto* spelling=std::get_if<IntegerLiteralExpr>(&n.operand->node)) {
                    auto value=literal(spelling->spelling,true);
                    if (!value) fail(e.span,"TH005-INTEGER-RANGE","negative integer literal is outside exact i64 range");
                    Instruction i; i.op=Op::Integer; i.span=e.span; i.integer=*value;
                    return emit(c,std::move(i),{scalar(TypeKind::I64),{},*value});
                }
                auto v=expr(*n.operand,c);
                if (!numeric(v.fact.type))
                    fail(e.span,"TH005-OPERAND-TYPE","unary minus requires i64 scalar/tensor");
                std::optional<std::int64_t> constant;
                if (v.fact.constant) {
                    if (*v.fact.constant==std::numeric_limits<std::int64_t>::min()) fail(e.span,"TH-SPEC-I64-OVERFLOW","constant negation overflows i64");
                    constant=-*v.fact.constant;
                }
                Instruction i; i.op=Op::Negate; i.span=e.span; i.operands={v.id};
                return emit(c,std::move(i),{v.fact.type,v.fact.shape,constant});
            } else if constexpr (std::is_same_v<N, BinaryExpr>) return binary(e,n,c);
            else if constexpr (std::is_same_v<N, CallExpr>) return call(e,n,c);
            else if constexpr (std::is_same_v<N, MemberExpr>) {
                auto v=expr(*n.object,c);
                if (n.member!="T" || !isTensor(v.fact.type) || v.fact.type.rank!=2 || v.fact.type.elements[0]!=scalar(TypeKind::I64))
                    fail(e.span,"TH005-UNSUPPORTED-MEMBER","only rank-2 i64 tensor .T is supported");
                auto shape=v.fact.shape; std::swap(shape.extents[0],shape.extents[1]);
                Instruction i; i.op=Op::Transpose; i.span=e.span; i.operands={v.id}; i.borrowedView=true;
                return emit(c,std::move(i),{v.fact.type,shape,{}});
            } else if constexpr (std::is_same_v<N, IndexExpr>) return index(e,n,c);
            else fail(e.span,"TH005-UNSUPPORTED-EXPR","unsupported expression");
        },e.node);
    }
    Located binary(const Expr& e, const BinaryExpr& n, BlockContext& c) {
        if (n.op==TokenKind::Slash || n.op==TokenKind::DotSlash)
            fail(e.span,"TH005-DIVISION-UNSPECIFIED","exact integer division semantics are deferred");
        auto a=expr(*n.left,c), b=expr(*n.right,c);
        bool at=isTensor(a.fact.type), bt=isTensor(b.fact.type);
        if (!numeric(a.fact.type) || !numeric(b.fact.type))
            fail(e.span,"TH005-OPERAND-TYPE","binary numerical operands must have exact i64 dtype");
        Op op= n.op==TokenKind::Plus ? Op::Add : n.op==TokenKind::Minus ? Op::Subtract :
               n.op==TokenKind::DotStar ? Op::ElementMultiply : Op::Multiply;
        if (op==Op::Multiply && at && bt) {
            if (a.fact.type.rank!=2 || b.fact.type.rank!=2) fail(e.span,"TH005-MATMUL-RANK","tensor * tensor requires rank 2");
            auto left=a.fact.shape.extents[1], right=b.fact.shape.extents[0];
            if (left && right && *left!=*right) fail(e.span,"TH-SPEC-SHAPE","known matmul inner dimensions disagree");
            if (!left || !right) check(c,CheckKind::MatmulShape,{a.id,b.id},e.span,"TH-SPEC-SHAPE");
            Instruction i; i.op=Op::Matmul; i.span=e.span; i.operands={a.id,b.id};
            return emit(c,std::move(i),{tensor(scalar(TypeKind::I64),2),{{a.fact.shape.extents[0],b.fact.shape.extents[1]}},{}});
        }
        ShapeFact shape;
        Type type=scalar(TypeKind::I64);
        if (at || bt) {
            auto rank=std::max(at ? a.fact.type.rank : 0U, bt ? b.fact.type.rank : 0U);
            type=tensor(scalar(TypeKind::I64),rank);
            shape.extents.resize(rank);
            bool needsCheck=false;
            for (std::size_t d=0; d<rank; ++d) {
                auto get=[&](const Fact& f, bool tensorOperand)->std::optional<std::int64_t> {
                    if (!tensorOperand || d < rank-f.type.rank) return 1;
                    return f.shape.extents[d-(rank-f.type.rank)];
                };
                auto x=get(a.fact,at), y=get(b.fact,bt);
                if (x && y) {
                    if (*x!=*y && *x!=1 && *y!=1) fail(e.span,"TH-SPEC-BROADCAST","known broadcast extents disagree");
                    shape.extents[d]= *x==1 ? *y : *y==1 ? *x : *x;
                } else if (x && *x==1) shape.extents[d]=y;
                else if (y && *y==1) shape.extents[d]=x;
                else needsCheck=true;
            }
            if (needsCheck) check(c,CheckKind::Broadcast,{a.id,b.id},e.span,"TH-SPEC-BROADCAST");
            if (op==Op::Multiply) op=Op::ElementMultiply;
        }
        std::optional<std::int64_t> constant;
        if (!at && !bt && a.fact.constant && b.fact.constant) {
            char operation=op==Op::Add?'+':op==Op::Subtract?'-':'*';
            constant=checked(operation,*a.fact.constant,*b.fact.constant);
            if (!constant) fail(e.span,"TH-SPEC-I64-OVERFLOW","constant arithmetic overflows i64");
        }
        Instruction i; i.op=op; i.span=e.span; i.operands={a.id,b.id};
        return emit(c,std::move(i),{type,shape,constant});
    }
    Located call(const Expr& e, const CallExpr& n, BlockContext& c) {
        auto* name=std::get_if<IdentifierExpr>(&n.callee->node);
        if (!name) fail(e.span,"TH005-UNKNOWN-FUNCTION","callee must be a declared function or sum intrinsic");
        if (name->name=="sum") {
            if (n.arguments.size()!=2) fail(e.span,"TH005-ARITY","sum expects two arguments");
            auto a=expr(*n.arguments[0],c), axis=expr(*n.arguments[1],c);
            if (!isTensor(a.fact.type) || !numeric(a.fact.type) || !i64(axis.fact.type))
                fail(e.span,"TH005-ARGUMENT-TYPE","sum requires i64 tensor and i64 axis");
            if (!axis.fact.constant) fail(e.span,"TH005-DYNAMIC-AXIS","sum axis must be compile-time constant in TH-005");
            if (*axis.fact.constant<0 || *axis.fact.constant>=a.fact.type.rank) fail(e.span,"TH-SPEC-AXIS","sum axis is out of rank");
            Type type=a.fact.type.rank==1 ? scalar(TypeKind::I64) : tensor(scalar(TypeKind::I64),a.fact.type.rank-1);
            ShapeFact shape=a.fact.shape;
            shape.extents.erase(shape.extents.begin()+*axis.fact.constant);
            Instruction i; i.op=Op::Sum; i.span=e.span; i.operands={a.id,axis.id}; i.axis=static_cast<std::uint32_t>(*axis.fact.constant);
            return emit(c,std::move(i),{type,shape,{}});
        }
        auto it=names_.find(name->name);
        if (it==names_.end()) fail(e.span,"TH005-UNKNOWN-FUNCTION","unknown function " + name->name);
        auto id=it->second;
        if (n.arguments.size()!=result_.functions[id-1].parameters.size()) fail(e.span,"TH005-ARITY","function argument count disagrees with signature");
        Instruction i; i.op=Op::Call; i.span=e.span; i.callee=id;
        for (std::size_t k=0;k<n.arguments.size();++k) {
            auto arg=expr(*n.arguments[k],c);
            if (arg.fact.type!=result_.functions[id-1].parameters[k].type)
                fail(n.arguments[k]->span,"TH005-ARGUMENT-TYPE","argument type disagrees with parameter type");
            i.operands.push_back(arg.id);
        }
        ensure(id);
        auto type=result_.functions[id-1].result;
        if (!executableType(type)) fail(e.span,"TH005-EXECUTION-UNSUPPORTED","call result is represented but not executable in TH-005");
        return emit(c,std::move(i),{type,unknownShape(type),{}});
    }
    Located index(const Expr& e, const IndexExpr& n, BlockContext& c) {
        auto object=expr(*n.object,c);
        if (!isTensor(object.fact.type) || object.fact.type.elements[0]!=scalar(TypeKind::I64) || object.fact.type.rank>2)
            fail(e.span,"TH005-INDEX-TYPE","indexing requires supported i64 tensor");
        if (n.axes.size()>object.fact.type.rank) fail(e.span,"TH005-INDEX-RANK","index count exceeds tensor rank");
        Instruction i; i.span=e.span; i.operands={object.id};
        ShapeFact shape; std::uint32_t remaining=object.fact.type.rank;
        bool anySlice=false;
        for (std::size_t d=0;d<n.axes.size();++d) {
            auto extent=object.fact.shape.extents[d];
            std::visit([&](const auto& selector) {
                using S=std::decay_t<decltype(selector)>;
                Selector ir;
                if constexpr (std::is_same_v<S, IndexSelector>) {
                    auto index=expr(*selector.value,c);
                    if (!i64(index.fact.type)) fail(selector.span,"TH005-INDEX-TYPE","index must be i64");
                    ir.index=index.id; --remaining;
                    if (index.fact.constant && (*index.fact.constant<0 || (extent && *index.fact.constant>=*extent)))
                        fail(selector.span,"TH-SPEC-BOUNDS","known index is out of bounds");
                    if (!index.fact.constant || !extent) check(c,CheckKind::Bounds,{object.id,index.id},selector.span,"TH-SPEC-BOUNDS",{},static_cast<std::uint32_t>(d));
                } else {
                    ir.slice=true; anySlice=true;
                    auto field=[&](const ExprPtr& source,std::optional<ValueId>& target)->std::optional<std::int64_t> {
                        if (!source) return {};
                        auto value=expr(*source,c);
                        if (!i64(value.fact.type)) fail(source->span,"TH005-SLICE-TYPE","slice field must be i64");
                        target=value.id; return value.fact.constant;
                    };
                    auto start=field(selector.start,ir.start), end=field(selector.end,ir.end), step=field(selector.step,ir.step);
                    auto s=selector.start ? start : std::optional<std::int64_t>(0);
                    auto t=selector.step ? step : std::optional<std::int64_t>(1);
                    auto z=selector.end ? end : extent;
                    if (t && *t<=0) fail(selector.span,"TH-SPEC-SLICE","slice step must be positive");
                    if ((s && *s<0) || (z && *z<0) || (s && z && *s>*z) || (extent && z && *z>*extent))
                        fail(selector.span,"TH-SPEC-SLICE","known slice bounds are invalid");
                    if (!s || !z || !t || !extent) check(c,CheckKind::Slice,{object.id},selector.span,"TH-SPEC-SLICE",{ir},static_cast<std::uint32_t>(d));
                    if (s && z && t) shape.extents.push_back(*z==*s ? 0 : 1+(*z-*s-1)/ *t);
                    else shape.extents.push_back({});
                }
                i.selectors.push_back(ir);
            },n.axes[d]);
        }
        for (std::size_t d=n.axes.size();d<object.fact.type.rank;++d) shape.extents.push_back(object.fact.shape.extents[d]);
        Type type=remaining==0 ? scalar(TypeKind::I64) : tensor(scalar(TypeKind::I64),remaining);
        i.op=anySlice ? Op::Slice : Op::Index; i.borrowedView=anySlice || remaining>0;
        return emit(c,std::move(i),{type,shape,{}});
    }
    void statement(const LetStmt& n, BlockContext& c) {
        if (c.locals.contains(n.name)) fail(n.span,"TH005-DUPLICATE-BINDING","duplicate binding " + n.name);
        auto value=expr(*n.value,c);
        c.locals[n.name]={value,n.mutableBinding};
        c.block.bindings.push_back({n.name,value.id,n.mutableBinding,n.span});
    }
    void statement(const RebindStmt& n, BlockContext& c) {
        auto it=c.locals.find(n.name);
        if (it==c.locals.end()) fail(n.span,"TH005-UNDEFINED-NAME","undefined rebind target " + n.name);
        if (!it->second.mutableBinding) fail(n.span,"TH005-IMMUTABLE-REBIND","binding is immutable: " + n.name);
        auto value=expr(*n.value,c);
        if (value.fact.type!=it->second.value.fact.type) fail(n.span,"TH005-REBIND-TYPE","rebind type disagrees with original binding");
        it->second.value=value;
        c.block.bindings.push_back({n.name,value.id,true,n.span});
    }
    void statement(const ReturnStmt& n, BlockContext& c) {
        auto value=expr(*n.value,c);
        if (c.function->result.kind==TypeKind::Invalid) c.function->result=value.fact.type;
        else if (value.fact.type!=c.function->result)
            fail(n.span,"TH005-RETURN-TYPE","return type " + typeName(value.fact.type) + " disagrees with " + typeName(c.function->result));
        c.block.returned=value.id; c.block.terminated=true;
    }
};
}
AnalysisResult analyze(const thiran::v0::Module& syntax) {
    AnalysisResult result;
    try { result.module=Analyzer(syntax).run(); }
    catch (const Failure& failure) { result.diagnostics.push_back(failure.diagnostic); }
    catch (const std::exception& error) { result.diagnostics.push_back({syntax.source,"TH005-INTERNAL",error.what(),syntax.span}); }
    return result;
}
}
