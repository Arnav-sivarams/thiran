#include "semantic/v0/Ir.hpp"
#include <sstream>

namespace thiran::v0::semantic {
Type scalar(TypeKind kind) { return {kind, {}, 0}; }
Type tensor(Type element, std::uint32_t rank) { return {TypeKind::Tensor, {std::move(element)}, rank}; }
std::string typeName(const Type& t) {
    switch (t.kind) {
        case TypeKind::Bool: return "bool"; case TypeKind::U8: return "u8"; case TypeKind::I32: return "i32";
        case TypeKind::I64: return "i64"; case TypeKind::U32: return "u32";
        case TypeKind::U64: return "u64"; case TypeKind::F32: return "f32";
        case TypeKind::F64: return "f64";
        case TypeKind::Tensor: return "Tensor<" + (t.elements.empty() ? "?" : typeName(t.elements[0])) + "," + std::to_string(t.rank) + ">";
        case TypeKind::Buffer: return "Buffer<" + (t.elements.empty() ? "?" : typeName(t.elements[0])) + ">";
        case TypeKind::Tuple: {
            std::string result = "(";
            for (std::size_t i = 0; i < t.elements.size(); ++i) { if (i) result += ','; result += typeName(t.elements[i]); }
            return result + ')';
        }
        default: return "<invalid>";
    }
}
bool validType(const Type& t) {
    switch (t.kind) {
        case TypeKind::Bool: case TypeKind::U8: case TypeKind::I32: case TypeKind::I64: case TypeKind::U32:
        case TypeKind::U64: case TypeKind::F32: case TypeKind::F64:
            return t.elements.empty() && t.rank == 0;
        case TypeKind::Tensor:
            return t.elements.size() == 1 && t.rank <= 32 && t.elements[0].elements.empty() &&
                t.elements[0].kind != TypeKind::Invalid && t.elements[0].kind != TypeKind::U8 && t.elements[0].kind != TypeKind::Buffer &&
                t.elements[0].kind != TypeKind::Tensor && t.elements[0].kind != TypeKind::Tuple;
        case TypeKind::Buffer:
            return t.elements.size() == 1 && t.rank == 0 && validType(t.elements[0]);
        case TypeKind::Tuple:
            if (t.elements.size() < 2 || t.rank) return false;
            for (const auto& e : t.elements) if (!validType(e)) return false;
            return true;
        default: return false;
    }
}
bool executableType(const Type& t) {
    if (t == scalar(TypeKind::I64) || t == scalar(TypeKind::Bool)) return true;
    if (t.kind == TypeKind::Tensor) return t.elements.size() == 1 && t.elements[0] == scalar(TypeKind::I64) && (t.rank == 1 || t.rank == 2);
    if (t.kind == TypeKind::Tuple) { for (const auto& e : t.elements) if (!executableType(e)) return false; return true; }
    return false;
}
namespace {
std::string opName(Op op) {
    switch (op) {
        case Op::Integer: return "integer"; case Op::Boolean: return "boolean";
        case Op::LoadBinding: return "load_binding";
        case Op::TensorLiteral: return "tensor_literal"; case Op::Tuple: return "tuple";
        case Op::Copy: return "copy"; case Op::Move: return "move";
        case Op::MutableBorrow: return "borrow_mut";
        case Op::Negate: return "negate"; case Op::Add: return "add";
        case Op::Subtract: return "subtract"; case Op::Multiply: return "multiply";
        case Op::ElementMultiply: return "element_multiply"; case Op::Matmul: return "matmul";
        case Op::Index: return "index"; case Op::Slice: return "slice_view";
        case Op::Transpose: return "transpose_view"; case Op::Sum: return "sum";
        case Op::Call: return "call";
    }
    return "invalid";
}
std::string checkName(CheckKind k) {
    switch (k) { case CheckKind::Bounds: return "bounds"; case CheckKind::Broadcast: return "broadcast";
        case CheckKind::MatmulShape: return "matmul_shape"; case CheckKind::Slice: return "slice"; }
    return "invalid";
}
void ids(std::ostringstream& out, const std::vector<ValueId>& values) {
    for (auto v : values) out << " %" << v;
}
void selectors(std::ostringstream& out, const std::vector<Selector>& values) {
    for (const auto& s : values) {
        out << (s.slice ? " slice(" : " index(");
        auto field = [&](std::optional<ValueId> id) { if (id) out << '%' << *id; else out << '_'; };
        if (s.slice) { field(s.start); out << ','; field(s.end); out << ','; field(s.step); }
        else field(s.index);
        out << ')';
    }
}
void block(std::ostringstream& out, const Block& b, std::string indent="  ") {
    out << indent << "block #" << b.id << " parent #" << b.parent << '\n';
    for (const auto& step : b.steps) {
        if (const auto* i = std::get_if<Instruction>(&step)) {
            out << indent << "%" << i->id << " = " << opName(i->op) << ':' << typeName(i->type);
            ids(out, i->operands); selectors(out, i->selectors);
            if (i->integer) out << " value=" << *i->integer;
            if (i->boolean) out << " value=" << (*i->boolean ? "true" : "false");
            if (i->op == Op::Call) out << " @" << i->callee;
            if (i->op == Op::LoadBinding || i->op == Op::Move || i->op == Op::MutableBorrow) out << " $" << i->binding;
            if (i->op == Op::Call && !i->argumentAccess.empty()) {
                out << " access=[";
                for (std::size_t k=0;k<i->argumentAccess.size();++k) {
                    if (k) out << ',';
                    out << (i->argumentAccess[k]==AccessMode::Read?"read":i->argumentAccess[k]==AccessMode::MutableBorrow?"borrow_mut":"move");
                }
                out << ']';
            }
            if (i->op == Op::Sum) out << " axis=" << i->axis;
            if (i->borrowedView) out << " borrowed_view";
            if (!i->shape.extents.empty()) {
                out << " shape=[";
                for (std::size_t d = 0; d < i->shape.extents.size(); ++d) {
                    if (d) out << ','; if (i->shape.extents[d]) out << *i->shape.extents[d]; else out << '?';
                }
                out << ']';
            }
            out << '\n';
        } else if (const auto* c=std::get_if<Check>(&step)) {
            out << indent << "check " << checkName(c->kind) << " -> " << c->failureId;
            ids(out, c->operands); selectors(out, c->selectors); out << '\n';
        } else if (const auto* w=std::get_if<BindingWrite>(&step)) {
            out << indent << (w->declaration?"declare ":"rebind ") << '$' << w->binding << "=%" << w->value << '\n';
        } else if (const auto* f=std::get_if<Flow>(&step)) {
            out << indent << (f->kind==Flow::Kind::Return?"return":f->kind==Flow::Kind::Break?"break":"continue");
            if (f->value) out << " %" << *f->value; out << '\n';
        } else {
            const auto& s=std::get<Structured>(step);
            if (s.kind==Structured::Kind::If) {
                out << indent << "if %" << s.condition << '\n';
                if (s.thenBlock) block(out,*s.thenBlock,indent+"  ");
                if (s.elseBlock) { out << indent << "else\n"; block(out,*s.elseBlock,indent+"  "); }
            } else if (s.kind==Structured::Kind::ForRange) {
                out << indent << "for_range $" << s.induction << " %" << s.start << ":%" << s.end << '\n';
                if (s.bodyBlock) block(out,*s.bodyBlock,indent+"  ");
            } else {
                out << indent << "while\n";
                if (s.conditionBlock) block(out,*s.conditionBlock,indent+"  ");
                if (s.conditionResult) out << indent << "condition %" << *s.conditionResult << '\n';
                if (s.bodyBlock) block(out,*s.bodyBlock,indent+"  ");
            }
        }
    }
    for (const auto& binding : b.bindings) out << indent << "bind " << binding.name << "=%" << binding.value << " $" << binding.id << (binding.mutableBinding ? " mut" : "") << '\n';
    if (b.terminated) out << indent << "end\n";
    else out << indent << "<unterminated>\n";
}
}
std::string dump(const Module& m) {
    std::ostringstream out;
    // Source identity is retained for diagnostics but omitted from the developer dump:
    // callers may supply an absolute machine path to the parser.
    out << "module\n";
    for (const auto& i : m.imports) out << "import " << i.path << " as " << i.alias << " unresolved\n";
    for (const auto& f : m.functions) {
        out << "fn @" << f.id << ' ' << f.name << " -> " << typeName(f.result) << '\n';
        for (const auto& p : f.parameters) out << "  param %" << p.id << " $" << p.binding << ' ' << p.name << ':' << typeName(p.type) << ' ' <<
            (p.access==AccessMode::Read?"read":p.access==AccessMode::MutableBorrow?"borrow_mut":"move") << '\n';
        block(out, f.body);
    }
    out << "initializer\n"; block(out, m.initializer);
    return out.str();
}
}
