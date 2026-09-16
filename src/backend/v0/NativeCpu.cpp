#include "backend/v0/NativeCpu.hpp"
#include "semantic/v0/Verifier.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace thiran::v0::backend {
namespace {
std::string opName(RegionOp op) {
    switch(op) {
        case RegionOp::Input: return "Input"; case RegionOp::Integer: return "Integer";
        case RegionOp::TensorLiteral: return "TensorLiteral"; case RegionOp::Alias: return "Alias";
        case RegionOp::Add: return "Add"; case RegionOp::Index: return "Index";
        default: return "Unsupported";
    }
}
bool i64(const semantic::Type& t) { return t==semantic::scalar(semantic::TypeKind::I64); }
bool tensorI64(const semantic::Type& t) {
    return t.kind==semantic::TypeKind::Tensor && t.elements.size()==1 && i64(t.elements[0]) &&
           (t.rank==1 || t.rank==2);
}
bool shapeKnown(const semantic::ShapeFact& s) {
    for(auto x:s.extents) if(!x) return false;
    return true;
}
bool knownMismatch(const semantic::ShapeFact& a,const semantic::ShapeFact& b) {
    if(a.extents.size()!=b.extents.size()) return true;
    for(std::size_t d=0;d<a.extents.size();++d)
        if(a.extents[d] && b.extents[d] && a.extents[d]!=b.extents[d]) return true;
    return false;
}
std::string v(semantic::ValueId id) { return "v"+std::to_string(id); }
std::string vec(const std::vector<semantic::ValueId>& ids) {
    std::string out="{";
    for(std::size_t j=0;j<ids.size();++j) { if(j) out+=","; out+=v(ids[j]); }
    return out+"}";
}
std::string shape(const semantic::ShapeFact& s) {
    std::string out="{";
    for(std::size_t j=0;j<s.extents.size();++j) {
        if(j) out+=",";
        out+=s.extents[j] ? std::to_string(*s.extents[j]) : "?";
    }
    return out+"}";
}
}
std::string TensorRegion::dump() const {
    std::ostringstream out;
    out<<"region f"<<function<<" "<<name<<"\n";
    for(const auto& n:nodes) {
        out<<"  v"<<n.id<<" "<<opName(n.op)<<" "<<semantic::typeName(n.type)<<" "<<shape(n.shape);
        for(auto dep:n.dependencies) out<<" v"<<dep;
        for(auto idx:n.indices) out<<" [v"<<idx<<"]";
        if(n.integer) out<<" ="<<*n.integer;
        for(const auto& c:n.checks) out<<" check="<<c.failureId;
        out<<"\n";
    }
    out<<"  output v"<<output<<"\n";
    return out.str();
}
RegionVerification verifyRegion(const TensorRegion& r) {
    RegionVerification result;
    std::set<semantic::ValueId> all,seen;
    for(const auto& n:r.nodes) if(!all.insert(n.id).second || n.id==0) result.errors.push_back("TRV01 duplicate/zero region value ID");
    for(const auto& n:r.nodes) {
        for(auto id:n.dependencies) {
            if(!all.contains(id)) result.errors.push_back("TRV02 unknown dependency");
            else if(!seen.contains(id)) result.errors.push_back("TRV03 use before definition");
        }
        for(auto id:n.indices) {
            if(!all.contains(id)) result.errors.push_back("TRV02 unknown index dependency");
            else if(!seen.contains(id)) result.errors.push_back("TRV03 index use before definition");
        }
        auto get=[&](std::size_t j)->const RegionNode* {
            if(j>=n.dependencies.size()) return nullptr;
            for(const auto& x:r.nodes) if(x.id==n.dependencies[j]) return &x;
            return nullptr;
        };
        if(n.op==RegionOp::Add) {
            if(!tensorI64(n.type)) result.errors.push_back("TRV04 Add result is not Tensor<i64,R>");
            auto a=get(0),b=get(1);
            if(n.dependencies.size()!=2 || !a || !b || !tensorI64(a->type) || !tensorI64(b->type) ||
               a->type!=b->type || a->type!=n.type)
                result.errors.push_back("TRV05 Add input type mismatch");
            if(a && b && (knownMismatch(a->shape,b->shape) || knownMismatch(n.shape,a->shape)))
                result.errors.push_back("TRV06 Add known shape mismatch");
        } else if(n.op==RegionOp::Index) {
            auto a=get(0);
            if(!a || !tensorI64(a->type) || n.indices.size()!=a->type.rank)
                result.errors.push_back("TRV07 Index non-tensor/full-rank input");
            if(!i64(n.type)) result.errors.push_back("TRV08 Index result not i64 scalar");
            for(auto id:n.indices) {
                for(const auto& x:r.nodes) if(x.id==id && !i64(x.type)) result.errors.push_back("TRV07 Index coordinate not i64");
            }
        } else if(n.op==RegionOp::Unsupported) result.errors.push_back("TRV10 unsupported operation admitted");
        else if(n.op==RegionOp::Input && (!i64(n.type) && !tensorI64(n.type)))
            result.errors.push_back("TRV10 malformed input");
        else if(n.op==RegionOp::TensorLiteral) {
            if(!tensorI64(n.type) || !shapeKnown(n.shape)) result.errors.push_back("TRV10 malformed tensor literal");
            else {
                std::uint64_t count=1;
                for(auto x:n.shape.extents) {
                    if(*x<0 || (count && static_cast<std::uint64_t>(*x)>std::numeric_limits<std::uint64_t>::max()/count)) {
                        result.errors.push_back("TRV10 tensor literal element-count overflow"); break;
                    }
                    count*=static_cast<std::uint64_t>(*x);
                }
                if(count!=n.dependencies.size()) result.errors.push_back("TRV10 tensor literal element count mismatch");
            }
            for(auto id:n.dependencies) for(const auto& x:r.nodes) if(x.id==id && !i64(x.type))
                result.errors.push_back("TRV10 tensor literal non-i64 element");
        }
        else if(n.op==RegionOp::Alias && (n.dependencies.size()!=1 || !get(0) || get(0)->type!=n.type))
            result.errors.push_back("TRV10 malformed alias");
        else if(n.op==RegionOp::Integer && (!i64(n.type) || !n.integer))
            result.errors.push_back("TRV10 malformed integer");
        for(const auto& c:n.checks) {
            if((n.op==RegionOp::Index && (c.kind!=semantic::CheckKind::Bounds || c.failureId!="TH-SPEC-BOUNDS")) ||
               (n.op==RegionOp::Add && (c.kind!=semantic::CheckKind::Broadcast || c.failureId!="TH-SPEC-BROADCAST")) ||
               (n.op!=RegionOp::Index && n.op!=RegionOp::Add))
                result.errors.push_back("TRV10 malformed attached Check");
        }
        seen.insert(n.id);
    }
    if(!all.contains(r.output) || r.output==0) result.errors.push_back("TRV09 missing region output");
    else for(const auto& n:r.nodes) if(n.id==r.output && n.type!=r.outputType)
        result.errors.push_back("TRV09 output type mismatch");
    return result;
}
NativeResult extractStrictNative(const semantic::Module& m,const analysis::OwnershipAnalysisResult& facts,
                                 const std::string& name,bool standalone) {
    NativeResult out; out.coverage="entry: "+name+"\n";
    auto fail=[&](std::string why,std::string op="") -> NativeResult {
        out.diagnostic="BACKEND-UNSUPPORTED: "+why;
        if(!op.empty()) out.coverage+=op+" unsupported-native\n";
        out.coverage+="fallback: NONE\n"; return out;
    };
    if(!semantic::verify(m).ok) return fail("unverified semantic IR");
    if(!facts.ok() || !analysis::auditFacts(m,facts).empty()) return fail("ownership/effect legality failed");
    if(!m.imports.empty() || !m.initializer.steps.empty()) return fail("imports or top-level initialization");
    const semantic::Function* f=nullptr; std::size_t count=0;
    for(const auto& x:m.functions) if(x.name==name) { f=&x; ++count; }
    if(!f || count!=1) return fail("exactly one named function required");
    if(standalone && (name!="main" || !f->parameters.empty())) return fail("standalone main must have zero parameters");
    if(!i64(f->result) && !tensorI64(f->result)) return fail("result type");
    auto effects=facts.functionEffects.find(f->id);
    if(effects==facts.functionEffects.end() ||
       (effects->second.kinds & ~static_cast<analysis::EffectSet>(analysis::EffectKind::MayTrap))!=0)
        return fail("mutation/RNG/IO/Transfer/Async effect");
    TensorRegion r; r.function=f->id; r.name=f->name; r.outputType=f->result;
    std::map<semantic::BindingId,semantic::ValueId> bindings;
    std::map<semantic::ValueId,std::vector<semantic::Check>> pending;
    for(const auto& p:f->parameters) {
        if(p.access!=semantic::AccessMode::Read || (!i64(p.type) && !tensorI64(p.type))) return fail("parameter ABI/access mode");
        r.nodes.push_back({p.id,RegionOp::Input,p.type,p.shape,p.span}); r.inputs.push_back(p.id);
        bindings[p.binding]=p.id;
    }
    for(const auto& step:f->body.steps) {
        if(auto c=std::get_if<semantic::Check>(&step)) {
            if(c->kind!=semantic::CheckKind::Bounds && c->kind!=semantic::CheckKind::Broadcast)
                return fail("unsupported ordered Check");
            // A broadcast check is retained but the backend accepts only equal extents.
            // Static known unequal extents are rejected below; dynamic inequality fails
            // as backend-incomplete rather than as a language broadcast error.
            pending[c->operands.at(0)].push_back(*c); continue;
        }
        if(auto w=std::get_if<semantic::BindingWrite>(&step)) {
            if(!w->declaration) return fail("mutable rebinding");
            bindings[w->binding]=w->value; continue;
        }
        if(auto flow=std::get_if<semantic::Flow>(&step)) {
            if(flow->kind!=semantic::Flow::Kind::Return || !flow->value) return fail("non-return flow");
            r.output=*flow->value; continue;
        }
        if(std::holds_alternative<semantic::Structured>(step)) return fail("structured control flow","Structured");
        const auto& i=std::get<semantic::Instruction>(step);
        RegionNode n; n.id=i.id; n.type=i.type; n.shape=i.shape; n.span=i.span; n.dependencies=i.operands;
        switch(i.op) {
            case semantic::Op::Integer: n.op=RegionOp::Integer; n.integer=i.integer; break;
            case semantic::Op::TensorLiteral: n.op=RegionOp::TensorLiteral; break;
            case semantic::Op::LoadBinding:
                if(!bindings.contains(i.binding)) return fail("unknown binding");
                n.op=RegionOp::Alias; n.dependencies={bindings.at(i.binding)}; break;
            case semantic::Op::Add:
                if(!tensorI64(i.type)) return fail("scalar or mixed Add not in native subset","Add");
                n.op=RegionOp::Add;
                if(i.operands.size()!=2) return fail("Add arity","Add");
                for(const auto& x:r.nodes) if(x.id==i.operands[0])
                    for(const auto& y:r.nodes) if(y.id==i.operands[1] && knownMismatch(x.shape,y.shape))
                        return fail("native broadcasting not implemented","Add");
                if(pending.contains(i.operands[0])) { n.checks=pending.at(i.operands[0]); pending.erase(i.operands[0]); }
                break;
            case semantic::Op::Index:
                if(!i64(i.type) || i.operands.size()!=1 ||
                   i.selectors.size()!= (i.operands.empty()?0:([&] { for(const auto& x:r.nodes) if(x.id==i.operands[0]) return x.type.rank; return 0U; })()))
                    return fail("partial/slice Index","Index");
                n.op=RegionOp::Index;
                for(const auto& s:i.selectors) { if(s.slice || !s.index) return fail("slice Index","Index"); n.indices.push_back(*s.index); }
                if(pending.contains(i.operands.at(0))) { n.checks=pending.at(i.operands[0]); pending.erase(i.operands[0]); }
                break;
            case semantic::Op::Matmul: return fail("MatMul lowering deferred","MatMul");
            case semantic::Op::Slice: return fail("Slice lowering deferred","Slice");
            case semantic::Op::Transpose: return fail("Transpose lowering deferred","Transpose");
            case semantic::Op::Sum: return fail("Sum lowering deferred","Sum");
            case semantic::Op::Call: return fail("Call lowering deferred","Call");
            default: return fail("operation "+std::to_string(static_cast<int>(i.op)),"Unsupported");
        }
        r.nodes.push_back(std::move(n));
        out.coverage+=opName(r.nodes.back().op)+" native-cpu\n";
    }
    if(!pending.empty()) return fail("orphan ordered Check");
    auto rv=verifyRegion(r); if(!rv.ok()) return fail("region verifier: "+rv.errors.front());
    out.coverage+="fallback: NONE\n"; out.region=std::move(r); return out;
}
std::string emitCpp20(const TensorRegion& r,bool standalone,std::string_view testWrapper) {
    auto verification=verifyRegion(r);
    if(!verification.ok()) throw std::invalid_argument(verification.errors.front());
    std::ostringstream out;
    out<<"#include \"storage/v0/Storage.hpp\"\n#include <cstdint>\n#include <iostream>\n#include <limits>\n#include <stdexcept>\n#include <string>\n#include <vector>\n";
    out<<"using thiran::v0::storage::Tensor;\n"
          "struct SemanticFailure { const char* id; };\n"
          "static std::int64_t checked_add(std::int64_t a, std::int64_t b) {\n"
          "  if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) ||\n"
          "      (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b))\n"
          "    throw SemanticFailure{\"TH-SPEC-I64-OVERFLOW\"};\n"
          "  return a + b;\n}\n"
          "static std::int64_t checked_index(const Tensor& t, const std::vector<std::int64_t>& coords) {\n"
          "  std::vector<std::uint64_t> checked;\n"
          "  for (auto i : coords) { if (i < 0) throw SemanticFailure{\"TH-SPEC-BOUNDS\"}; checked.push_back(static_cast<std::uint64_t>(i)); }\n"
          "  try { return t.loadI64(checked); }\n"
          "  catch (const std::out_of_range&) { throw SemanticFailure{\"TH-SPEC-BOUNDS\"}; }\n"
          "  catch (const std::runtime_error& e) { if (std::string(e.what()) == \"TH-SPEC-BOUNDS\") throw SemanticFailure{\"TH-SPEC-BOUNDS\"}; throw; }\n}\n";
    out<<(tensorI64(r.outputType)?"Tensor":"std::int64_t")<<" native_f"<<r.function<<"(";
    for(std::size_t j=0;j<r.inputs.size();++j) {
        if(j) out<<", ";
        const auto& n=*std::find_if(r.nodes.begin(),r.nodes.end(),[&](const auto& x){return x.id==r.inputs[j];});
        out<<(tensorI64(n.type)?"const Tensor& ":"std::int64_t ")<<v(n.id);
    }
    out<<") {\n";
    for(const auto& n:r.nodes) {
        if(n.op==RegionOp::Input) continue;
        if(n.op==RegionOp::Integer) {
            out<<"  std::int64_t "<<v(n.id)<<" = ";
            if(*n.integer==std::numeric_limits<std::int64_t>::min())
                out<<"std::numeric_limits<std::int64_t>::min()";
            else out<<*n.integer<<"LL";
            out<<";\n";
        }
        if(n.op==RegionOp::Alias) out<<"  "<<(tensorI64(n.type)?"Tensor ":"std::int64_t ")<<v(n.id)<<" = "<<v(n.dependencies[0])<<";\n";
        if(n.op==RegionOp::TensorLiteral) {
            out<<"  Tensor "<<v(n.id)<<" = Tensor::materializeI64("<<shape(n.shape)<<", "<<vec(n.dependencies)<<");\n";
        }
        if(n.op==RegionOp::Add) {
            out<<"  if ("<<v(n.dependencies[0])<<".descriptor().shape != "<<v(n.dependencies[1])<<".descriptor().shape)\n"
               "    throw std::runtime_error(\"BACKEND-UNSUPPORTED: runtime unequal-shape Add\");\n";
            out<<"  auto a"<<n.id<<" = "<<v(n.dependencies[0])<<".logicalI64Values();\n"
               <<"  auto b"<<n.id<<" = "<<v(n.dependencies[1])<<".logicalI64Values();\n"
               <<"  std::vector<std::int64_t> sum"<<n.id<<"; sum"<<n.id<<".reserve(a"<<n.id<<".size());\n"
               <<"  for (std::size_t k = 0; k < a"<<n.id<<".size(); ++k) sum"<<n.id<<".push_back(checked_add(a"<<n.id<<"[k], b"<<n.id<<"[k]));\n"
               <<"  Tensor "<<v(n.id)<<" = Tensor::materializeI64("<<v(n.dependencies[0])<<".descriptor().shape, sum"<<n.id<<");\n";
        }
        if(n.op==RegionOp::Index) out<<"  std::int64_t "<<v(n.id)<<" = checked_index("<<v(n.dependencies[0])<<", "<<vec(n.indices)<<");\n";
    }
    out<<"  return "<<v(r.output)<<";\n}\n";
    if(standalone) {
        out<<"int main() {\n  try {\n    auto result = native_f"<<r.function<<"();\n";
        if(tensorI64(r.outputType)) {
            out<<"    std::cout << \"{\\\"status\\\":\\\"ok\\\",\\\"kind\\\":\\\"tensor\\\",\\\"dtype\\\":\\\"i64\\\",\\\"shape\\\":[\";\n"
                  "    auto shape = result.descriptor().shape; for (std::size_t k=0;k<shape.size();++k) { if(k) std::cout<<','; std::cout<<shape[k]; }\n"
                  "    std::cout << \"],\\\"values\\\":[\"; auto values = result.logicalI64Values();\n"
                  "    for (std::size_t k=0;k<values.size();++k) { if(k) std::cout<<','; std::cout<<values[k]; }\n"
                  "    std::cout << \"]}\\n\";\n";
        } else out<<"    std::cout << \"{\\\"status\\\":\\\"ok\\\",\\\"kind\\\":\\\"scalar\\\",\\\"dtype\\\":\\\"i64\\\",\\\"value\\\":\" << result << \"}\\n\";\n";
        out<<"    return 0;\n  } catch (const SemanticFailure& e) {\n"
              "    std::cout << \"{\\\"status\\\":\\\"error\\\",\\\"error_id\\\":\\\"\" << e.id << \"\\\"}\\n\"; return 0;\n"
              "  } catch (const std::exception& e) { std::cerr << \"native internal failure: \" << e.what() << '\\n'; return 2; }\n}\n";
    } else out<<testWrapper;
    return out.str();
}
}
