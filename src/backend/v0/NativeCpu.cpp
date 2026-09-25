#include "backend/v0/NativeCpu.hpp"
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace thiran::v0::backend {
namespace {
bool i64(const semantic::Type& t) { return t==semantic::scalar(semantic::TypeKind::I64); }
bool tensorI64(const semantic::Type& t) {
    return t.kind==semantic::TypeKind::Tensor && t.elements.size()==1 && i64(t.elements[0]) &&
           (t.rank==1 || t.rank==2);
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

NativeResult extractStrictNative(const semantic::Module& module,
                                 const analysis::OwnershipAnalysisResult& facts,
                                 const std::string& name,
                                 bool standalone) {
    return extractStrictRegion(module, facts, name, standalone, NativeTarget::Cpu);
}

std::string emitCpp20(const TensorRegion& r,bool standalone,std::string_view testWrapper) {
    auto verification=verifyRegion(r);
    if(!verification.ok()) throw std::invalid_argument(verification.errors.front());
    for(const auto& node:r.nodes) {
        if((!i64(node.type) && !tensorI64(node.type)) ||
           (node.op!=RegionOp::Input && node.op!=RegionOp::Integer &&
            node.op!=RegionOp::TensorLiteral && node.op!=RegionOp::Alias &&
            node.op!=RegionOp::Add && node.op!=RegionOp::Index))
            throw std::invalid_argument("BACKEND-UNSUPPORTED: region is outside native CPU emission subset");
    }
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
