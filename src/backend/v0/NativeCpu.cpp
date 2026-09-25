#include "backend/v0/NativeCpu.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace thiran::v0::backend {
namespace {

bool i64(const semantic::Type& type) { return type == semantic::scalar(semantic::TypeKind::I64); }
bool f32(const semantic::Type& type) { return type == semantic::scalar(semantic::TypeKind::F32); }
bool tensorNumeric(const semantic::Type& type) {
    return type.kind == semantic::TypeKind::Tensor && type.elements.size() == 1 &&
           (i64(type.elements[0]) || f32(type.elements[0])) && (type.rank == 1 || type.rank == 2);
}
bool tensorI64(const semantic::Type& type) { return tensorNumeric(type) && i64(type.elements[0]); }
bool elementwise(RegionOp op) {
    return op == RegionOp::Negate || op == RegionOp::Add ||
           op == RegionOp::Subtract || op == RegionOp::ElementMultiply;
}
std::string v(semantic::ValueId id) { return "v" + std::to_string(id); }
std::string data(semantic::ValueId id) { return "data_v" + std::to_string(id); }
std::string shapeName(semantic::ValueId id) { return "shape_v" + std::to_string(id); }
std::string element(semantic::ValueId id) { return "element_v" + std::to_string(id); }
std::string slotName(PhysicalSlotId id) { return "physical_slot_" + std::to_string(id); }

std::string valueList(const std::vector<semantic::ValueId>& ids) {
    std::string result = "{";
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index) result += ',';
        result += v(ids[index]);
    }
    return result + '}';
}

std::string concreteShape(const semantic::ShapeFact& fact) {
    std::string result = "{";
    for (std::size_t index = 0; index < fact.extents.size(); ++index) {
        if (index) result += ',';
        if (!fact.extents[index] || *fact.extents[index] < 0)
            throw std::invalid_argument("BACKEND-UNSUPPORTED: concrete tensor literal shape required");
        result += std::to_string(*fact.extents[index]);
    }
    return result + '}';
}

const RegionNode& findNode(const TensorRegion& region, semantic::ValueId id) {
    const auto found = std::find_if(region.nodes.begin(), region.nodes.end(),
        [&](const RegionNode& candidate) { return candidate.id == id; });
    if (found == region.nodes.end()) throw std::invalid_argument("BACKEND-UNSUPPORTED: missing region value");
    return *found;
}

semantic::ValueId canonicalAlias(const TensorRegion& region, semantic::ValueId id) {
    std::set<semantic::ValueId> seen;
    while (seen.insert(id).second) {
        const auto& current = findNode(region, id);
        if (current.op != RegionOp::Alias || current.dependencies.size() != 1) break;
        id = current.dependencies.front();
    }
    return id;
}

std::string cppScalar(const semantic::Type& type) {
    if (i64(type)) return "std::int64_t";
    if (f32(type)) return "float";
    throw std::invalid_argument("BACKEND-UNSUPPORTED: non-numeric scalar type");
}

std::size_t logicalIntermediates(const TensorRegion& region) {
    return static_cast<std::size_t>(std::count_if(region.nodes.begin(), region.nodes.end(),
        [&](const RegionNode& node) {
            return tensorNumeric(node.type) && node.op != RegionOp::Input &&
                   node.op != RegionOp::Alias && node.id != region.output;
        }));
}

std::size_t temporarySlots(const PhysicalPlan& plan) {
    return static_cast<std::size_t>(std::count_if(plan.slots.begin(), plan.slots.end(),
        [](const PhysicalSlot& slot) { return !slot.external && slot.reusable; }));
}

std::string emit(const TensorRegion& region, bool standalone, std::string_view testWrapper,
                 PhysicalPlanOptions options) {
    const auto verification = verifyRegion(region);
    if (!verification.ok()) throw std::invalid_argument(verification.errors.front());
    for (const auto& node : region.nodes) {
        const bool typeSupported = i64(node.type) || f32(node.type) || tensorNumeric(node.type);
        const bool opSupported = node.op == RegionOp::Input || node.op == RegionOp::Integer ||
            node.op == RegionOp::Float || node.op == RegionOp::TensorLiteral ||
            node.op == RegionOp::Alias || node.op == RegionOp::Copy || elementwise(node.op) ||
            node.op == RegionOp::Index;
        if (!typeSupported || !opSupported)
            throw std::invalid_argument("BACKEND-UNSUPPORTED: region is outside native CPU emission subset");
    }
    auto planned = buildPhysicalPlan(region, PhysicalDevice::Host, options);
    if (!planned.ok())
        throw std::invalid_argument(planned.errors.empty() ? "invalid physical plan" : planned.errors.front());
    const auto& plan = *planned.plan;

    std::ostringstream out;
    out << "#include \"storage/v0/Storage.hpp\"\n"
           "#include <cstdint>\n#include <iomanip>\n#include <iostream>\n#include <limits>\n"
           "#include <stdexcept>\n#include <string>\n#include <vector>\n";
    out << "using thiran::v0::storage::Tensor;\n"
           "struct SemanticFailure { const char* id; };\n"
           "static std::int64_t checked_add(std::int64_t a, std::int64_t b) {\n"
           "  if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) ||\n"
           "      (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b))\n"
           "    throw SemanticFailure{\"TH-SPEC-I64-OVERFLOW\"};\n"
           "  return a + b;\n}\n"
           "static std::int64_t checked_sub(std::int64_t a, std::int64_t b) {\n"
           "  if ((b < 0 && a > std::numeric_limits<std::int64_t>::max() + b) ||\n"
           "      (b > 0 && a < std::numeric_limits<std::int64_t>::min() + b))\n"
           "    throw SemanticFailure{\"TH-SPEC-I64-OVERFLOW\"};\n"
           "  return a - b;\n}\n"
           "static std::int64_t checked_mul(std::int64_t a, std::int64_t b) {\n"
           "  std::int64_t result = 0;\n"
           "  if (__builtin_mul_overflow(a, b, &result)) throw SemanticFailure{\"TH-SPEC-I64-OVERFLOW\"};\n"
           "  return result;\n}\n"
           "static std::int64_t checked_neg(std::int64_t a) {\n"
           "  if (a == std::numeric_limits<std::int64_t>::min()) throw SemanticFailure{\"TH-SPEC-I64-OVERFLOW\"};\n"
           "  return -a;\n}\n"
           "static std::size_t checked_flat_index(const std::vector<std::uint64_t>& shape,\n"
           "                                      const std::vector<std::int64_t>& coords) {\n"
           "  if (shape.size() != coords.size()) throw SemanticFailure{\"TH-SPEC-BOUNDS\"};\n"
           "  std::uint64_t linear = 0;\n"
           "  for (std::size_t axis = 0; axis < shape.size(); ++axis) {\n"
           "    if (coords[axis] < 0 || static_cast<std::uint64_t>(coords[axis]) >= shape[axis])\n"
           "      throw SemanticFailure{\"TH-SPEC-BOUNDS\"};\n"
           "    if (shape[axis] && linear > (std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(coords[axis])) / shape[axis])\n"
           "      throw std::runtime_error(\"BACKEND-INDEX-OVERFLOW\");\n"
           "    linear = linear * shape[axis] + static_cast<std::uint64_t>(coords[axis]);\n"
           "  }\n"
           "  if (linear > std::numeric_limits<std::size_t>::max()) throw std::runtime_error(\"BACKEND-INDEX-OVERFLOW\");\n"
           "  return static_cast<std::size_t>(linear);\n}\n";
    out << "// TH015 logical_intermediates=" << logicalIntermediates(region)
        << " physical_temporary_slots=" << temporarySlots(plan)
        << " fusion_groups=" << plan.fusionGroups.size() << "\n";
    out << "// Tensor values that are immutable aliases share one planned physical root.\n";

    const bool tensorOutput = tensorNumeric(region.outputType);
    out << (tensorOutput ? "Tensor" : cppScalar(region.outputType))
        << " native_f" << region.function << '(';
    for (std::size_t index = 0; index < region.inputs.size(); ++index) {
        if (index) out << ", ";
        const auto& input = findNode(region, region.inputs[index]);
        out << (tensorNumeric(input.type) ? "const Tensor& " : cppScalar(input.type) + " ") << v(input.id);
    }
    out << ") {\n";

    for (const auto& slot : plan.slots) {
        if (slot.external) continue;
        out << "  std::vector<" << (slot.dtype == storage::DType::I64 ? "std::int64_t" : "float")
            << "> " << slotName(slot.id) << ";\n";
    }

    std::set<FusionGroupId> emittedGroups;
    for (const auto& node : region.nodes) {
        if (node.op == RegionOp::Input) {
            if (tensorNumeric(node.type)) {
                out << "  auto " << data(node.id) << " = " << v(node.id)
                    << (tensorI64(node.type) ? ".logicalI64Values();\n" : ".logicalF32Values();\n");
                out << "  auto " << shapeName(node.id) << " = " << v(node.id) << ".descriptor().shape;\n";
            }
            continue;
        }
        if (node.op == RegionOp::Integer) {
            out << "  std::int64_t " << v(node.id) << " = ";
            if (*node.integer == std::numeric_limits<std::int64_t>::min())
                out << "std::numeric_limits<std::int64_t>::min()";
            else out << *node.integer << "LL";
            out << ";\n";
            continue;
        }
        if (node.op == RegionOp::Float) {
            out << "  float " << v(node.id) << " = " << *node.floating << "f;\n";
            continue;
        }
        if (node.op == RegionOp::Alias) {
            const auto* aliasPlan = plan.value(node.id);
            if (aliasPlan && !aliasPlan->materialized) continue;
            if (tensorNumeric(node.type)) {
                out << "  auto& " << data(node.id) << " = " << data(node.dependencies[0]) << ";\n"
                    << "  const auto& " << shapeName(node.id) << " = " << shapeName(node.dependencies[0]) << ";\n";
            } else {
                out << "  auto " << v(node.id) << " = " << v(node.dependencies[0]) << ";\n";
            }
            continue;
        }
        if (node.op == RegionOp::TensorLiteral || node.op == RegionOp::Copy) {
            const auto* valuePlan = plan.value(node.id);
            if (!valuePlan || !valuePlan->slot)
                throw std::invalid_argument("BACKEND-INVALID-PLAN: tensor materialization lacks slot");
            out << "  auto& " << data(node.id) << " = " << slotName(*valuePlan->slot) << ";\n";
            if (node.op == RegionOp::TensorLiteral) {
                out << "  " << data(node.id) << " = " << valueList(node.dependencies) << ";\n"
                    << "  std::vector<std::uint64_t> " << shapeName(node.id) << " = " << concreteShape(node.shape) << ";\n";
            } else {
                out << "  " << data(node.id) << " = " << data(node.dependencies[0]) << ";\n"
                    << "  auto " << shapeName(node.id) << " = " << shapeName(node.dependencies[0]) << ";\n";
            }
            continue;
        }
        if (node.op == RegionOp::Index) {
            out << "  " << cppScalar(node.type) << ' ' << v(node.id) << " = "
                << data(node.dependencies[0]) << "[checked_flat_index(" << shapeName(node.dependencies[0])
                << ", " << valueList(node.indices) << ")];\n";
            continue;
        }
        if (!elementwise(node.op)) continue;
        const auto* group = plan.groupFor(node.id);
        if (!group || group->nodes.front() != node.id || !emittedGroups.insert(group->id).second) continue;
        const auto& terminal = findNode(region, group->output);
        const auto* terminalPlan = plan.value(terminal.id);
        if (!terminalPlan || !terminalPlan->slot)
            throw std::invalid_argument("BACKEND-INVALID-PLAN: fused terminal lacks slot");
        const auto& first = findNode(region, group->nodes.front());
        const auto shapeSource = canonicalAlias(region, first.dependencies.front());
        std::set<semantic::ValueId> insideGroup(group->nodes.begin(), group->nodes.end());
        for (auto groupNodeId : group->nodes) {
            const auto& operation = findNode(region, groupNodeId);
            if (operation.op == RegionOp::Negate) continue;
            // Both operands are available either as materialized inputs or as
            // earlier scalar expressions in this group. Shape checks use the
            // first materialized ancestor for a fused dependency.
            auto shapeOperand = [&](semantic::ValueId id) {
                for (;;) {
                    id = canonicalAlias(region, id);
                    if (!insideGroup.contains(id)) return id;
                    id = findNode(region, id).dependencies.front();
                }
            };
            out << "  if (" << shapeName(shapeOperand(operation.dependencies[0])) << " != "
                << shapeName(shapeOperand(operation.dependencies[1])) << ")\n"
                << "    throw std::runtime_error(\"BACKEND-UNSUPPORTED: runtime unequal-shape elementwise operation\");\n";
        }
        out << "  auto " << shapeName(terminal.id) << " = " << shapeName(shapeSource) << ";\n"
            << "  auto& " << data(terminal.id) << " = " << slotName(*terminalPlan->slot) << ";\n"
            << "  " << data(terminal.id) << ".clear();\n"
            << "  " << data(terminal.id) << ".reserve(" << data(shapeSource) << ".size());\n"
            << "  for (std::size_t k = 0; k < " << data(shapeSource) << ".size(); ++k) {\n";
        std::set<semantic::ValueId> emittedInside;
        for (auto groupNodeId : group->nodes) {
            const auto& operation = findNode(region, groupNodeId);
            auto operand = [&](semantic::ValueId id) {
                id = canonicalAlias(region, id);
                return emittedInside.contains(id) ? element(id) : data(id) + "[k]";
            };
            out << "    " << (tensorI64(operation.type) ? "std::int64_t" : "volatile float")
                << ' ' << element(operation.id) << " = ";
            if (tensorI64(operation.type)) {
                if (operation.op == RegionOp::Negate)
                    out << "checked_neg(" << operand(operation.dependencies[0]) << ')';
                else {
                    const char* function = operation.op == RegionOp::Add ? "checked_add" :
                        operation.op == RegionOp::Subtract ? "checked_sub" : "checked_mul";
                    out << function << '(' << operand(operation.dependencies[0]) << ", "
                        << operand(operation.dependencies[1]) << ')';
                }
            } else {
                if (operation.op == RegionOp::Negate)
                    out << '-' << operand(operation.dependencies[0]);
                else {
                    const char symbol = operation.op == RegionOp::Add ? '+' :
                        operation.op == RegionOp::Subtract ? '-' : '*';
                    out << operand(operation.dependencies[0]) << ' ' << symbol << ' '
                        << operand(operation.dependencies[1]);
                }
            }
            out << ";\n";
            emittedInside.insert(operation.id);
        }
        out << "    " << data(terminal.id) << ".push_back(";
        if (!tensorI64(terminal.type)) out << "static_cast<float>(";
        out << element(terminal.id);
        if (!tensorI64(terminal.type)) out << ')';
        out << ");\n"
            << "  }\n";
    }

    const auto& outputNode = findNode(region, region.output);
    if (tensorOutput) {
        const auto* outputPlan = plan.value(region.output);
        if (outputPlan && outputPlan->root != region.output &&
            findNode(region, outputPlan->root).op == RegionOp::Input)
            out << "  return " << v(outputPlan->root) << ";\n";
        else
            out << "  return Tensor::materialize" << (tensorI64(outputNode.type) ? "I64" : "F32")
                << '(' << shapeName(region.output) << ", " << data(region.output) << ");\n";
    } else {
        out << "  return " << v(region.output) << ";\n";
    }
    out << "}\n";

    if (standalone) {
        out << "int main() {\n  try {\n    auto result = native_f" << region.function << "();\n";
        if (tensorOutput) {
            out << "    std::cout << \"{\\\"status\\\":\\\"ok\\\",\\\"kind\\\":\\\"tensor\\\",\\\"dtype\\\":\\\""
                << (tensorI64(region.outputType) ? "i64" : "f32")
                << "\\\",\\\"shape\\\":[\";\n"
                   "    auto shape = result.descriptor().shape; for (std::size_t k=0;k<shape.size();++k) { if(k) std::cout<<','; std::cout<<shape[k]; }\n"
                   "    std::cout << \"],\\\"values\\\":[\"; auto values = result."
                << (tensorI64(region.outputType) ? "logicalI64Values" : "logicalF32Values") << "();\n"
                   "    std::cout << std::setprecision(std::numeric_limits<"
                << (tensorI64(region.outputType) ? "std::int64_t" : "float") << ">::max_digits10);\n"
                   "    for (std::size_t k=0;k<values.size();++k) { if(k) std::cout<<','; std::cout<<values[k]; }\n"
                   "    std::cout << \"]}\\n\";\n";
        } else {
            out << "    std::cout << \"{\\\"status\\\":\\\"ok\\\",\\\"kind\\\":\\\"scalar\\\",\\\"dtype\\\":\\\""
                << (i64(region.outputType) ? "i64" : "f32") << "\\\",\\\"value\\\":\" << result << \"}\\n\";\n";
        }
        out << "    return 0;\n  } catch (const SemanticFailure& e) {\n"
               "    std::cout << \"{\\\"status\\\":\\\"error\\\",\\\"error_id\\\":\\\"\" << e.id << \"\\\"}\\n\"; return 0;\n"
               "  } catch (const std::exception& e) { std::cerr << \"native internal failure: \" << e.what() << '\\n'; return 2; }\n}\n";
    } else {
        out << testWrapper;
    }
    return out.str();
}

} // namespace

NativeResult extractStrictNative(const semantic::Module& module,
                                 const analysis::OwnershipAnalysisResult& facts,
                                 const std::string& name,
                                 bool standalone) {
    return extractStrictRegion(module, facts, name, standalone, NativeTarget::Cpu);
}

std::string emitCpp20(const TensorRegion& region, bool standalone, std::string_view testWrapper) {
    return emit(region, standalone, testWrapper, {});
}

std::string emitCpp20(const TensorRegion& region, bool standalone, std::string_view testWrapper,
                      PhysicalPlanOptions options) {
    return emit(region, standalone, testWrapper, options);
}

} // namespace thiran::v0::backend
