#include "backend/v0/TensorRegion.hpp"
#include "semantic/v0/Verifier.hpp"

#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace thiran::v0::backend {
namespace {

std::string opName(RegionOp op) {
    switch (op) {
    case RegionOp::Input: return "Input";
    case RegionOp::Integer: return "Integer";
    case RegionOp::Float: return "Float";
    case RegionOp::TensorLiteral: return "TensorLiteral";
    case RegionOp::Alias: return "Alias";
    case RegionOp::Copy: return "Copy";
    case RegionOp::Negate: return "Negate";
    case RegionOp::Add: return "Add";
    case RegionOp::Subtract: return "Subtract";
    case RegionOp::ElementMultiply: return "ElementMultiply";
    case RegionOp::Index: return "Index";
    case RegionOp::Unsupported: return "Unsupported";
    }
    return "Unsupported";
}

bool scalarI64(const semantic::Type& t) {
    return t == semantic::scalar(semantic::TypeKind::I64);
}

bool scalarF32(const semantic::Type& t) {
    return t == semantic::scalar(semantic::TypeKind::F32);
}

bool scalarNumeric(const semantic::Type& t) {
    return scalarI64(t) || scalarF32(t);
}

bool tensorNumeric(const semantic::Type& t) {
    return t.kind == semantic::TypeKind::Tensor && t.elements.size() == 1 &&
           scalarNumeric(t.elements[0]) && (t.rank == 1 || t.rank == 2);
}

bool gpuType(const semantic::Type& t) {
    return scalarNumeric(t) || tensorNumeric(t);
}

bool cpuType(const semantic::Type& t) {
    return gpuType(t);
}

bool shapeKnown(const semantic::ShapeFact& shape) {
    for (auto extent : shape.extents) if (!extent) return false;
    return true;
}

bool knownMismatch(const semantic::ShapeFact& a, const semantic::ShapeFact& b) {
    if (a.extents.size() != b.extents.size()) return true;
    for (std::size_t axis = 0; axis < a.extents.size(); ++axis)
        if (a.extents[axis] && b.extents[axis] && a.extents[axis] != b.extents[axis]) return true;
    return false;
}

std::string shapeText(const semantic::ShapeFact& shape) {
    std::string out = "{";
    for (std::size_t axis = 0; axis < shape.extents.size(); ++axis) {
        if (axis) out += ',';
        out += shape.extents[axis] ? std::to_string(*shape.extents[axis]) : "?";
    }
    return out + '}';
}

bool elementwise(RegionOp op) {
    return op == RegionOp::Add || op == RegionOp::Subtract || op == RegionOp::ElementMultiply;
}

}

std::string TensorRegion::dump() const {
    std::ostringstream out;
    out << "region f" << function << ' ' << name << '\n';
    for (const auto& node : nodes) {
        out << "  v" << node.id << ' ' << opName(node.op) << ' '
            << semantic::typeName(node.type) << ' ' << shapeText(node.shape);
        for (auto dependency : node.dependencies) out << " v" << dependency;
        for (auto index : node.indices) out << " [v" << index << ']';
        if (node.integer) out << " =" << *node.integer;
        if (node.floating) out << " =" << *node.floating << 'f';
        for (const auto& check : node.checks) out << " check=" << check.failureId;
        out << '\n';
    }
    out << "  output v" << output << '\n';
    return out.str();
}

RegionVerification verifyRegion(const TensorRegion& region) {
    RegionVerification result;
    std::set<semantic::ValueId> all;
    std::set<semantic::ValueId> seen;
    for (const auto& node : region.nodes)
        if (!all.insert(node.id).second || node.id == 0)
            result.errors.push_back("TRV01 duplicate/zero region value ID");

    for (const auto& node : region.nodes) {
        for (auto id : node.dependencies) {
            if (!all.contains(id)) result.errors.push_back("TRV02 unknown dependency");
            else if (!seen.contains(id)) result.errors.push_back("TRV03 use before definition");
        }
        for (auto id : node.indices) {
            if (!all.contains(id)) result.errors.push_back("TRV02 unknown index dependency");
            else if (!seen.contains(id)) result.errors.push_back("TRV03 index use before definition");
        }
        auto dependency = [&](std::size_t index) -> const RegionNode* {
            if (index >= node.dependencies.size()) return nullptr;
            for (const auto& candidate : region.nodes)
                if (candidate.id == node.dependencies[index]) return &candidate;
            return nullptr;
        };

        if (elementwise(node.op)) {
            if (!tensorNumeric(node.type))
                result.errors.push_back("TRV04 elementwise result is not Tensor<i64/f32,R>");
            const auto* lhs = dependency(0);
            const auto* rhs = dependency(1);
            if (node.dependencies.size() != 2 || !lhs || !rhs || !tensorNumeric(lhs->type) ||
                !tensorNumeric(rhs->type) || lhs->type != rhs->type || lhs->type != node.type)
                result.errors.push_back("TRV05 elementwise input type mismatch");
            if (lhs && rhs && (knownMismatch(lhs->shape, rhs->shape) || knownMismatch(node.shape, lhs->shape)))
                result.errors.push_back("TRV06 elementwise known shape mismatch");
        } else if (node.op == RegionOp::Negate) {
            const auto* input = dependency(0);
            if (node.dependencies.size() != 1 || !input || !tensorNumeric(node.type) || input->type != node.type ||
                knownMismatch(input->shape, node.shape))
                result.errors.push_back("TRV10 malformed Negate");
        } else if (node.op == RegionOp::Index) {
            const auto* input = dependency(0);
            if (!input || !tensorNumeric(input->type) || node.indices.size() != input->type.rank)
                result.errors.push_back("TRV07 Index non-tensor/full-rank input");
            if (!input || input->type.elements.empty() || node.type != input->type.elements[0])
                result.errors.push_back("TRV08 Index result dtype mismatch");
            for (auto id : node.indices)
                for (const auto& candidate : region.nodes)
                    if (candidate.id == id && !scalarI64(candidate.type))
                        result.errors.push_back("TRV07 Index coordinate not i64");
        } else if (node.op == RegionOp::Unsupported) {
            result.errors.push_back("TRV10 unsupported operation admitted");
        } else if (node.op == RegionOp::Input && !gpuType(node.type)) {
            result.errors.push_back("TRV10 malformed input");
        } else if (node.op == RegionOp::TensorLiteral) {
            if (!tensorNumeric(node.type) || !shapeKnown(node.shape)) {
                result.errors.push_back("TRV10 malformed tensor literal");
            } else {
                std::uint64_t count = 1;
                for (auto extent : node.shape.extents) {
                    if (*extent < 0 || (count && static_cast<std::uint64_t>(*extent) >
                                               std::numeric_limits<std::uint64_t>::max() / count)) {
                        result.errors.push_back("TRV10 tensor literal element-count overflow");
                        break;
                    }
                    count *= static_cast<std::uint64_t>(*extent);
                }
                if (count != node.dependencies.size())
                    result.errors.push_back("TRV10 tensor literal element count mismatch");
            }
            for (auto id : node.dependencies)
                for (const auto& candidate : region.nodes)
                    if (candidate.id == id && (node.type.elements.empty() || candidate.type != node.type.elements[0]))
                        result.errors.push_back("TRV10 tensor literal element dtype mismatch");
        } else if (node.op == RegionOp::Alias &&
                   (node.dependencies.size() != 1 || !dependency(0) || dependency(0)->type != node.type)) {
            result.errors.push_back("TRV10 malformed alias");
        } else if (node.op == RegionOp::Copy) {
            const auto* source = dependency(0);
            if (node.dependencies.size() != 1 || !source || !tensorNumeric(node.type) ||
                source->type != node.type || knownMismatch(source->shape, node.shape))
                result.errors.push_back("TRV10 malformed Copy");
        } else if (node.op == RegionOp::Integer && (!scalarI64(node.type) || !node.integer)) {
            result.errors.push_back("TRV10 malformed integer");
        } else if (node.op == RegionOp::Float && (!scalarF32(node.type) || !node.floating)) {
            result.errors.push_back("TRV10 malformed float");
        }

        for (const auto& check : node.checks) {
            const bool validIndex = node.op == RegionOp::Index &&
                check.kind == semantic::CheckKind::Bounds && check.failureId == "TH-SPEC-BOUNDS";
            const bool validElementwise = elementwise(node.op) &&
                check.kind == semantic::CheckKind::Broadcast && check.failureId == "TH-SPEC-BROADCAST";
            if (!validIndex && !validElementwise)
                result.errors.push_back("TRV10 malformed attached Check");
        }
        seen.insert(node.id);
    }

    if (!all.contains(region.output) || region.output == 0)
        result.errors.push_back("TRV09 missing region output");
    else
        for (const auto& node : region.nodes)
            if (node.id == region.output && node.type != region.outputType)
                result.errors.push_back("TRV09 output type mismatch");
    return result;
}

NativeResult extractStrictRegion(const semantic::Module& module,
                                 const analysis::OwnershipAnalysisResult& facts,
                                 const std::string& name,
                                 bool standalone,
                                 NativeTarget target) {
    NativeResult out;
    out.coverage = "entry: " + name + '\n';
    const std::string backendName = target == NativeTarget::Cpu ? "native-cpu" : "native-gpu";
    auto fail = [&](std::string reason, std::string operation = "") -> NativeResult {
        out.diagnostic = "BACKEND-UNSUPPORTED: " + reason;
        if (!operation.empty()) out.coverage += operation + " unsupported-" + backendName + '\n';
        out.coverage += "fallback: NONE\n";
        return out;
    };

    if (!semantic::verify(module).ok) return fail("unverified semantic IR");
    if (!facts.ok() || !analysis::auditFacts(module, facts).empty())
        return fail("ownership/effect legality failed");
    if (!module.imports.empty() || !module.initializer.steps.empty())
        return fail("imports or top-level initialization");

    const semantic::Function* function = nullptr;
    std::size_t matches = 0;
    for (const auto& candidate : module.functions)
        if (candidate.name == name) { function = &candidate; ++matches; }
    if (!function || matches != 1) return fail("exactly one named function required");
    if (standalone && (name != "main" || !function->parameters.empty()))
        return fail("standalone main must have zero parameters");

    std::function<bool(const semantic::Block&)> containsScan = [&](const semantic::Block& block) {
        for (const auto& step : block.steps)
            if (const auto* structured = std::get_if<semantic::Structured>(&step)) {
                if (structured->kind == semantic::Structured::Kind::Scan) return true;
                if (structured->thenBlock && containsScan(*structured->thenBlock)) return true;
                if (structured->elseBlock && containsScan(*structured->elseBlock)) return true;
                if (structured->conditionBlock && containsScan(*structured->conditionBlock)) return true;
                if (structured->bodyBlock && containsScan(*structured->bodyBlock)) return true;
            }
        return false;
    };
    if (containsScan(function->body)) return fail("Scan lowering deferred", "Scan");

    const auto acceptedType = target == NativeTarget::Cpu ? cpuType : gpuType;
    if (!acceptedType(function->result)) return fail("result type");
    const auto effects = facts.functionEffects.find(function->id);
    if (effects == facts.functionEffects.end() ||
        (effects->second.kinds & ~static_cast<analysis::EffectSet>(analysis::EffectKind::MayTrap)) != 0)
        return fail("mutation/RNG/IO/Transfer/Async effect");

    TensorRegion region;
    region.function = function->id;
    region.name = function->name;
    region.outputType = function->result;
    std::map<semantic::BindingId, semantic::ValueId> bindings;
    std::map<semantic::ValueId, std::vector<semantic::Check>> pending;
    const auto provenance = facts.valueProvenance.find(function->id);
    auto attachOwnership = [&](RegionNode& node) {
        if (provenance == facts.valueProvenance.end()) return;
        const auto found = provenance->second.find(node.id);
        if (found == provenance->second.end()) return;
        node.provenance = found->second.kind;
        node.resources = found->second.resources;
        node.viewRoots = found->second.viewRoots;
    };

    for (const auto& parameter : function->parameters) {
        if (parameter.access != semantic::AccessMode::Read || !acceptedType(parameter.type))
            return fail("parameter ABI/access mode");
        RegionNode input;
        input.id = parameter.id;
        input.op = RegionOp::Input;
        input.type = parameter.type;
        input.shape = parameter.shape;
        input.span = parameter.span;
        attachOwnership(input);
        region.nodes.push_back(std::move(input));
        region.inputs.push_back(parameter.id);
        bindings[parameter.binding] = parameter.id;
    }

    for (const auto& step : function->body.steps) {
        if (const auto* check = std::get_if<semantic::Check>(&step)) {
            if (check->kind != semantic::CheckKind::Bounds && check->kind != semantic::CheckKind::Broadcast)
                return fail("unsupported ordered Check");
            pending[check->operands.at(0)].push_back(*check);
            continue;
        }
        if (const auto* write = std::get_if<semantic::BindingWrite>(&step)) {
            if (!write->declaration) return fail("mutable rebinding");
            bindings[write->binding] = write->value;
            continue;
        }
        if (const auto* flow = std::get_if<semantic::Flow>(&step)) {
            if (flow->kind != semantic::Flow::Kind::Return || !flow->value) return fail("non-return flow");
            region.output = *flow->value;
            continue;
        }
        if (const auto* structured = std::get_if<semantic::Structured>(&step))
            return fail(structured->kind == semantic::Structured::Kind::Scan ? "Scan lowering deferred" :
                        "structured control flow",
                        structured->kind == semantic::Structured::Kind::Scan ? "Scan" : "Structured");

        const auto& instruction = std::get<semantic::Instruction>(step);
        RegionNode node;
        node.id = instruction.id;
        node.type = instruction.type;
        node.shape = instruction.shape;
        node.span = instruction.span;
        node.dependencies = instruction.operands;
        auto attachCheck = [&] {
            if (node.dependencies.empty() || !pending.contains(node.dependencies[0])) return;
            node.checks = pending.at(node.dependencies[0]);
            pending.erase(node.dependencies[0]);
        };
        auto admitElementwise = [&](RegionOp op, std::string_view operation) -> bool {
            if (!tensorNumeric(instruction.type) || instruction.operands.size() != 2) return false;
            node.op = op;
            for (const auto& lhs : region.nodes)
                if (lhs.id == instruction.operands[0])
                    for (const auto& rhs : region.nodes)
                        if (rhs.id == instruction.operands[1] && knownMismatch(lhs.shape, rhs.shape))
                            return false;
            attachCheck();
            (void)operation;
            return true;
        };

        switch (instruction.op) {
        case semantic::Op::Integer:
            node.op = RegionOp::Integer;
            node.integer = instruction.integer;
            break;
        case semantic::Op::Float:
            node.op = RegionOp::Float;
            node.floating = instruction.floating;
            break;
        case semantic::Op::TensorLiteral:
            node.op = RegionOp::TensorLiteral;
            break;
        case semantic::Op::LoadBinding:
            if (!bindings.contains(instruction.binding)) return fail("unknown binding");
            node.op = RegionOp::Alias;
            node.dependencies = {bindings.at(instruction.binding)};
            break;
        case semantic::Op::Copy:
            if (!tensorNumeric(instruction.type) || instruction.operands.size() != 1)
                return fail("Copy not in native tensor subset", "Copy");
            node.op = RegionOp::Copy;
            break;
        case semantic::Op::Negate:
            if (!tensorNumeric(instruction.type) || instruction.operands.size() != 1)
                return fail("tensor Negate not in native subset", "Negate");
            node.op = RegionOp::Negate;
            break;
        case semantic::Op::Add:
            if (!admitElementwise(RegionOp::Add, "Add"))
                return fail("scalar, mixed, or broadcasting Add not in native subset", "Add");
            break;
        case semantic::Op::Subtract:
            if (!admitElementwise(RegionOp::Subtract, "Subtract"))
                return fail("tensor Subtract not in native subset", "Subtract");
            break;
        case semantic::Op::ElementMultiply:
            if (!admitElementwise(RegionOp::ElementMultiply, "ElementMultiply"))
                return fail("tensor ElementMultiply not in native subset", "ElementMultiply");
            break;
        case semantic::Op::Index: {
            const RegionNode* input = nullptr;
            if (!instruction.operands.empty())
                for (const auto& candidate : region.nodes)
                    if (candidate.id == instruction.operands[0]) input = &candidate;
            if (!input || !tensorNumeric(input->type) || instruction.type != input->type.elements[0] ||
                instruction.operands.size() != 1 || instruction.selectors.size() != input->type.rank)
                return fail("partial/slice Index", "Index");
            node.op = RegionOp::Index;
            for (const auto& selector : instruction.selectors) {
                if (selector.slice || !selector.index) return fail("slice Index", "Index");
                node.indices.push_back(*selector.index);
            }
            attachCheck();
            break;
        }
        case semantic::Op::Matmul: return fail("MatMul lowering deferred", "MatMul");
        case semantic::Op::Sum: return fail("Sum lowering deferred", "Sum");
        case semantic::Op::Slice: return fail("Slice lowering deferred", "Slice");
        case semantic::Op::Transpose: return fail("Transpose lowering deferred", "Transpose");
        case semantic::Op::StopGradient: return fail("AD stop-gradient lowering deferred", "StopGradient");
        case semantic::Op::ZeroLike: return fail("AD helper lowering deferred", "ZeroLike");
        case semantic::Op::ReduceToShape: return fail("AD helper lowering deferred", "ReduceToShape");
        case semantic::Op::BroadcastToShape: return fail("AD helper lowering deferred", "BroadcastToShape");
        case semantic::Op::Call: return fail("Call lowering deferred", "Call");
        default:
            return fail("operation " + std::to_string(static_cast<int>(instruction.op)), "Unsupported");
        }
        attachOwnership(node);
        region.nodes.push_back(std::move(node));
        out.coverage += opName(region.nodes.back().op) + ' ' + backendName + '\n';
    }

    if (!pending.empty()) return fail("orphan ordered Check");
    const auto verification = verifyRegion(region);
    if (!verification.ok()) return fail("region verifier: " + verification.errors.front());
    out.coverage += "fallback: NONE\n";
    out.region = std::move(region);
    return out;
}

}
