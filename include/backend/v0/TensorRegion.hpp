#pragma once

#include "analysis/v0/Ownership.hpp"

#include <optional>
#include <string>
#include <vector>

namespace thiran::v0::backend {

// Device-neutral, straight-line tensor/dataflow IR. Backend selection and
// physical placement are deliberately absent from this representation.
enum class RegionOp {
    Input,
    Integer,
    Float,
    TensorLiteral,
    Alias,
    Negate,
    Add,
    Subtract,
    ElementMultiply,
    Index,
    Unsupported
};

struct RegionNode {
    semantic::ValueId id = 0;
    RegionOp op = RegionOp::Unsupported;
    semantic::Type type;
    semantic::ShapeFact shape;
    SourceSpan span;
    std::vector<semantic::ValueId> dependencies;
    std::vector<semantic::ValueId> indices;
    std::optional<std::int64_t> integer;
    std::optional<float> floating;
    std::vector<semantic::Check> checks;
};

struct TensorRegion {
    semantic::FunctionId function = 0;
    std::string name;
    std::vector<RegionNode> nodes;
    std::vector<semantic::ValueId> inputs;
    semantic::ValueId output = 0;
    semantic::Type outputType;
    std::string dump() const;
};

struct RegionVerification {
    std::vector<std::string> errors;
    bool ok() const { return errors.empty(); }
};

RegionVerification verifyRegion(const TensorRegion&);

struct NativeResult {
    std::optional<TensorRegion> region;
    std::string diagnostic;
    std::string coverage;
    bool ok() const { return region.has_value() && diagnostic.empty(); }
};

enum class NativeTarget { Cpu, Gpu };

// STRICT_NATIVE: no evaluator, CPU, GPU, legacy, or framework fallback is
// present. The target only controls which already-typed operations may enter
// the shared region; it is not a source-language device concept.
NativeResult extractStrictRegion(const semantic::Module&,
                                 const analysis::OwnershipAnalysisResult&,
                                 const std::string& functionName,
                                 bool standalone,
                                 NativeTarget);

}
