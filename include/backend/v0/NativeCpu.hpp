#pragma once

#include "analysis/v0/Ownership.hpp"
#include <optional>

namespace thiran::v0::backend {
enum class RegionOp { Input, Integer, TensorLiteral, Alias, Add, Index, Unsupported };
struct RegionNode {
    semantic::ValueId id = 0;
    RegionOp op = RegionOp::Unsupported;
    semantic::Type type;
    semantic::ShapeFact shape;
    SourceSpan span;
    std::vector<semantic::ValueId> dependencies;
    std::vector<semantic::ValueId> indices;
    std::optional<std::int64_t> integer;
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
struct RegionVerification { std::vector<std::string> errors; bool ok() const { return errors.empty(); } };
RegionVerification verifyRegion(const TensorRegion&);
struct NativeResult {
    std::optional<TensorRegion> region;
    std::string diagnostic;
    std::string coverage;
    bool ok() const { return region.has_value() && diagnostic.empty(); }
};
// STRICT_NATIVE: no evaluator or legacy execution fallback is present.
NativeResult extractStrictNative(const semantic::Module&, const analysis::OwnershipAnalysisResult&,
                                 const std::string& functionName, bool standalone);
std::string emitCpp20(const TensorRegion&, bool standalone, std::string_view testWrapper = {});
}
