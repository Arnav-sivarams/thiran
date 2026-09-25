#pragma once

#include "backend/v0/TensorRegion.hpp"
#include <string_view>

namespace thiran::v0::backend {
// STRICT_NATIVE: no evaluator or legacy execution fallback is present.
NativeResult extractStrictNative(const semantic::Module&, const analysis::OwnershipAnalysisResult&,
                                 const std::string& functionName, bool standalone);
std::string emitCpp20(const TensorRegion&, bool standalone, std::string_view testWrapper = {});
}
