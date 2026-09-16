#pragma once

#include "analysis/v0/Ownership.hpp"
#include "semantic/v0/Evaluator.hpp"
#include <optional>

namespace thiran::v0::autodiff {
struct ReverseModeRequest {
    semantic::FunctionId function = 0;
    std::vector<std::size_t> wrtParameters;
};

struct AdDiagnostic {
    std::string code;
    std::string message;
};

struct SavedValue {
    std::size_t slot = 0;
    semantic::ValueId primal = 0;
    semantic::Type type;
    semantic::ShapeFact shape;
    analysis::ProvenanceKind provenance = analysis::ProvenanceKind::NoResource;
    std::set<analysis::ResourceId> resources;
    std::string reason;
    bool implicitCopy = false;
};

struct GradientResult {
    std::size_t parameterIndex = 0;
    semantic::Type type;
};

struct DifferentiationResult {
    std::optional<semantic::Module> module;
    std::vector<AdDiagnostic> diagnostics;
    semantic::FunctionId sourceFunction = 0;
    semantic::FunctionId forwardFunction = 1;
    semantic::FunctionId backwardFunction = 2;
    std::vector<std::size_t> wrtParameters;
    std::vector<SavedValue> saves;
    std::vector<GradientResult> gradients;
    bool ok() const { return module.has_value() && diagnostics.empty(); }
    std::string dump() const;
};

DifferentiationResult differentiate(const semantic::Module&,
                                    const analysis::OwnershipAnalysisResult&,
                                    const ReverseModeRequest&);
std::vector<AdDiagnostic> verify(const semantic::Module& source,
                                 const DifferentiationResult& result);
semantic::Observation executeVjp(const DifferentiationResult&,
                                 const std::vector<semantic::RuntimeValue>& arguments,
                                 const semantic::RuntimeValue& outputCotangent);
semantic::Observation executeGrad(const DifferentiationResult&,
                                  const std::vector<semantic::RuntimeValue>& arguments);
}
