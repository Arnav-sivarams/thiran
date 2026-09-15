#pragma once
#include "semantic/v0/Ir.hpp"
#include <optional>

namespace thiran::v0::semantic {
struct SemanticDiagnostic {
    std::string source, category, message;
    SourceSpan span;
    std::string format() const;
};
struct AnalysisResult {
    std::optional<Module> module;
    std::vector<SemanticDiagnostic> diagnostics;
};
AnalysisResult analyze(const thiran::v0::Module& syntax);
}
