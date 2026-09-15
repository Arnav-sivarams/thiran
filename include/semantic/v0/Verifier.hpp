#pragma once
#include "semantic/v0/Ir.hpp"

namespace thiran::v0::semantic {
struct VerificationResult { bool ok = false; std::vector<std::string> errors; };
VerificationResult verify(const Module&);
}
