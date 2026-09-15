#pragma once
#include "semantic/v0/Ir.hpp"
#include <memory>

namespace thiran::v0::semantic {
struct RuntimeTensor { TypeKind dtype = TypeKind::I64; std::vector<std::int64_t> shape, values; };
struct RuntimeValue;
using RuntimeTuple = std::vector<RuntimeValue>;
struct RuntimeValue { std::variant<std::int64_t, bool, RuntimeTensor, RuntimeTuple> data; };
struct Observation {
    bool ok = false;
    std::optional<RuntimeValue> value;
    std::string errorId;
    std::string format() const;
};
Observation evaluateBinding(const Module&, const std::string& name);
Observation evaluateCall(const Module&, const std::string& name, const std::vector<RuntimeValue>& args);
}
