#pragma once

#include "frontend/v0/Ast.hpp"
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace thiran::v0::semantic {
using ValueId = std::uint32_t;
using FunctionId = std::uint32_t;
using BindingId = std::uint32_t;
using BlockId = std::uint32_t;
enum class TypeKind { Bool, U8, I32, I64, U32, U64, F32, F64, Tensor, Buffer, Tuple, Invalid };
struct Type {
    TypeKind kind = TypeKind::Invalid;
    std::vector<Type> elements;
    std::uint32_t rank = 0;
    bool operator==(const Type&) const = default;
};
Type scalar(TypeKind kind);
Type tensor(Type element, std::uint32_t rank);
std::string typeName(const Type&);
bool validType(const Type&);
bool executableType(const Type&);
struct ShapeFact { std::vector<std::optional<std::int64_t>> extents; bool operator==(const ShapeFact&) const = default; };
enum class EffectClass { Pure, CheckedFailure, Mutation, Rng, Io, Transfer, Async };
enum class Op { Integer, Boolean, LoadBinding, TensorLiteral, Tuple, Negate, Add, Subtract, Multiply,
                ElementMultiply, Matmul, Index, Slice, Transpose, Sum, Call };
enum class CheckKind { Bounds, Broadcast, MatmulShape, Slice };
struct Selector {
    bool slice = false;
    std::optional<ValueId> index, start, end, step;
};
struct Instruction {
    ValueId id = 0;
    Op op = Op::Integer;
    Type type;
    ShapeFact shape;
    SourceSpan span;
    std::vector<ValueId> operands;
    std::vector<Selector> selectors;
    std::optional<std::int64_t> integer;
    std::optional<bool> boolean;
    FunctionId callee = 0;
    BindingId binding = 0;
    std::uint32_t axis = 0;
    bool borrowedView = false;
    EffectClass effect = EffectClass::Pure;
};
struct Check {
    CheckKind kind = CheckKind::Bounds;
    std::vector<ValueId> operands;
    std::vector<Selector> selectors;
    SourceSpan span;
    std::string failureId;
    std::uint32_t axis = 0;
    EffectClass effect = EffectClass::CheckedFailure;
};
struct Block;
struct BindingWrite { BindingId binding = 0; ValueId value = 0; bool declaration = false; SourceSpan span; };
struct Flow { enum class Kind { Return, Break, Continue }; Kind kind = Kind::Return; std::optional<ValueId> value; SourceSpan span; };
struct Structured {
    enum class Kind { If, ForRange, While }; Kind kind = Kind::If;
    SourceSpan span;
    ValueId condition = 0, start = 0, end = 0;
    BindingId induction = 0;
    std::shared_ptr<Block> thenBlock, elseBlock, conditionBlock, bodyBlock;
    std::optional<ValueId> conditionResult;
};
using Step = std::variant<Instruction, Check, BindingWrite, Flow, Structured>;
struct ParameterValue { ValueId id = 0; std::string name; Type type; ShapeFact shape; SourceSpan span;
    BindingId binding = 0;
    enum class Access { ReadOnly, ExclusiveMutable, Consuming } access = Access::ReadOnly; };
struct Binding { std::string name; ValueId value = 0; bool mutableBinding = false; SourceSpan span;
    BindingId id = 0; Type type; ShapeFact shape; };
struct Block {
    BlockId id = 0, parent = 0;
    std::vector<Step> steps;
    std::vector<Binding> bindings;
    std::optional<ValueId> returned;
    bool terminated = false;
};
struct Function {
    FunctionId id = 0;
    std::string name;
    SourceSpan span;
    bool exported = false;
    std::vector<ParameterValue> parameters;
    Type result;
    Block body;
};
struct Import { std::string path, alias; SourceSpan span; };
struct Module {
    std::string source;
    SourceSpan span;
    std::vector<Import> imports;
    std::vector<Function> functions;
    Block initializer;
};
std::string dump(const Module&);
}
