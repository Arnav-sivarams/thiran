#include "training/v0/Checkpoint.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace thiran::v0::training {
namespace {

constexpr std::array<char,8> magic{'T','H','I','R','A','N','C','P'};
constexpr std::uint64_t maxCheckpointBytes = 128U << 20;
constexpr std::uint32_t maxParameters = 1U << 16;

CheckpointError makeError(std::string code, std::string message) {
    return {std::move(code), std::move(message)};
}

persistence::StoredValue storeValue(const semantic::RuntimeValue& runtime,
                                    const semantic::Type& type,
                                    const std::vector<std::int64_t>& shape) {
    persistence::StoredValue stored;
    stored.dtype = storage::DType::F32;
    persistence::BinaryWriter bytes;
    if (type == semantic::scalar(semantic::TypeKind::F32)) {
        const auto* value = std::get_if<float>(&runtime.data);
        if (!value || !shape.empty()) throw std::runtime_error("CKPT-DTYPE: expected scalar f32 value");
        stored.kind = persistence::StoredValueKind::Scalar;
        bytes.u32(std::bit_cast<std::uint32_t>(*value));
    } else {
        if (type.kind != semantic::TypeKind::Tensor || type.elements.size() != 1 ||
            type.elements.front() != semantic::scalar(semantic::TypeKind::F32))
            throw std::runtime_error("CKPT-DTYPE: checkpoint supports only TH-011 f32 state");
        const auto* tensor = std::get_if<semantic::RuntimeTensor>(&runtime.data);
        if (!tensor || tensor->dtype != semantic::TypeKind::F32 || tensor->shape != shape ||
            !tensor->values.empty())
            throw std::runtime_error("CKPT-DTYPE: runtime tensor representation mismatch");
        stored.kind = persistence::StoredValueKind::Tensor;
        for (auto extent : shape) {
            if (extent < 0) throw std::runtime_error("CKPT-SHAPE: negative concrete extent");
            stored.shape.push_back(static_cast<std::uint64_t>(extent));
        }
        for (float value : tensor->f32Values) bytes.u32(std::bit_cast<std::uint32_t>(value));
    }
    stored.bytes = bytes.take();
    if (const auto invalid = persistence::validateStoredValue(stored))
        throw std::runtime_error("CKPT-NUMERIC-PAYLOAD: " + *invalid);
    return stored;
}

semantic::RuntimeValue restoreValue(const persistence::StoredValue& stored) {
    if (stored.dtype != storage::DType::F32)
        throw std::runtime_error("CKPT-DTYPE: training state requires f32");
    persistence::BinaryReader reader(stored.bytes, "CKPT-NUMERIC-PAYLOAD");
    if (stored.kind == persistence::StoredValueKind::Scalar) {
        const float value = std::bit_cast<float>(reader.u32());
        if (!reader.done()) throw std::runtime_error("CKPT-NUMERIC-PAYLOAD: trailing scalar bytes");
        return semantic::RuntimeValue{value};
    }
    std::vector<float> values;
    const auto count = storage::checkedElementCount(stored.shape);
    if (count > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("CKPT-LENGTH: tensor element count exceeds host size");
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index)
        values.push_back(std::bit_cast<float>(reader.u32()));
    if (!reader.done()) throw std::runtime_error("CKPT-NUMERIC-PAYLOAD: trailing tensor bytes");
    std::vector<std::int64_t> shape;
    shape.reserve(stored.shape.size());
    for (auto extent : stored.shape) {
        if (extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::runtime_error("CKPT-SHAPE: extent exceeds semantic representation");
        shape.push_back(static_cast<std::int64_t>(extent));
    }
    return semantic::RuntimeValue{semantic::RuntimeTensor{
        semantic::TypeKind::F32, std::move(shape), {}, std::move(values)}};
}

std::vector<std::int64_t> signedShape(const persistence::StoredValue& value) {
    std::vector<std::int64_t> shape;
    shape.reserve(value.shape.size());
    for (auto extent : value.shape) {
        if (extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::runtime_error("CKPT-SHAPE: extent exceeds semantic representation");
        shape.push_back(static_cast<std::int64_t>(extent));
    }
    return shape;
}

std::optional<CheckpointError> checkStoredAgainst(const persistence::StoredValue& value,
                                                  const ParameterDescriptor& descriptor,
                                                  const std::vector<std::int64_t>* exactShape = nullptr) {
    if (value.dtype != storage::DType::F32)
        return makeError("CKPT-DTYPE", "parameter dtype disagrees with training schema");
    const bool scalar = descriptor.type == semantic::scalar(semantic::TypeKind::F32);
    if (scalar && value.kind != persistence::StoredValueKind::Scalar)
        return makeError("CKPT-RANK", "scalar parameter was stored as a tensor");
    if (!scalar) {
        if (descriptor.type.kind != semantic::TypeKind::Tensor ||
            value.kind != persistence::StoredValueKind::Tensor || value.shape.size() != descriptor.type.rank)
            return makeError("CKPT-RANK", "parameter rank disagrees with training schema");
        for (std::size_t axis = 0; axis < value.shape.size(); ++axis) {
            const auto expected = descriptor.expectedShape.extents.at(axis);
            if (expected && (*expected < 0 || value.shape[axis] != static_cast<std::uint64_t>(*expected)))
                return makeError("CKPT-SHAPE", "parameter shape disagrees with training schema");
        }
    }
    if (exactShape) {
        const auto restoredShape = signedShape(value);
        if (restoredShape != *exactShape)
            return makeError("CKPT-OPTIMIZER-STATE-SHAPE", "optimizer-state shape disagrees with parameter");
    }
    return {};
}

std::vector<std::byte> envelope(const std::vector<std::byte>& payload) {
    persistence::BinaryWriter writer;
    writer.raw(reinterpret_cast<const std::byte*>(magic.data()), magic.size());
    writer.u32(checkpointFormatVersion);
    writer.u32(trainingCheckpointAbiVersion);
    writer.u64(payload.size());
    writer.u64(persistence::fnv1a64(payload.data(), payload.size()));
    writer.raw(payload.data(), payload.size());
    return writer.take();
}

std::string exceptionCode(const std::string& message) {
    if (message.starts_with("CKPT-")) return message.substr(0, message.find(':'));
    if (message.starts_with("PERSISTENCE-RANK")) return "CKPT-RANK";
    if (message.starts_with("PERSISTENCE-VALUE")) return "CKPT-NUMERIC-PAYLOAD";
    return "CKPT-MALFORMED";
}

std::optional<CheckpointError> validateSchema(const TrainingCheckpointSchema& schema,
                                              const TrainingPlan& plan) {
    if (schema.modelIdentity.empty())
        return makeError("CKPT-MODEL-IDENTITY", "checkpoint schema model identity is empty");
    if (schema.planSignature != plan.signature)
        return makeError("CKPT-TRAINING-ABI", "checkpoint schema plan signature is incompatible");
    if (schema.parameters.size() != plan.parameters.size())
        return makeError("CKPT-PARAMETER-SCHEMA", "checkpoint schema parameter count mismatch");
    for (std::size_t index = 0; index < schema.parameters.size(); ++index) {
        const auto& item = schema.parameters[index];
        const auto& descriptor = plan.parameters[index];
        if (item.id != descriptor.id || item.type != descriptor.type)
            return makeError("CKPT-PARAMETER-SCHEMA", "checkpoint schema parameter identity/type mismatch");
        if (item.shape.size() != descriptor.type.rank)
            return makeError("CKPT-RANK", "checkpoint schema parameter rank mismatch");
        for (std::size_t axis = 0; axis < item.shape.size(); ++axis)
            if (item.shape[axis] < 0 || (descriptor.expectedShape.extents[axis] &&
                item.shape[axis] != *descriptor.expectedShape.extents[axis]))
                return makeError("CKPT-SHAPE", "checkpoint schema parameter shape mismatch");
    }
    return {};
}

} // namespace

std::vector<persistence::StoredParameter> storedParameters(const TrainingPlan& plan,
                                                           const TrainingState& state) {
    const auto errors = verifyTrainingState(plan, state);
    if (!errors.empty()) throw std::invalid_argument(errors.front().code + ": " + errors.front().message);
    std::vector<persistence::StoredParameter> result;
    result.reserve(state.parameters.size());
    for (std::size_t index = 0; index < state.parameters.size(); ++index) {
        const auto& descriptor = plan.parameters[index];
        const auto& parameter = state.parameters[index];
        result.push_back({parameter.id.value, descriptor.sourceParameterName,
                          storeValue(parameter.value, parameter.type, parameter.shape)});
    }
    return result;
}

CheckpointSchemaResult createTrainingCheckpointSchema(const std::string& modelIdentity,
                                                       const TrainingPlan& plan,
                                                       const TrainingState& initialState) {
    CheckpointSchemaResult result;
    const auto errors = verifyTrainingState(plan, initialState);
    if (!errors.empty()) {
        result.error = makeError("CKPT-STATE", errors.front().code + ": " + errors.front().message);
        return result;
    }
    if (initialState.step != 0) {
        result.error = makeError("CKPT-PARAMETER-SCHEMA", "checkpoint schema must derive from initial step zero");
        return result;
    }
    TrainingCheckpointSchema schema;
    schema.modelIdentity = modelIdentity;
    schema.planSignature = plan.signature;
    for (const auto& parameter : initialState.parameters)
        schema.parameters.push_back({parameter.id, parameter.type, parameter.shape});
    if (auto invalid = validateSchema(schema, plan)) { result.error = std::move(invalid); return result; }
    result.schema = std::move(schema);
    return result;
}

CheckpointSaveResult saveTrainingCheckpoint(const std::filesystem::path& path,
                                             const TrainingCheckpointSchema& schema,
                                             const TrainingPlan& plan,
                                             const TrainingState& state) {
    CheckpointSaveResult result;
    try {
        const auto planErrors = verifyTrainingPlan(plan);
        if (!planErrors.empty()) {
            result.error = makeError("CKPT-PLAN", planErrors.front().code + ": " + planErrors.front().message);
            return result;
        }
        if (auto invalid = validateSchema(schema, plan)) { result.error = std::move(invalid); return result; }
        const auto stateErrors = verifyTrainingState(plan, state);
        if (!stateErrors.empty()) {
            result.error = makeError("CKPT-STATE", stateErrors.front().code + ": " + stateErrors.front().message);
            return result;
        }
        for (std::size_t index = 0; index < state.parameters.size(); ++index)
            if (state.parameters[index].shape != schema.parameters[index].shape) {
                result.error = makeError("CKPT-SHAPE", "state shape disagrees with checkpoint schema");
                return result;
            }
        if (state.step == std::numeric_limits<std::uint64_t>::max()) {
            result.error = makeError("CKPT-STEP", "maximum step cannot be resumed");
            return result;
        }
        const auto parameters = storedParameters(plan, state);
        const auto parameterDigest = persistence::parameterDigest(parameters);
        persistence::BinaryWriter payload;
        payload.text(schema.modelIdentity);
        payload.u64(plan.signature);
        payload.u64(state.step);
        payload.u64(state.optimizer.step);
        payload.u8(static_cast<std::uint8_t>(plan.optimizer.kind));
        payload.u32(std::bit_cast<std::uint32_t>(plan.optimizer.learningRate));
        payload.u32(std::bit_cast<std::uint32_t>(plan.optimizer.momentum));
        payload.text(parameterDigest);
        payload.u32(static_cast<std::uint32_t>(parameters.size()));
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            payload.u64(plan.parameters[index].sourceParameterIndex);
            payload.u64(plan.parameters[index].position);
            persistence::writeStoredParameter(payload, parameters[index]);
        }
        payload.u32(static_cast<std::uint32_t>(state.optimizer.velocities.size()));
        for (const auto& velocity : state.optimizer.velocities) {
            payload.u64(velocity.id.value);
            persistence::writeStoredValue(payload,
                storeValue(velocity.value, velocity.type, velocity.shape));
        }
        const auto bytes = envelope(payload.bytes());
        if (const auto failure = persistence::transactionalWrite(path, bytes)) {
            result.error = makeError("CKPT-WRITE", *failure);
            return result;
        }
        result.metadata = CheckpointMetadata{schema.modelIdentity, parameterDigest, state.step, plan.optimizer};
    } catch (const std::exception& failure) {
        result.error = makeError(exceptionCode(failure.what()), failure.what());
    }
    return result;
}

CheckpointLoadResult loadTrainingCheckpoint(const std::filesystem::path& path,
                                             const TrainingCheckpointSchema& schema,
                                             const TrainingPlan& plan) {
    CheckpointLoadResult result;
    try {
        const auto planErrors = verifyTrainingPlan(plan);
        if (!planErrors.empty()) {
            result.error = makeError("CKPT-PLAN", planErrors.front().code + ": " + planErrors.front().message);
            return result;
        }
        if (auto invalid = validateSchema(schema, plan)) { result.error = std::move(invalid); return result; }
        const auto bytes = persistence::readFile(path, maxCheckpointBytes);
        persistence::BinaryReader reader(bytes, "CKPT");
        for (char expected : magic)
            if (reader.u8() != static_cast<std::uint8_t>(expected))
                throw std::runtime_error("CKPT-MAGIC: missing THIRANCP header");
        const auto version = reader.u32();
        if (version != checkpointFormatVersion)
            throw std::runtime_error("CKPT-FORMAT-VERSION: unsupported checkpoint format version");
        const auto abi = reader.u32();
        if (abi != trainingCheckpointAbiVersion)
            throw std::runtime_error("CKPT-TRAINING-ABI: unsupported training checkpoint ABI");
        const auto payloadSize = reader.u64();
        const auto expectedDigest = reader.u64();
        const auto payloadBytes = reader.raw(payloadSize, maxCheckpointBytes);
        if (!reader.done()) throw std::runtime_error("CKPT-TRAILING-DATA: bytes follow declared checkpoint payload");
        if (persistence::fnv1a64(payloadBytes.data(), payloadBytes.size()) != expectedDigest)
            throw std::runtime_error("CKPT-PAYLOAD-INTEGRITY: checkpoint payload digest mismatch");

        persistence::BinaryReader payload(payloadBytes, "CKPT");
        const auto modelIdentity = payload.text();
        if (modelIdentity != schema.modelIdentity)
            throw std::runtime_error("CKPT-MODEL-IDENTITY: checkpoint model identity mismatch");
        const auto planSignature = payload.u64();
        const auto step = payload.u64();
        const auto optimizerStep = payload.u64();
        if (step == std::numeric_limits<std::uint64_t>::max() || optimizerStep != step)
            throw std::runtime_error("CKPT-STEP: impossible or inconsistent step metadata");
        const auto optimizerKind = static_cast<OptimizerKind>(payload.u8());
        const auto learningRate = std::bit_cast<float>(payload.u32());
        const auto momentum = std::bit_cast<float>(payload.u32());
        if (optimizerKind != plan.optimizer.kind ||
            std::bit_cast<std::uint32_t>(learningRate) != std::bit_cast<std::uint32_t>(plan.optimizer.learningRate) ||
            std::bit_cast<std::uint32_t>(momentum) != std::bit_cast<std::uint32_t>(plan.optimizer.momentum))
            throw std::runtime_error("CKPT-OPTIMIZER-MISMATCH: optimizer identity/configuration mismatch");
        const auto recordedParameterDigest = payload.text();
        const auto parameterCount = payload.u32();
        if (parameterCount > maxParameters)
            throw std::runtime_error("CKPT-LENGTH: parameter count exceeds V0 limit");

        struct LoadedParameter {
            std::uint64_t sourceIndex = 0;
            std::uint64_t position = 0;
            persistence::StoredParameter parameter;
        };
        std::vector<LoadedParameter> loaded;
        loaded.reserve(parameterCount);
        std::set<std::uint64_t> seen;
        std::set<std::uint64_t> expectedIds;
        for (const auto& descriptor : plan.parameters) expectedIds.insert(descriptor.id.value);
        for (std::uint32_t index = 0; index < parameterCount; ++index) {
            LoadedParameter item;
            item.sourceIndex = payload.u64();
            item.position = payload.u64();
            item.parameter = persistence::readStoredParameter(payload);
            if (!seen.insert(item.parameter.id).second)
                throw std::runtime_error("CKPT-DUPLICATE-PARAMETER: duplicate ParameterId");
            if (!expectedIds.contains(item.parameter.id))
                throw std::runtime_error("CKPT-UNEXPECTED-PARAMETER: checkpoint contains an unexpected ParameterId");
            loaded.push_back(std::move(item));
        }
        for (auto id : expectedIds)
            if (!seen.contains(id))
                throw std::runtime_error("CKPT-MISSING-PARAMETER: checkpoint omits an expected ParameterId");
        if (loaded.size() != plan.parameters.size())
            throw std::runtime_error(loaded.size() < plan.parameters.size() ?
                "CKPT-MISSING-PARAMETER: checkpoint parameter count is too small" :
                "CKPT-UNEXPECTED-PARAMETER: checkpoint parameter count is too large");

        std::vector<persistence::StoredParameter> digestParameters;
        TrainingState state;
        state.planSignature = planSignature;
        state.step = step;
        state.optimizer.kind = optimizerKind;
        state.optimizer.step = optimizerStep;
        for (std::size_t index = 0; index < loaded.size(); ++index) {
            const auto& expected = plan.parameters[index];
            const auto& item = loaded[index];
            if (item.parameter.id != expected.id.value || item.position != index)
                throw std::runtime_error("CKPT-PARAMETER-ORDER: parameter ordering metadata is corrupted");
            if (item.sourceIndex != expected.sourceParameterIndex || item.parameter.name != expected.sourceParameterName)
                throw std::runtime_error("CKPT-PARAMETER-SCHEMA: parameter semantic identity is corrupted");
            if (auto invalid = checkStoredAgainst(item.parameter.value, expected))
                throw std::runtime_error(invalid->code + ": " + invalid->message);
            const auto shape = signedShape(item.parameter.value);
            if (shape != schema.parameters[index].shape)
                throw std::runtime_error("CKPT-SHAPE: parameter shape disagrees with checkpoint schema");
            state.parameters.push_back({expected.id, expected.type, shape,
                                        restoreValue(item.parameter.value)});
            digestParameters.push_back(item.parameter);
        }
        const auto actualParameterDigest = persistence::parameterDigest(digestParameters);
        if (actualParameterDigest != recordedParameterDigest)
            throw std::runtime_error("CKPT-PARAMETER-DIGEST: parameter identity digest mismatch");
        if (planSignature != plan.signature)
            throw std::runtime_error("CKPT-TRAINING-ABI: checkpoint plan signature is incompatible");

        const auto velocityCount = payload.u32();
        if (velocityCount > maxParameters)
            throw std::runtime_error("CKPT-LENGTH: optimizer state count exceeds V0 limit");
        std::set<std::uint64_t> velocityIds;
        for (std::uint32_t index = 0; index < velocityCount; ++index) {
            const auto id = payload.u64();
            auto value = persistence::readStoredValue(payload);
            if (!velocityIds.insert(id).second)
                throw std::runtime_error("CKPT-DUPLICATE-OPTIMIZER-STATE: duplicate optimizer ParameterId");
            const auto expected = std::find_if(plan.parameters.begin(), plan.parameters.end(),
                [&](const auto& descriptor) { return descriptor.id.value == id; });
            if (expected == plan.parameters.end())
                throw std::runtime_error("CKPT-UNEXPECTED-OPTIMIZER-STATE: optimizer state has an unknown ParameterId");
            const auto parameter = std::find_if(state.parameters.begin(), state.parameters.end(),
                [&](const auto& candidate) { return candidate.id.value == id; });
            if (auto invalid = checkStoredAgainst(value, *expected, &parameter->shape))
                throw std::runtime_error(invalid->code + ": " + invalid->message);
            state.optimizer.velocities.push_back({expected->id, expected->type,
                                                  parameter->shape, restoreValue(value)});
        }
        const auto expectedVelocities = optimizerKind == OptimizerKind::SGDMomentum ? plan.parameters.size() : 0U;
        if (state.optimizer.velocities.size() != expectedVelocities)
            throw std::runtime_error("CKPT-MISSING-OPTIMIZER-STATE: optimizer state count is incomplete");
        for (std::size_t index = 0; index < state.optimizer.velocities.size(); ++index)
            if (state.optimizer.velocities[index].id != plan.parameters[index].id)
                throw std::runtime_error("CKPT-OPTIMIZER-STATE-ORDER: optimizer state order is corrupted");
        if (!payload.done()) throw std::runtime_error("CKPT-TRAILING-DATA: bytes follow checkpoint state");
        const auto stateErrors = verifyTrainingState(plan, state);
        if (!stateErrors.empty())
            throw std::runtime_error("CKPT-STATE: " + stateErrors.front().code + " " + stateErrors.front().message);

        result.metadata = CheckpointMetadata{modelIdentity, actualParameterDigest, step,
                                             {optimizerKind, learningRate, momentum}};
        result.state = std::move(state);
    } catch (const std::exception& failure) {
        result.error = makeError(exceptionCode(failure.what()), failure.what());
    }
    return result;
}

} // namespace thiran::v0::training
