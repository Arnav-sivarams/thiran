#include "model/v0/ModelBundle.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace thiran::v0::model {
namespace {

constexpr std::array<char,8> magic{'T','H','I','R','A','N','M','B'};
constexpr std::uint64_t maxBundleBytes = 2ULL << 30;
constexpr std::uint64_t maxEmbeddedArtifactBytes = 768U << 20;
constexpr std::uint32_t maxBundleItems = 1U << 16;

ModelError makeError(ModelErrorCategory category, std::string code, std::string message) {
    return {category, std::move(code), std::move(message)};
}

ModelBackend modelBackend(artifact::NativeBackend backend) {
    return backend == artifact::NativeBackend::Cpu ? ModelBackend::Cpu : ModelBackend::Gpu;
}

artifact::NativeBackend artifactBackend(ModelBackend backend) {
    return backend == ModelBackend::Cpu ? artifact::NativeBackend::Cpu : artifact::NativeBackend::Gpu;
}

std::optional<std::string> validateType(const artifact::ArtifactType& type) {
    if (type.dtype != storage::DType::I64 && type.dtype != storage::DType::F32)
        return "unsupported dtype";
    if (type.kind == artifact::ValueKind::Scalar) {
        if (type.rank != 0 || !type.extents.empty()) return "scalar has rank/extents";
    } else if (type.kind == artifact::ValueKind::Tensor) {
        if ((type.rank != 1 && type.rank != 2) || type.extents.size() != type.rank)
            return "tensor rank/extents mismatch";
    } else return "unknown value kind";
    return {};
}

artifact::ArtifactType storedType(const persistence::StoredValue& value) {
    artifact::ArtifactType type;
    type.kind = value.kind == persistence::StoredValueKind::Scalar ?
        artifact::ValueKind::Scalar : artifact::ValueKind::Tensor;
    type.dtype = value.dtype;
    type.rank = static_cast<std::uint32_t>(value.shape.size());
    for (auto extent : value.shape) type.extents.emplace_back(extent);
    return type;
}

artifact::ArtifactValue runtimeValue(const persistence::StoredValue& value) {
    persistence::BinaryReader reader(value.bytes, "MODEL-PARAMETER");
    if (value.kind == persistence::StoredValueKind::Scalar) {
        if (value.dtype == storage::DType::F32) {
            const auto result = std::bit_cast<float>(reader.u32());
            if (!reader.done()) throw std::runtime_error("MODEL-PARAMETER-PAYLOAD: trailing scalar bytes");
            return result;
        }
        const auto result = std::bit_cast<std::int64_t>(reader.u64());
        if (!reader.done()) throw std::runtime_error("MODEL-PARAMETER-PAYLOAD: trailing scalar bytes");
        return result;
    }
    const auto count = storage::checkedElementCount(value.shape);
    if (count > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("MODEL-PARAMETER-PAYLOAD: element count exceeds host size");
    if (value.dtype == storage::DType::F32) {
        std::vector<float> values;
        values.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t index = 0; index < count; ++index)
            values.push_back(std::bit_cast<float>(reader.u32()));
        if (!reader.done()) throw std::runtime_error("MODEL-PARAMETER-PAYLOAD: trailing tensor bytes");
        return storage::Tensor::materializeF32(value.shape, values);
    }
    std::vector<std::int64_t> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index)
        values.push_back(std::bit_cast<std::int64_t>(reader.u64()));
    if (!reader.done()) throw std::runtime_error("MODEL-PARAMETER-PAYLOAD: trailing tensor bytes");
    return storage::Tensor::materializeI64(value.shape, values);
}

std::optional<ModelError> validateRuntimeValue(const artifact::ArtifactType& type,
                                               const artifact::ArtifactValue& value,
                                               std::size_t index) {
    auto mismatch = [&](std::string detail) {
        return makeError(ModelErrorCategory::Execution, "MODEL-PUBLIC-INPUT-ABI",
                         "public input " + std::to_string(index) + ": " + detail);
    };
    if (type.kind == artifact::ValueKind::Scalar) {
        if (type.dtype == storage::DType::F32 && !std::holds_alternative<float>(value))
            return mismatch("expected scalar f32");
        if (type.dtype == storage::DType::I64 && !std::holds_alternative<std::int64_t>(value))
            return mismatch("expected scalar i64");
        return {};
    }
    const auto* tensor = std::get_if<storage::Tensor>(&value);
    if (!tensor) return mismatch("expected tensor");
    const auto& descriptor = tensor->descriptor();
    if (descriptor.dtype != type.dtype || descriptor.shape.size() != type.rank)
        return mismatch("dtype/rank mismatch");
    for (std::size_t axis = 0; axis < type.extents.size(); ++axis)
        if (type.extents[axis] && descriptor.shape[axis] != *type.extents[axis])
            return mismatch("shape mismatch");
    if (!tensor->isContiguousRowMajor()) return mismatch("non-contiguous layout");
    return {};
}

void writeType(persistence::BinaryWriter& writer, const artifact::ArtifactType& type) {
    writer.u8(static_cast<std::uint8_t>(type.kind));
    writer.u8(persistence::dtypeCode(type.dtype));
    writer.u32(type.rank);
    writer.u32(static_cast<std::uint32_t>(type.extents.size()));
    for (const auto& extent : type.extents) {
        writer.u8(extent.has_value());
        if (extent) writer.u64(*extent);
    }
}

artifact::ArtifactType readType(persistence::BinaryReader& reader) {
    artifact::ArtifactType type;
    type.kind = static_cast<artifact::ValueKind>(reader.u8());
    type.dtype = persistence::decodeDtype(reader.u8());
    type.rank = reader.u32();
    const auto count = reader.u32();
    if (count > 64) throw std::runtime_error("MODEL-ABI: rank exceeds V0 limit");
    for (std::uint32_t index = 0; index < count; ++index) {
        if (reader.u8()) type.extents.emplace_back(reader.u64());
        else type.extents.emplace_back(std::nullopt);
    }
    if (const auto invalid = validateType(type))
        throw std::runtime_error("MODEL-ABI: " + *invalid);
    return type;
}

std::vector<std::byte> serializeBundle(const ModelBundle& bundle) {
    persistence::BinaryWriter payload;
    payload.text(bundle.modelIdentity);
    payload.text(bundle.inferenceEntry);
    payload.text(bundle.parameterDigest);
    payload.u32(static_cast<std::uint32_t>(bundle.parameters.size()));
    for (const auto& parameter : bundle.parameters)
        persistence::writeStoredParameter(payload, parameter);
    payload.u32(static_cast<std::uint32_t>(bundle.publicInputs.size()));
    for (const auto& input : bundle.publicInputs) writeType(payload, input);
    writeType(payload, bundle.result);
    payload.u32(static_cast<std::uint32_t>(bundle.publicBindings.size()));
    for (const auto& binding : bundle.publicBindings) {
        payload.u32(binding.publicInputIndex);
        payload.u32(binding.artifactInputIndex);
    }
    payload.u32(static_cast<std::uint32_t>(bundle.parameterBindings.size()));
    for (const auto& binding : bundle.parameterBindings) {
        payload.u64(binding.parameterId);
        payload.u32(binding.artifactInputIndex);
    }
    payload.u32(static_cast<std::uint32_t>(bundle.artifacts.size()));
    for (const auto& embedded : bundle.artifacts) {
        payload.text(embedded.modelIdentity);
        payload.u8(static_cast<std::uint8_t>(embedded.artifact.manifest.backend));
        auto encoded = artifact::encodeArtifact(embedded.artifact);
        if (!encoded.ok()) throw std::runtime_error(
            "MODEL-EMBEDDED-ARTIFACT: " + encoded.error->code + " " + encoded.error->message);
        payload.u64(encoded.bytes->size());
        payload.raw(encoded.bytes->data(), encoded.bytes->size());
    }
    persistence::BinaryWriter writer;
    writer.raw(reinterpret_cast<const std::byte*>(magic.data()), magic.size());
    writer.u32(bundle.formatVersion);
    writer.u32(bundle.runtimeAbiVersion);
    writer.u64(payload.bytes().size());
    writer.u64(persistence::fnv1a64(payload.bytes().data(), payload.bytes().size()));
    writer.raw(payload.bytes().data(), payload.bytes().size());
    return writer.take();
}

struct PreparedExecution {
    const artifact::NativeArtifact* artifact = nullptr;
    std::vector<artifact::ArtifactValue> inputs;
    std::optional<ModelError> error;
};

PreparedExecution prepare(const ModelBundle& bundle, ModelBackend backend,
                          const std::vector<artifact::ArtifactValue>& publicInputs) {
    PreparedExecution result;
    if (auto invalid = validateModelBundle(bundle)) { result.error = std::move(invalid); return result; }
    const auto expectedBackend = artifactBackend(backend);
    const auto found = std::find_if(bundle.artifacts.begin(), bundle.artifacts.end(),
        [&](const auto& embedded) { return embedded.artifact.manifest.backend == expectedBackend; });
    if (found == bundle.artifacts.end()) {
        result.error = makeError(ModelErrorCategory::Execution, "MODEL-BACKEND-UNAVAILABLE",
            backend == ModelBackend::Gpu ? "GPU artifact is not present; no CPU fallback" :
                                           "CPU artifact is not present");
        return result;
    }
    if (publicInputs.size() != bundle.publicInputs.size()) {
        result.error = makeError(ModelErrorCategory::Execution, "MODEL-PUBLIC-INPUT-COUNT",
                                 "public inference input count mismatch");
        return result;
    }
    for (std::size_t index = 0; index < publicInputs.size(); ++index)
        if (auto invalid = validateRuntimeValue(bundle.publicInputs[index], publicInputs[index], index)) {
            result.error = std::move(invalid);
            return result;
        }
    result.inputs.resize(found->artifact.manifest.entry.parameters.size());
    std::vector<bool> assigned(result.inputs.size(), false);
    for (const auto& binding : bundle.publicBindings) {
        result.inputs[binding.artifactInputIndex] = publicInputs[binding.publicInputIndex];
        assigned[binding.artifactInputIndex] = true;
    }
    for (const auto& binding : bundle.parameterBindings) {
        const auto parameter = std::find_if(bundle.parameters.begin(), bundle.parameters.end(),
            [&](const auto& candidate) { return candidate.id == binding.parameterId; });
        try { result.inputs[binding.artifactInputIndex] = runtimeValue(parameter->value); }
        catch (const std::exception& failure) {
            result.error = makeError(ModelErrorCategory::Execution, "MODEL-PARAMETER-PAYLOAD", failure.what());
            return result;
        }
        assigned[binding.artifactInputIndex] = true;
    }
    if (std::find(assigned.begin(), assigned.end(), false) != assigned.end()) {
        result.error = makeError(ModelErrorCategory::Execution, "MODEL-MISSING-BINDING",
                                 "artifact input is not bound");
        return result;
    }
    result.artifact = &found->artifact;
    return result;
}

std::string exceptionCode(const std::string& message) {
    if (message.starts_with("MODEL-")) return message.substr(0, message.find(':'));
    if (message.starts_with("PERSISTENCE-")) return "MODEL-MALFORMED";
    return "MODEL-MALFORMED";
}

} // namespace

std::optional<ModelError> validateModelBundle(const ModelBundle& bundle) {
    if (bundle.formatVersion != modelBundleFormatVersion)
        return makeError(ModelErrorCategory::Load, "MODEL-FORMAT-VERSION", "unsupported model bundle version");
    if (bundle.runtimeAbiVersion != modelRuntimeAbiVersion)
        return makeError(ModelErrorCategory::Load, "MODEL-RUNTIME-ABI", "unsupported model runtime ABI");
    if (bundle.modelIdentity.empty() || bundle.inferenceEntry.empty())
        return makeError(ModelErrorCategory::Load, "MODEL-IDENTITY", "model/entry identity is empty");
    if (const auto invalid = validateType(bundle.result))
        return makeError(ModelErrorCategory::Load, "MODEL-PUBLIC-ABI", *invalid);
    for (const auto& input : bundle.publicInputs)
        if (const auto invalid = validateType(input))
            return makeError(ModelErrorCategory::Load, "MODEL-PUBLIC-ABI", *invalid);

    std::set<std::uint64_t> parameterIds;
    std::set<std::string> parameterNames;
    for (const auto& parameter : bundle.parameters) {
        if (parameter.id == 0 || !parameterIds.insert(parameter.id).second)
            return makeError(ModelErrorCategory::Load, "MODEL-DUPLICATE-PARAMETER", "duplicate/zero ParameterId");
        if (parameter.name.empty() || !parameterNames.insert(parameter.name).second)
            return makeError(ModelErrorCategory::Load, "MODEL-DUPLICATE-PARAMETER", "duplicate/empty parameter name");
        if (const auto invalid = persistence::validateStoredValue(parameter.value))
            return makeError(ModelErrorCategory::Load, "MODEL-PARAMETER-SCHEMA", *invalid);
    }
    if (persistence::parameterDigest(bundle.parameters) != bundle.parameterDigest)
        return makeError(ModelErrorCategory::Load, "MODEL-PARAMETER-DIGEST", "frozen parameter digest mismatch");
    if (bundle.artifacts.empty())
        return makeError(ModelErrorCategory::Load, "MODEL-BACKEND-UNAVAILABLE", "bundle contains no native artifact");

    const artifact::EntryPoint* commonEntry = nullptr;
    std::set<artifact::NativeBackend> backends;
    for (const auto& embedded : bundle.artifacts) {
        if (embedded.modelIdentity != bundle.modelIdentity)
            return makeError(ModelErrorCategory::Load, "MODEL-ARTIFACT-IDENTITY",
                             "embedded artifact model identity mismatch");
        if (!backends.insert(embedded.artifact.manifest.backend).second)
            return makeError(ModelErrorCategory::Load, "MODEL-DUPLICATE-BACKEND", "duplicate backend artifact");
        auto encoded = artifact::encodeArtifact(embedded.artifact);
        if (!encoded.ok())
            return makeError(ModelErrorCategory::Load, "MODEL-EMBEDDED-ARTIFACT",
                             encoded.error->code + ": " + encoded.error->message);
        const auto& entry = embedded.artifact.manifest.entry;
        if (entry.name != bundle.inferenceEntry)
            return makeError(ModelErrorCategory::Load, "MODEL-ARTIFACT-IDENTITY", "artifact entry identity mismatch");
        if (!commonEntry) commonEntry = &entry;
        else if (entry != *commonEntry)
            return makeError(ModelErrorCategory::Load, "MODEL-ARTIFACT-SIGNATURE",
                             "CPU/GPU artifact signatures disagree");
    }
    if (!commonEntry || commonEntry->result != bundle.result)
        return makeError(ModelErrorCategory::Load, "MODEL-PUBLIC-ABI", "artifact result disagrees with public ABI");

    std::set<std::uint32_t> slots;
    std::set<std::uint32_t> publicIndices;
    for (const auto& binding : bundle.publicBindings) {
        if (binding.publicInputIndex >= bundle.publicInputs.size() ||
            binding.artifactInputIndex >= commonEntry->parameters.size())
            return makeError(ModelErrorCategory::Load, "MODEL-PUBLIC-BINDING", "public binding is out of range");
        if (!publicIndices.insert(binding.publicInputIndex).second || !slots.insert(binding.artifactInputIndex).second)
            return makeError(ModelErrorCategory::Load, "MODEL-DUPLICATE-BINDING", "duplicate public/artifact binding");
        if (bundle.publicInputs[binding.publicInputIndex] != commonEntry->parameters[binding.artifactInputIndex])
            return makeError(ModelErrorCategory::Load, "MODEL-PUBLIC-ABI", "public input ABI disagrees with artifact slot");
    }
    if (publicIndices.size() != bundle.publicInputs.size())
        return makeError(ModelErrorCategory::Load, "MODEL-MISSING-BINDING", "a public input is not bound");

    std::set<std::uint64_t> boundParameters;
    for (const auto& binding : bundle.parameterBindings) {
        if (!parameterIds.contains(binding.parameterId))
            return makeError(ModelErrorCategory::Load, "MODEL-UNEXPECTED-PARAMETER", "binding references unknown ParameterId");
        if (binding.artifactInputIndex >= commonEntry->parameters.size())
            return makeError(ModelErrorCategory::Load, "MODEL-PARAMETER-BINDING", "parameter binding is out of range");
        if (!boundParameters.insert(binding.parameterId).second || !slots.insert(binding.artifactInputIndex).second)
            return makeError(ModelErrorCategory::Load, "MODEL-DUPLICATE-BINDING", "duplicate parameter/artifact binding");
        const auto parameter = std::find_if(bundle.parameters.begin(), bundle.parameters.end(),
            [&](const auto& candidate) { return candidate.id == binding.parameterId; });
        if (storedType(parameter->value) != commonEntry->parameters[binding.artifactInputIndex])
            return makeError(ModelErrorCategory::Load, "MODEL-PARAMETER-ABI",
                             "frozen parameter dtype/rank/shape disagrees with artifact slot");
    }
    if (boundParameters.size() != parameterIds.size())
        return makeError(ModelErrorCategory::Load, "MODEL-MISSING-PARAMETER", "a frozen parameter is not bound");
    if (slots.size() != commonEntry->parameters.size())
        return makeError(ModelErrorCategory::Load, "MODEL-MISSING-BINDING", "an artifact input slot is not bound");
    return {};
}

ModelBundleResult createModelBundle(const DeploymentSnapshot& snapshot,
                                    std::vector<EmbeddedArtifact> artifacts) {
    ModelBundleResult result;
    ModelBundle bundle;
    bundle.modelIdentity = snapshot.modelIdentity;
    bundle.inferenceEntry = snapshot.inferenceEntry;
    bundle.parameters = snapshot.parameters;
    bundle.publicInputs = snapshot.publicInputs;
    bundle.result = snapshot.result;
    bundle.publicBindings = snapshot.publicBindings;
    bundle.parameterBindings = snapshot.parameterBindings;
    bundle.parameterDigest = snapshot.parameterDigest;
    bundle.artifacts = std::move(artifacts);
    if (auto invalid = validateModelBundle(bundle)) { result.error = std::move(invalid); return result; }
    result.bundle = std::move(bundle);
    return result;
}

ModelWriteResult writeModelBundle(const ModelBundle& bundle, const std::filesystem::path& path) {
    ModelWriteResult result;
    if (auto invalid = validateModelBundle(bundle)) { result.error = std::move(invalid); return result; }
    try {
        const auto bytes = serializeBundle(bundle);
        if (const auto failure = persistence::transactionalWrite(path, bytes)) {
            result.error = makeError(ModelErrorCategory::Build, "MODEL-WRITE", *failure);
            return result;
        }
        result.success = true;
    } catch (const std::exception& failure) {
        result.error = makeError(ModelErrorCategory::Build, exceptionCode(failure.what()), failure.what());
    }
    return result;
}

ModelBundleResult loadModelBundle(const std::filesystem::path& path,
                                  std::optional<std::string> expectedModelIdentity) {
    ModelBundleResult result;
    try {
        const auto bytes = persistence::readFile(path, maxBundleBytes);
        persistence::BinaryReader reader(bytes, "MODEL");
        for (char expected : magic)
            if (reader.u8() != static_cast<std::uint8_t>(expected))
                throw std::runtime_error("MODEL-MAGIC: missing THIRANMB header");
        ModelBundle bundle;
        bundle.formatVersion = reader.u32();
        bundle.runtimeAbiVersion = reader.u32();
        if (bundle.formatVersion != modelBundleFormatVersion)
            throw std::runtime_error("MODEL-FORMAT-VERSION: unsupported model bundle version");
        if (bundle.runtimeAbiVersion != modelRuntimeAbiVersion)
            throw std::runtime_error("MODEL-RUNTIME-ABI: unsupported model runtime ABI");
        const auto payloadSize = reader.u64();
        const auto digest = reader.u64();
        const auto payloadBytes = reader.raw(payloadSize, maxBundleBytes);
        if (!reader.done()) throw std::runtime_error("MODEL-TRAILING-DATA: bytes follow declared bundle payload");
        if (persistence::fnv1a64(payloadBytes.data(), payloadBytes.size()) != digest)
            throw std::runtime_error("MODEL-BUNDLE-INTEGRITY: bundle payload digest mismatch");
        persistence::BinaryReader payload(payloadBytes, "MODEL");
        bundle.modelIdentity = payload.text();
        if (expectedModelIdentity && bundle.modelIdentity != *expectedModelIdentity)
            throw std::runtime_error("MODEL-IDENTITY: unexpected model identity");
        bundle.inferenceEntry = payload.text();
        bundle.parameterDigest = payload.text();
        const auto parameterCount = payload.u32();
        if (parameterCount > maxBundleItems) throw std::runtime_error("MODEL-LENGTH: excessive parameter count");
        for (std::uint32_t index = 0; index < parameterCount; ++index)
            bundle.parameters.push_back(persistence::readStoredParameter(payload));
        const auto publicCount = payload.u32();
        if (publicCount > maxBundleItems) throw std::runtime_error("MODEL-LENGTH: excessive public input count");
        for (std::uint32_t index = 0; index < publicCount; ++index)
            bundle.publicInputs.push_back(readType(payload));
        bundle.result = readType(payload);
        const auto publicBindingCount = payload.u32();
        if (publicBindingCount > maxBundleItems) throw std::runtime_error("MODEL-LENGTH: excessive public binding count");
        for (std::uint32_t index = 0; index < publicBindingCount; ++index)
            bundle.publicBindings.push_back({payload.u32(), payload.u32()});
        const auto parameterBindingCount = payload.u32();
        if (parameterBindingCount > maxBundleItems) throw std::runtime_error("MODEL-LENGTH: excessive parameter binding count");
        for (std::uint32_t index = 0; index < parameterBindingCount; ++index)
            bundle.parameterBindings.push_back({payload.u64(), payload.u32()});
        const auto artifactCount = payload.u32();
        if (artifactCount == 0 || artifactCount > 2)
            throw std::runtime_error("MODEL-BACKEND-COUNT: bundle requires one or two artifacts");
        for (std::uint32_t index = 0; index < artifactCount; ++index) {
            EmbeddedArtifact embedded;
            embedded.modelIdentity = payload.text();
            const auto declaredBackend = static_cast<artifact::NativeBackend>(payload.u8());
            const auto artifactBytes = payload.raw(payload.u64(), maxEmbeddedArtifactBytes);
            auto loaded = artifact::loadArtifactBytes(artifactBytes);
            if (!loaded.ok())
                throw std::runtime_error("MODEL-EMBEDDED-ARTIFACT: " + loaded.error->code + " " + loaded.error->message);
            if (loaded.artifact->manifest.backend != declaredBackend)
                throw std::runtime_error("MODEL-ARTIFACT-BACKEND: embedded backend declaration mismatch");
            embedded.artifact = std::move(*loaded.artifact);
            bundle.artifacts.push_back(std::move(embedded));
        }
        if (!payload.done()) throw std::runtime_error("MODEL-TRAILING-DATA: bytes follow model bundle state");
        if (auto invalid = validateModelBundle(bundle)) { result.error = std::move(invalid); return result; }
        result.bundle = std::move(bundle);
    } catch (const std::exception& failure) {
        result.error = makeError(ModelErrorCategory::Load, exceptionCode(failure.what()), failure.what());
    }
    return result;
}

std::string inspectModelBundle(const ModelBundle& bundle) {
    std::ostringstream out;
    out << "format_version=" << bundle.formatVersion << '\n'
        << "runtime_abi=" << bundle.runtimeAbiVersion << '\n'
        << "model_identity=" << bundle.modelIdentity << '\n'
        << "inference_entry=" << bundle.inferenceEntry << '\n'
        << "public_inputs=" << bundle.publicInputs.size() << '\n'
        << "frozen_parameters=" << bundle.parameters.size() << '\n'
        << "parameter_digest=" << bundle.parameterDigest << '\n';
    for (std::size_t index = 0; index < bundle.parameters.size(); ++index) {
        const auto& parameter = bundle.parameters[index];
        out << "parameter[" << index << "].id=" << parameter.id << '\n'
            << "parameter[" << index << "].name=" << parameter.name << '\n';
    }
    bool cpu = false, gpu = false;
    for (const auto& embedded : bundle.artifacts) {
        cpu |= embedded.artifact.manifest.backend == artifact::NativeBackend::Cpu;
        gpu |= embedded.artifact.manifest.backend == artifact::NativeBackend::Gpu;
    }
    out << "backend.cpu=" << (cpu ? "available" : "unavailable") << '\n'
        << "backend.gpu=" << (gpu ? "available" : "unavailable") << '\n';
    for (const auto& embedded : bundle.artifacts)
        out << "artifact." << (modelBackend(embedded.artifact.manifest.backend) == ModelBackend::Cpu ? "cpu" : "gpu")
            << ".payload_digest=" << embedded.artifact.manifest.payloadDigest << '\n'
            << "artifact." << (modelBackend(embedded.artifact.manifest.backend) == ModelBackend::Cpu ? "cpu" : "gpu")
            << ".plan_digest=" << embedded.artifact.manifest.planDigest << '\n';
    return out.str();
}

ModelGpuSubmission submitModelGpu(const ModelBundle& bundle,
                                  const std::vector<artifact::ArtifactValue>& publicInputs,
                                  int device) {
    ModelGpuSubmission result;
    auto prepared = prepare(bundle, ModelBackend::Gpu, publicInputs);
    if (prepared.error) { result.error = std::move(prepared.error); return result; }
    auto submitted = artifact::submitArtifactGpu(*prepared.artifact, prepared.inputs, device);
    result.gpuEvidence = submitted.gpuEvidence;
    if (!submitted.ok()) {
        result.error = makeError(ModelErrorCategory::Execution, submitted.error->code,
                                 submitted.error->message);
    } else result.pending = std::move(submitted.pending);
    return result;
}

ModelExecutionResult executeModel(const ModelBundle& bundle, ModelBackend backend,
                                  const std::vector<artifact::ArtifactValue>& publicInputs,
                                  int device) {
    ModelExecutionResult result;
    if (backend == ModelBackend::Gpu) {
        auto submitted = submitModelGpu(bundle, publicInputs, device);
        result.gpuEvidence = submitted.gpuEvidence;
        if (!submitted.ok()) { result.error = std::move(submitted.error); return result; }
        auto observed = submitted.pending->observe();
        result.gpuEvidence = observed.evidence;
        if (!observed.ok()) {
            result.error = makeError(ModelErrorCategory::Execution,
                observed.error ? observed.error->code : "GPU-EXECUTION",
                observed.error ? observed.error->message : "GPU model execution failed");
        } else result.value = std::move(observed.value);
        return result;
    }
    auto prepared = prepare(bundle, ModelBackend::Cpu, publicInputs);
    if (prepared.error) { result.error = std::move(prepared.error); return result; }
    auto executed = artifact::executeArtifact(*prepared.artifact, prepared.inputs, device);
    result.gpuEvidence = executed.gpuEvidence;
    if (!executed.ok())
        result.error = makeError(ModelErrorCategory::Execution, executed.error->code, executed.error->message);
    else result.value = std::move(executed.value);
    return result;
}

} // namespace thiran::v0::model
