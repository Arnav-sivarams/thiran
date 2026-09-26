#include "model/v0/ReferenceModel.hpp"
#include "tooling/v0/BuildConfig.hpp"
#include "tooling/v0/Process.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;
namespace artifact = thiran::v0::artifact;
namespace backend = thiran::v0::backend;
namespace model = thiran::v0::model;
namespace persistence = thiran::v0::persistence;
namespace semantic = thiran::v0::semantic;
namespace storage = thiran::v0::storage;
namespace tooling = thiran::v0::tooling;
namespace training = thiran::v0::training;

namespace {

int checks = 0;
void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void close(float actual, float expected, float tolerance = 1e-6f) {
    require(std::fabs(actual - expected) <= tolerance,
            "numeric mismatch actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}
fs::path temporary() {
    std::string pattern = "/tmp/th017-model-XXXXXX";
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
    return pattern;
}
std::vector<std::byte> readBytes(const fs::path& path) {
    return persistence::readFile(path, 2ULL << 30);
}
void writeBytes(const fs::path& path, const std::vector<std::byte>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("test byte write failed");
}
void putU32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        bytes.at(offset++) = static_cast<std::byte>(static_cast<std::uint8_t>(value >> shift));
}
void putU64(std::vector<std::byte>& bytes, std::size_t offset, std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
        bytes.at(offset++) = static_cast<std::byte>(static_cast<std::uint8_t>(value >> shift));
}
void refreshEnvelopeDigest(std::vector<std::byte>& bytes) {
    if (bytes.size() < 32) throw std::runtime_error("test envelope is too small");
    putU64(bytes, 16, bytes.size() - 32);
    putU64(bytes, 24, persistence::fnv1a64(bytes.data() + 32, bytes.size() - 32));
}
void skipArtifactType(persistence::BinaryReader& reader) {
    (void)reader.u8();
    (void)reader.u8();
    (void)reader.u32();
    const auto extents = reader.u32();
    for (std::uint32_t axis = 0; axis < extents; ++axis)
        if (reader.u8()) (void)reader.u64();
}
std::size_t firstDeclaredArtifactBackendOffset(const std::vector<std::byte>& bytes) {
    if (bytes.size() < 32) throw std::runtime_error("test bundle is too small");
    std::vector<std::byte> payload(bytes.begin() + 32, bytes.end());
    persistence::BinaryReader reader(payload, "TEST-MODEL");
    (void)reader.text();
    (void)reader.text();
    (void)reader.text();
    const auto parameters = reader.u32();
    for (std::uint32_t index = 0; index < parameters; ++index)
        (void)persistence::readStoredParameter(reader);
    const auto publicInputs = reader.u32();
    for (std::uint32_t index = 0; index < publicInputs; ++index) skipArtifactType(reader);
    skipArtifactType(reader);
    const auto publicBindings = reader.u32();
    for (std::uint32_t index = 0; index < publicBindings; ++index) {
        (void)reader.u32(); (void)reader.u32();
    }
    const auto parameterBindings = reader.u32();
    for (std::uint32_t index = 0; index < parameterBindings; ++index) {
        (void)reader.u64(); (void)reader.u32();
    }
    if (reader.u32() == 0) throw std::runtime_error("test bundle has no artifact");
    (void)reader.text();
    return 32 + reader.offset();
}

bool equalValue(const semantic::RuntimeValue& left, const semantic::RuntimeValue& right) {
    if (left.data.index() != right.data.index()) return false;
    if (const auto* value = std::get_if<float>(&left.data))
        return std::bit_cast<std::uint32_t>(*value) ==
               std::bit_cast<std::uint32_t>(std::get<float>(right.data));
    const auto* tensor = std::get_if<semantic::RuntimeTensor>(&left.data);
    if (!tensor) return false;
    const auto& other = std::get<semantic::RuntimeTensor>(right.data);
    if (tensor->dtype != other.dtype || tensor->shape != other.shape ||
        tensor->values != other.values || tensor->f32Values.size() != other.f32Values.size()) return false;
    for (std::size_t index = 0; index < tensor->f32Values.size(); ++index)
        if (std::bit_cast<std::uint32_t>(tensor->f32Values[index]) !=
            std::bit_cast<std::uint32_t>(other.f32Values[index])) return false;
    return true;
}
bool equalState(const training::TrainingState& left, const training::TrainingState& right) {
    if (left.planSignature != right.planSignature || left.step != right.step ||
        left.optimizer.kind != right.optimizer.kind || left.optimizer.step != right.optimizer.step ||
        left.parameters.size() != right.parameters.size() ||
        left.optimizer.velocities.size() != right.optimizer.velocities.size()) return false;
    for (std::size_t index = 0; index < left.parameters.size(); ++index) {
        const auto& a = left.parameters[index]; const auto& b = right.parameters[index];
        if (a.id != b.id || a.type != b.type || a.shape != b.shape || !equalValue(a.value, b.value)) return false;
    }
    for (std::size_t index = 0; index < left.optimizer.velocities.size(); ++index) {
        const auto& a = left.optimizer.velocities[index]; const auto& b = right.optimizer.velocities[index];
        if (a.id != b.id || a.type != b.type || a.shape != b.shape || !equalValue(a.value, b.value)) return false;
    }
    return true;
}
float tensorOnly(const semantic::RuntimeValue& value) {
    return std::get<semantic::RuntimeTensor>(value.data).f32Values.at(0);
}
float artifactOnly(const artifact::ArtifactValue& value) {
    return std::get<storage::Tensor>(value).logicalF32Values().at(0);
}
const training::ParameterValue& parameter(const training::TrainingPlan& plan,
                                          const training::TrainingState& state,
                                          std::size_t sourceIndex) {
    const auto descriptor = std::find_if(plan.parameters.begin(), plan.parameters.end(),
        [&](const auto& candidate) { return candidate.sourceParameterIndex == sourceIndex; });
    if (descriptor == plan.parameters.end()) throw std::runtime_error("parameter descriptor missing");
    const auto value = std::find_if(state.parameters.begin(), state.parameters.end(),
        [&](const auto& candidate) { return candidate.id == descriptor->id; });
    if (value == state.parameters.end()) throw std::runtime_error("parameter state missing");
    return *value;
}
artifact::ArtifactValue artifactValue(const training::ParameterValue& value) {
    const auto& tensor = std::get<semantic::RuntimeTensor>(value.value.data);
    std::vector<std::uint64_t> shape;
    for (auto extent : tensor.shape) shape.push_back(static_cast<std::uint64_t>(extent));
    return storage::Tensor::materializeF32(shape, tensor.f32Values);
}

struct Trace {
    float firstLoss = 0.0f;
    float lastLoss = 0.0f;
    training::TrainingState state;
};
Trace train(const training::TrainingPlan& plan, training::TrainingState state, std::size_t steps) {
    Trace trace;
    for (std::size_t index = 0; index < steps; ++index) {
        auto stepped = training::trainingStep(plan, state, model::referenceAffineBatch());
        if (!stepped.ok()) throw std::runtime_error("reference training step failed");
        if (index == 0) trace.firstLoss = *stepped.loss;
        trace.lastLoss = *stepped.loss;
        state = std::move(*stepped.nextState);
    }
    trace.state = std::move(state);
    return trace;
}
Trace trainFresh(const training::TrainingPlan& plan, std::size_t steps) {
    auto initialized = model::initializeReferenceAffineState(plan);
    require(initialized.ok(), "reference initial state failed");
    return train(plan, std::move(*initialized.state), steps);
}

persistence::StoredValue storedVelocity(const training::VelocityValue& velocity) {
    persistence::StoredValue stored;
    stored.kind = velocity.type.kind == semantic::TypeKind::Tensor ?
        persistence::StoredValueKind::Tensor : persistence::StoredValueKind::Scalar;
    stored.dtype = storage::DType::F32;
    for (auto extent : velocity.shape) stored.shape.push_back(static_cast<std::uint64_t>(extent));
    persistence::BinaryWriter bytes;
    if (const auto* scalar = std::get_if<float>(&velocity.value.data))
        bytes.u32(std::bit_cast<std::uint32_t>(*scalar));
    else for (float value : std::get<semantic::RuntimeTensor>(velocity.value.data).f32Values)
        bytes.u32(std::bit_cast<std::uint32_t>(value));
    stored.bytes = bytes.take();
    return stored;
}

struct RawParameter {
    std::uint64_t sourceIndex = 0;
    std::uint64_t position = 0;
    persistence::StoredParameter value;
};
struct RawCheckpoint {
    std::string modelIdentity;
    std::uint64_t signature = 0;
    std::uint64_t step = 0;
    std::uint64_t optimizerStep = 0;
    training::OptimizerSpec optimizer;
    std::string parameterDigest;
    std::vector<RawParameter> parameters;
    std::vector<std::pair<std::uint64_t,persistence::StoredValue>> velocities;
};
RawCheckpoint rawCheckpoint(const training::TrainingPlan& plan, const training::TrainingState& state) {
    RawCheckpoint raw;
    raw.modelIdentity = std::string(model::referenceAffineModelIdentity);
    raw.signature = plan.signature;
    raw.step = state.step;
    raw.optimizerStep = state.optimizer.step;
    raw.optimizer = plan.optimizer;
    const auto stored = training::storedParameters(plan, state);
    for (std::size_t index = 0; index < stored.size(); ++index)
        raw.parameters.push_back({plan.parameters[index].sourceParameterIndex,
                                  plan.parameters[index].position, stored[index]});
    raw.parameterDigest = persistence::parameterDigest(stored);
    for (const auto& velocity : state.optimizer.velocities)
        raw.velocities.push_back({velocity.id.value, storedVelocity(velocity)});
    return raw;
}
std::vector<std::byte> encodeRaw(RawCheckpoint raw) {
    std::vector<persistence::StoredParameter> digestParameters;
    for (const auto& parameter : raw.parameters) digestParameters.push_back(parameter.value);
    if (raw.parameterDigest.empty()) raw.parameterDigest = persistence::parameterDigest(digestParameters);
    persistence::BinaryWriter payload;
    payload.text(raw.modelIdentity);
    payload.u64(raw.signature);
    payload.u64(raw.step);
    payload.u64(raw.optimizerStep);
    payload.u8(static_cast<std::uint8_t>(raw.optimizer.kind));
    payload.u32(std::bit_cast<std::uint32_t>(raw.optimizer.learningRate));
    payload.u32(std::bit_cast<std::uint32_t>(raw.optimizer.momentum));
    payload.text(raw.parameterDigest);
    payload.u32(static_cast<std::uint32_t>(raw.parameters.size()));
    for (const auto& parameter : raw.parameters) {
        payload.u64(parameter.sourceIndex);
        payload.u64(parameter.position);
        persistence::writeStoredParameter(payload, parameter.value);
    }
    payload.u32(static_cast<std::uint32_t>(raw.velocities.size()));
    for (const auto& velocity : raw.velocities) {
        payload.u64(velocity.first);
        persistence::writeStoredValue(payload, velocity.second);
    }
    persistence::BinaryWriter writer;
    const std::string magic = "THIRANCP";
    writer.raw(reinterpret_cast<const std::byte*>(magic.data()), magic.size());
    writer.u32(training::checkpointFormatVersion);
    writer.u32(training::trainingCheckpointAbiVersion);
    writer.u64(payload.bytes().size());
    writer.u64(persistence::fnv1a64(payload.bytes().data(), payload.bytes().size()));
    writer.raw(payload.bytes().data(), payload.bytes().size());
    return writer.take();
}
void checkpointReject(const fs::path& path, const training::TrainingPlan& plan,
                      const training::TrainingCheckpointSchema& schema,
                      std::vector<std::byte> bytes, const std::string& code) {
    writeBytes(path, bytes);
    const auto loaded = training::loadTrainingCheckpoint(path, schema, plan);
    require(!loaded.ok() && loaded.error && loaded.error->code == code,
            code + " checkpoint case was not rejected; actual=" +
            (loaded.error ? loaded.error->code + " " + loaded.error->message : "success"));
}

artifact::NativeToolchain toolchain() {
    const fs::path build = TH017_BUILD;
    return {TH017_CXX, tooling::configuredHostCompilerArguments(), {TH017_INCLUDE},
            {build / "libthiran_v0_storage.a", build / "libthiran_v0_async.a",
             build / "libthiran_v0_analysis.a", build / "libthiran_v0_semantic.a",
             build / "libthiran_v0_frontend.a"}};
}

struct Lifecycle {
    semantic::Module module;
    training::TrainingPlan plan;
    training::TrainingCheckpointSchema checkpointSchema;
    training::CheckpointLoadResult restored;
    model::DeploymentSnapshot snapshot;
    backend::TensorRegion region;
    artifact::NativeArtifact cpuArtifact;
    model::ModelBundle bundle;
    float initialLoss = 0.0f;
    float finalLoss = 0.0f;
    float weight = 0.0f;
    float bias = 0.0f;
};

Lifecycle buildLifecycle(const fs::path& root) {
    Lifecycle life;
    auto reference = model::createReferenceAffinePlan();
    require(reference.ok(), "TH-011 reference model plan failed: " + reference.error);
    life.module = std::move(*reference.module);
    life.plan = std::move(*reference.plan);
    require(life.plan.parameters.size() == 2 &&
            life.plan.parameters[0].sourceParameterIndex == 3 &&
            life.plan.parameters[1].sourceParameterIndex == 2,
            "reference TrainingPlan lost [bias,weight] order");

    auto initialized = model::initializeReferenceAffineState(life.plan);
    require(initialized.ok(), "reference initial state failed");
    require(tensorOnly(parameter(life.plan, *initialized.state, 2).value) == 0.0f &&
            tensorOnly(parameter(life.plan, *initialized.state, 3).value) == 0.0f &&
            initialized.state->step == 0,
            "reference training did not begin from the accepted zero parameter state");
    auto schema = training::createTrainingCheckpointSchema(
        std::string(model::referenceAffineModelIdentity), life.plan, *initialized.state);
    require(schema.ok(), schema.error ? schema.error->message : "reference checkpoint schema failed");
    life.checkpointSchema = std::move(*schema.schema);
    auto canonical = train(life.plan, std::move(*initialized.state), 200);
    auto trained = std::optional<training::TrainingState>{std::move(canonical.state)};
    auto trace = trainFresh(life.plan, 200);
    life.initialLoss = canonical.firstLoss;
    life.finalLoss = canonical.lastLoss;
    // Retain only the canonical state and scalar evidence; the second trace is
    // independent determinism evidence, not the state used for deployment.
    require(equalState(*trained, trace.state), "reference training was not bit deterministic");
    life.weight = tensorOnly(parameter(life.plan, *trained, 2).value);
    life.bias = tensorOnly(parameter(life.plan, *trained, 3).value);
    require(life.initialLoss == 84.0f && life.finalLoss < 1e-6f &&
            std::fabs(life.weight - 2.0f) < 0.001f && std::fabs(life.bias - 1.0f) < 0.001f,
            "learned state failed accepted TH-011 quality bounds");

    const auto beforeParameters = training::storedParameters(life.plan, *trained);
    const auto beforeDigest = persistence::parameterDigest(beforeParameters);
    auto beforeEvaluation = model::evaluateReferenceAffine(life.module, life.plan, *trained, 2.0f);
    require(beforeEvaluation.ok, "pre-save reference evaluation failed");
    const auto checkpoint = root / "trained.thc";
    auto saved = training::saveTrainingCheckpoint(
        checkpoint, life.checkpointSchema, life.plan, *trained);
    require(saved.ok() && saved.metadata->parameterDigest == beforeDigest && saved.metadata->step == 200,
            "actual learned TrainingState was not checkpointed");
    trained.reset();
    require(!trained.has_value(), "original trained state was not destroyed before reload");

    life.restored = training::loadTrainingCheckpoint(checkpoint, life.checkpointSchema, life.plan);
    require(life.restored.ok() && life.restored.metadata->parameterDigest == beforeDigest &&
            life.restored.state->step == 200 && life.restored.state->optimizer.step == 200,
            "checkpoint reload did not restore learned/training metadata");
    require(training::storedParameters(life.plan, *life.restored.state) == beforeParameters,
            "restored learned parameters are not representation-exact");
    auto afterEvaluation = model::evaluateReferenceAffine(
        life.module, life.plan, *life.restored.state, 2.0f);
    require(afterEvaluation.ok && equalValue(*beforeEvaluation.value, *afterEvaluation.value),
            "restored reference evaluation differs from pre-save evaluation");

    auto snapshot = model::createDeploymentSnapshot(
        life.plan, life.restored, model::referenceAffineSnapshotRequest());
    require(snapshot.ok() && snapshot.snapshot->parameterDigest == beforeDigest &&
            snapshot.snapshot->parameters.size() == 2,
            "deployment snapshot was not derived from restored checkpoint");
    life.snapshot = std::move(*snapshot.snapshot);

    auto cpuRegion = model::referenceAffineInferenceRegion(life.module, backend::NativeTarget::Cpu);
    auto gpuRegion = model::referenceAffineInferenceRegion(life.module, backend::NativeTarget::Gpu);
    require(cpuRegion.ok() && gpuRegion.ok() && cpuRegion.region->dump() == gpuRegion.region->dump(),
            "CPU/GPU inference regions do not share model semantics");
    life.region = std::move(*cpuRegion.region);
    auto hostPlan = backend::buildPhysicalPlan(life.region, backend::PhysicalDevice::Host, {true, true});
    require(hostPlan.ok() && std::any_of(hostPlan.plan->fusionGroups.begin(), hostPlan.plan->fusionGroups.end(),
            [](const auto& group) { return group.fused && group.nodes.size() == 2; }),
            "CPU reference deployment did not retain eligible TH-015 fusion");
    const auto sample = model::referenceAffinePublicInput(2.0f);
    const auto weight = artifactValue(parameter(life.plan, *life.restored.state, 2));
    const auto bias = artifactValue(parameter(life.plan, *life.restored.state, 3));
    const auto entry = artifact::specializeEntry(life.region, {sample, weight, bias});
    auto built = artifact::buildCpuAot(life.region, toolchain(),
        {root / "affine-cpu.tha", {true, true}, entry});
    require(built.success && built.manifest->planning.enableFusion && built.manifest->planning.enableReuse,
            built.error ? built.error->message : "CPU model artifact build failed");
    auto loadedArtifact = artifact::loadArtifact(root / "affine-cpu.tha");
    require(loadedArtifact.ok(), loadedArtifact.error ? loadedArtifact.error->message : "CPU model artifact load failed");
    life.cpuArtifact = std::move(*loadedArtifact.artifact);
    auto bundled = model::createModelBundle(life.snapshot,
        {{std::string(model::referenceAffineModelIdentity), life.cpuArtifact}});
    require(bundled.ok(), bundled.error ? bundled.error->message : "CPU model bundle creation failed");
    life.bundle = std::move(*bundled.bundle);
    return life;
}

void checkpointMatrix(const fs::path& root, const Lifecycle& life) {
    const auto checkpoint = root / "trained.thc";
    const auto validBytes = readBytes(checkpoint);
    auto changed = validBytes;
    changed[0] ^= std::byte{1}; checkpointReject(root / "bad-magic.thc", life.plan, life.checkpointSchema, changed, "CKPT-MAGIC");
    changed = validBytes; putU32(changed, 8, 99); checkpointReject(root / "bad-version.thc", life.plan, life.checkpointSchema, changed, "CKPT-FORMAT-VERSION");
    changed = validBytes; putU32(changed, 12, 99); checkpointReject(root / "bad-abi.thc", life.plan, life.checkpointSchema, changed, "CKPT-TRAINING-ABI");
    changed = validBytes; changed.pop_back(); checkpointReject(root / "truncated.thc", life.plan, life.checkpointSchema, changed, "CKPT-TRUNCATED");
    changed = validBytes; changed.push_back(std::byte{0}); checkpointReject(root / "trailing.thc", life.plan, life.checkpointSchema, changed, "CKPT-TRAILING-DATA");
    changed = validBytes; putU64(changed, 16, std::numeric_limits<std::uint64_t>::max());
    checkpointReject(root / "bad-length.thc", life.plan, life.checkpointSchema, changed, "CKPT-LENGTH");
    changed = validBytes; changed.back() ^= std::byte{1};
    checkpointReject(root / "bad-digest.thc", life.plan, life.checkpointSchema, changed, "CKPT-PAYLOAD-INTEGRITY");

    auto raw = rawCheckpoint(life.plan, *life.restored.state);
    raw.parameters[1].value.id = raw.parameters[0].value.id; raw.parameterDigest.clear();
    checkpointReject(root / "duplicate.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-DUPLICATE-PARAMETER");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.parameters.pop_back(); raw.parameterDigest.clear();
    checkpointReject(root / "missing.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-MISSING-PARAMETER");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.parameters.push_back(raw.parameters.back());
    raw.parameters.back().value.id = 999; raw.parameters.back().value.name = "unexpected"; raw.parameterDigest.clear();
    checkpointReject(root / "unexpected.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-UNEXPECTED-PARAMETER");
    raw = rawCheckpoint(life.plan, *life.restored.state); std::swap(raw.parameters[0], raw.parameters[1]); raw.parameterDigest.clear();
    checkpointReject(root / "order.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-PARAMETER-ORDER");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.parameters[0].value.value.dtype = storage::DType::I64;
    raw.parameters[0].value.value.bytes.resize(8); raw.parameterDigest.clear();
    checkpointReject(root / "dtype.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-DTYPE");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.parameters[0].value.value.shape = {1,1}; raw.parameterDigest.clear();
    checkpointReject(root / "rank.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-RANK");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.parameters[0].value.value.shape = {2};
    raw.parameters[0].value.value.bytes.resize(8); raw.parameterDigest.clear();
    checkpointReject(root / "shape.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-SHAPE");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.optimizer.kind = training::OptimizerKind::SGDMomentum;
    checkpointReject(root / "optimizer.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-OPTIMIZER-MISMATCH");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.step = std::numeric_limits<std::uint64_t>::max();
    raw.optimizerStep = raw.step;
    checkpointReject(root / "step.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-STEP");
    raw = rawCheckpoint(life.plan, *life.restored.state); ++raw.signature;
    checkpointReject(root / "signature.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-TRAINING-ABI");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.parameterDigest = "fnv1a64:0000000000000000";
    checkpointReject(root / "parameter-digest.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-PARAMETER-DIGEST");
    raw = rawCheckpoint(life.plan, *life.restored.state); raw.parameters[0].value.value.shape = {2}; raw.parameterDigest.clear();
    checkpointReject(root / "numeric.thc", life.plan, life.checkpointSchema, encodeRaw(raw), "CKPT-NUMERIC-PAYLOAD");
    changed = validBytes; for (std::size_t index = 32; index < 40; ++index) changed[index] = std::byte{0xff};
    refreshEnvelopeDigest(changed);
    checkpointReject(root / "excessive.thc", life.plan, life.checkpointSchema, changed, "CKPT-LENGTH");

    auto wrongSchema = life.checkpointSchema; wrongSchema.modelIdentity = "wrong.model";
    const auto wrongIdentity = training::loadTrainingCheckpoint(checkpoint, wrongSchema, life.plan);
    require(!wrongIdentity.ok() && wrongIdentity.error->code == "CKPT-MODEL-IDENTITY",
            "wrong checkpoint model identity accepted");

    const auto prior = readBytes(checkpoint);
    writeBytes(fs::path(checkpoint.string() + ".tmp"), {std::byte{1}});
    auto failedSave = training::saveTrainingCheckpoint(
        checkpoint, life.checkpointSchema, life.plan, *life.restored.state);
    require(!failedSave.ok() && failedSave.error->code == "CKPT-WRITE" && readBytes(checkpoint) == prior,
            "transactional checkpoint failure replaced the prior valid checkpoint");
    fs::remove(fs::path(checkpoint.string() + ".tmp"));
    for (const char* failpoint : {"before_temp_creation", "after_temp_creation",
                                  "after_partial_write", "after_full_write", "after_file_fsync"}) {
        ::setenv("THIRAN_TEST_PERSISTENCE_WRITE_FAILPOINT", failpoint, 1);
        const auto injected = training::saveTrainingCheckpoint(
            checkpoint, life.checkpointSchema, life.plan, *life.restored.state);
        ::unsetenv("THIRAN_TEST_PERSISTENCE_WRITE_FAILPOINT");
        require(!injected.ok() && injected.error->code == "CKPT-WRITE" &&
                readBytes(checkpoint) == prior && !fs::exists(fs::path(checkpoint.string() + ".tmp")),
                std::string("checkpoint failpoint did not preserve old bytes/temp hygiene: ") + failpoint);
    }
    ::setenv("THIRAN_TEST_PERSISTENCE_WRITE_FAILPOINT", "after_rename", 1);
    const auto postRename = training::saveTrainingCheckpoint(
        checkpoint, life.checkpointSchema, life.plan, *life.restored.state);
    ::unsetenv("THIRAN_TEST_PERSISTENCE_WRITE_FAILPOINT");
    require(!postRename.ok() && postRename.error->code == "CKPT-WRITE" &&
            training::loadTrainingCheckpoint(checkpoint, life.checkpointSchema, life.plan).ok(),
            "checkpoint post-rename failpoint left an invalid final file or claimed success");
}

void resumeMatrix(const fs::path& root) {
    auto reference = model::createReferenceAffinePlan();
    require(reference.ok(), "resume SGD plan failed");
    auto referenceInitial = model::initializeReferenceAffineState(*reference.plan);
    auto referenceSchema = training::createTrainingCheckpointSchema(
        std::string(model::referenceAffineModelIdentity), *reference.plan, *referenceInitial.state);
    require(referenceSchema.ok(), "resume SGD schema failed");
    auto partial = trainFresh(*reference.plan, 80);
    auto save = training::saveTrainingCheckpoint(root / "resume-sgd.thc",
        *referenceSchema.schema, *reference.plan, partial.state);
    auto reload = training::loadTrainingCheckpoint(root / "resume-sgd.thc",
        *referenceSchema.schema, *reference.plan);
    require(save.ok() && reload.ok(), "SGD resume checkpoint roundtrip failed");
    auto resumed = train(*reference.plan, std::move(*reload.state), 120);
    auto continuous = trainFresh(*reference.plan, 200);
    require(equalState(resumed.state, continuous.state), "80+reload+120 SGD differs from continuous 200");

    auto momentum = model::createReferenceAffinePlan(
        {training::OptimizerKind::SGDMomentum, 0.01f, 0.5f});
    require(momentum.ok(), "momentum resume plan failed");
    auto momentumInitial = model::initializeReferenceAffineState(*momentum.plan);
    auto momentumSchema = training::createTrainingCheckpointSchema(
        std::string(model::referenceAffineModelIdentity), *momentum.plan, *momentumInitial.state);
    require(momentumSchema.ok(), "momentum resume schema failed");
    auto first = trainFresh(*momentum.plan, 10);
    auto momentumSave = training::saveTrainingCheckpoint(root / "resume-momentum.thc",
        *momentumSchema.schema, *momentum.plan, first.state);
    auto momentumLoad = training::loadTrainingCheckpoint(root / "resume-momentum.thc",
        *momentumSchema.schema, *momentum.plan);
    require(momentumSave.ok() && momentumLoad.ok() &&
            momentumLoad.state->optimizer.velocities.size() == 2,
            "momentum optimizer state did not reload");
    auto resumedMomentum = train(*momentum.plan, std::move(*momentumLoad.state), 10);
    auto continuousMomentum = trainFresh(*momentum.plan, 20);
    require(equalState(resumedMomentum.state, continuousMomentum.state),
            "10+reload+10 momentum differs from continuous 20");
    auto raw = rawCheckpoint(*momentum.plan, continuousMomentum.state);
    raw.velocities[0].second.shape = {2}; raw.velocities[0].second.bytes.resize(8);
    checkpointReject(root / "velocity-shape.thc", *momentum.plan, *momentumSchema.schema, encodeRaw(raw),
                     "CKPT-OPTIMIZER-STATE-SHAPE");
    raw = rawCheckpoint(*momentum.plan, continuousMomentum.state);
    raw.velocities[0].second.dtype = storage::DType::I64; raw.velocities[0].second.bytes.resize(8);
    checkpointReject(root / "velocity-dtype.thc", *momentum.plan, *momentumSchema.schema, encodeRaw(raw), "CKPT-DTYPE");
}

void bundleMatrix(const fs::path& root, const Lifecycle& life) {
    const auto bundlePath = root / "affine.thm";
    auto written = model::writeModelBundle(life.bundle, bundlePath);
    require(written.ok(), written.error ? written.error->message : "bundle write failed");
    auto loaded = model::loadModelBundle(bundlePath, std::string(model::referenceAffineModelIdentity));
    require(loaded.ok() && loaded.bundle->parameterDigest == life.restored.metadata->parameterDigest,
            "bundle/checkpoint parameter identity mismatch");
    require(model::inspectModelBundle(*loaded.bundle).find("backend.cpu=available") != std::string::npos &&
            model::inspectModelBundle(*loaded.bundle).find("backend.gpu=unavailable") != std::string::npos,
            "bundle inspection omitted backend availability");
    auto primaryInspect = tooling::runProcess({TH022_CLI,
        {"model", "inspect", bundlePath.string()}});
    auto primaryRun = tooling::runProcess({TH022_CLI,
        {"model", "run", bundlePath.string(), "--backend", "cpu", "--input", "2", "--verbose"}});
    auto primaryMissing = tooling::runProcess({TH022_CLI,
        {"model", "inspect", (root / "missing-model.thm").string()}});
    auto primaryUnavailable = tooling::runProcess({TH022_CLI,
        {"model", "run", bundlePath.string(), "--backend", "gpu", "--input", "2"}});
    require(primaryInspect.exitStatus == 0 &&
            primaryInspect.standardOutput.find("backend.cpu=available") != std::string::npos &&
            primaryRun.exitStatus == 0 &&
            primaryRun.standardError.find("compiler_invocations=0") != std::string::npos &&
            primaryRun.standardError.find("fallback=NONE") != std::string::npos &&
            primaryMissing.exitStatus != 0 &&
            primaryMissing.standardError.find("cannot load model") != std::string::npos &&
            primaryUnavailable.exitStatus != 0 &&
            primaryUnavailable.standardError.find("MODEL-BACKEND-UNAVAILABLE") != std::string::npos,
            "primary CLI model workflow failed");

    for (float input : {0.0f, 1.0f, 2.0f, 5.0f}) {
        auto reference = model::evaluateReferenceAffine(life.module, life.plan, *life.restored.state, input);
        auto cpu = model::executeModel(*loaded.bundle, model::ModelBackend::Cpu,
                                       {model::referenceAffinePublicInput(input)});
        require(reference.ok && cpu.ok(), cpu.error ? cpu.error->message : "CPU deployed inference failed");
        const auto expected = std::get<semantic::RuntimeTensor>(reference.value->data).f32Values.at(0);
        close(artifactOnly(*cpu.value), expected);
        close(expected, input * life.weight + life.bias);
    }
    auto wrongType = model::executeModel(*loaded.bundle, model::ModelBackend::Cpu, {std::int64_t{3}});
    auto wrongShape = model::executeModel(*loaded.bundle, model::ModelBackend::Cpu,
        {storage::Tensor::materializeF32({2}, {1,2})});
    auto wrongCount = model::executeModel(*loaded.bundle, model::ModelBackend::Cpu, {});
    require(!wrongType.ok() && wrongType.error->code == "MODEL-PUBLIC-INPUT-ABI" &&
            !wrongShape.ok() && wrongShape.error->code == "MODEL-PUBLIC-INPUT-ABI" &&
            !wrongCount.ok() && wrongCount.error->code == "MODEL-PUBLIC-INPUT-COUNT",
            "public model ABI failure semantics are incomplete");
    auto gpuRequest = model::executeModel(*loaded.bundle, model::ModelBackend::Gpu,
        {model::referenceAffinePublicInput(2.0f)});
    require(!gpuRequest.ok() && gpuRequest.error->code == "MODEL-BACKEND-UNAVAILABLE",
            "CPU-only bundle silently fell back for explicit GPU request");

    auto malformed = life.bundle;
    malformed.parameterDigest = "fnv1a64:0000000000000000";
    require(model::validateModelBundle(malformed)->code == "MODEL-PARAMETER-DIGEST",
            "bundle parameter digest corruption accepted");
    malformed = life.bundle; malformed.parameters.push_back(malformed.parameters.front());
    require(model::validateModelBundle(malformed)->code == "MODEL-DUPLICATE-PARAMETER",
            "duplicate bundle parameter accepted");
    malformed = life.bundle; malformed.parameters.pop_back();
    malformed.parameterDigest = persistence::parameterDigest(malformed.parameters);
    require(model::validateModelBundle(malformed)->code == "MODEL-UNEXPECTED-PARAMETER",
            "missing bundle parameter accepted");
    malformed = life.bundle; malformed.parameterBindings.pop_back();
    require(model::validateModelBundle(malformed)->code == "MODEL-MISSING-PARAMETER",
            "missing parameter binding accepted");
    malformed = life.bundle; malformed.parameterBindings[1] = malformed.parameterBindings[0];
    require(model::validateModelBundle(malformed)->code == "MODEL-DUPLICATE-BINDING",
            "duplicate parameter binding accepted");
    malformed = life.bundle; malformed.parameters[0].value.shape = {2}; malformed.parameters[0].value.bytes.resize(8);
    malformed.parameterDigest = persistence::parameterDigest(malformed.parameters);
    require(model::validateModelBundle(malformed)->code == "MODEL-PARAMETER-ABI",
            "wrong frozen parameter shape accepted");
    malformed = life.bundle; malformed.parameters[0].value.dtype = storage::DType::I64;
    malformed.parameters[0].value.bytes.resize(8);
    malformed.parameterDigest = persistence::parameterDigest(malformed.parameters);
    require(model::validateModelBundle(malformed)->code == "MODEL-PARAMETER-ABI",
            "wrong frozen parameter dtype accepted");
    malformed = life.bundle; malformed.publicInputs[0].dtype = storage::DType::I64;
    require(model::validateModelBundle(malformed)->code == "MODEL-PUBLIC-ABI",
            "wrong public input ABI accepted");
    malformed = life.bundle; malformed.publicInputs[0].rank = 3;
    malformed.publicInputs[0].extents = {1,1,1};
    require(model::validateModelBundle(malformed)->code == "MODEL-PUBLIC-ABI",
            "impossible public input rank accepted");
    malformed = life.bundle; malformed.artifacts[0].modelIdentity = "wrong.model";
    require(model::validateModelBundle(malformed)->code == "MODEL-ARTIFACT-IDENTITY",
            "inconsistent artifact model identity accepted");

    const auto valid = readBytes(bundlePath);
    auto bytes = valid; bytes[0] ^= std::byte{1}; writeBytes(root / "bundle-magic.thm", bytes);
    require(model::loadModelBundle(root / "bundle-magic.thm").error->code == "MODEL-MAGIC", "bad bundle magic accepted");
    auto primaryMalformed = tooling::runProcess({TH022_CLI,
        {"model", "inspect", (root / "bundle-magic.thm").string()}});
    require(primaryMalformed.exitStatus != 0 &&
            primaryMalformed.standardError.find("MODEL-MAGIC") != std::string::npos,
            "primary CLI accepted malformed model bundle");
    bytes = valid; putU32(bytes, 8, 99); writeBytes(root / "bundle-version.thm", bytes);
    require(model::loadModelBundle(root / "bundle-version.thm").error->code == "MODEL-FORMAT-VERSION", "bad bundle version accepted");
    bytes = valid; putU32(bytes, 12, 99); writeBytes(root / "bundle-abi.thm", bytes);
    require(model::loadModelBundle(root / "bundle-abi.thm").error->code == "MODEL-RUNTIME-ABI", "bad bundle ABI accepted");
    bytes = valid; bytes.pop_back(); writeBytes(root / "bundle-truncated.thm", bytes);
    require(model::loadModelBundle(root / "bundle-truncated.thm").error->code == "MODEL-TRUNCATED", "truncated bundle accepted");
    bytes = valid; bytes.push_back(std::byte{0}); writeBytes(root / "bundle-trailing.thm", bytes);
    require(model::loadModelBundle(root / "bundle-trailing.thm").error->code == "MODEL-TRAILING-DATA", "bundle trailing junk accepted");
    bytes = valid; bytes.back() ^= std::byte{1}; writeBytes(root / "bundle-digest.thm", bytes);
    require(model::loadModelBundle(root / "bundle-digest.thm").error->code == "MODEL-BUNDLE-INTEGRITY", "bundle corruption accepted");
    bytes = valid; bytes.back() ^= std::byte{1}; refreshEnvelopeDigest(bytes); writeBytes(root / "embedded-corrupt.thm", bytes);
    require(model::loadModelBundle(root / "embedded-corrupt.thm").error->code == "MODEL-EMBEDDED-ARTIFACT",
            "embedded artifact corruption accepted");
    bytes = valid; bytes.at(firstDeclaredArtifactBackendOffset(bytes)) =
        static_cast<std::byte>(artifact::NativeBackend::Gpu); refreshEnvelopeDigest(bytes);
    writeBytes(root / "embedded-backend.thm", bytes);
    require(model::loadModelBundle(root / "embedded-backend.thm").error->code == "MODEL-ARTIFACT-BACKEND",
            "embedded artifact backend mismatch accepted");
    bytes = valid; putU64(bytes, 16, std::numeric_limits<std::uint64_t>::max());
    writeBytes(root / "bundle-length.thm", bytes);
    require(model::loadModelBundle(root / "bundle-length.thm").error->code == "MODEL-LENGTH",
            "excessive bundle payload length accepted");
    require(model::loadModelBundle(bundlePath, "wrong.model").error->code == "MODEL-IDENTITY",
            "wrong expected model identity accepted");

    const auto prior = readBytes(bundlePath);
    for (const char* failpoint : {"before_temp_creation", "after_temp_creation",
                                  "after_partial_write", "after_full_write", "after_file_fsync"}) {
        ::setenv("THIRAN_TEST_PERSISTENCE_WRITE_FAILPOINT", failpoint, 1);
        const auto injected = model::writeModelBundle(life.bundle, bundlePath);
        ::unsetenv("THIRAN_TEST_PERSISTENCE_WRITE_FAILPOINT");
        require(!injected.ok() && injected.error->code == "MODEL-WRITE" &&
                readBytes(bundlePath) == prior && !fs::exists(fs::path(bundlePath.string() + ".tmp")),
                std::string("model failpoint did not preserve old bytes/temp hygiene: ") + failpoint);
    }
    ::setenv("THIRAN_TEST_PERSISTENCE_WRITE_FAILPOINT", "after_rename", 1);
    const auto postRename = model::writeModelBundle(life.bundle, bundlePath);
    ::unsetenv("THIRAN_TEST_PERSISTENCE_WRITE_FAILPOINT");
    require(!postRename.ok() && postRename.error->code == "MODEL-WRITE" &&
            model::loadModelBundle(bundlePath).ok(),
            "model post-rename failpoint left an invalid final file or claimed success");
}

void freshProcessAndAudit(const fs::path& root, const Lifecycle& life) {
    const auto bundlePath = root / "affine.thm";
    const auto fresh = root / "fresh deployment";
    fs::create_directory(fresh);
    fs::copy_file(bundlePath, fresh / "model.thm");
    fs::copy_file(TH017_RUNTIME, fresh / "thiran-model");
    fs::permissions(fresh / "thiran-model", fs::perms::owner_read | fs::perms::owner_write |
                    fs::perms::owner_exec, fs::perm_options::add);
    require(!fs::exists(fresh / "trained.thc") && !fs::exists(fresh / "model.th"),
            "fresh deployment directory contains checkpoint/source");
    const auto prior = fs::current_path();
    const std::string oldPath = std::getenv("PATH") ? std::getenv("PATH") : "";
    const std::string oldCxx = std::getenv("CXX") ? std::getenv("CXX") : "";
    const bool hadCxx = std::getenv("CXX") != nullptr;
    fs::current_path(fresh);
    ::setenv("PATH", "/nonexistent", 1);
    ::setenv("CXX", "/nonexistent", 1);
    auto inspected = tooling::runProcess({(fresh / "thiran-model").string(), {"inspect", "model.thm"}});
    auto run = tooling::runProcess({(fresh / "thiran-model").string(),
                                    {"run", "model.thm", "--backend", "cpu", "--input", "5"}});
    ::setenv("PATH", oldPath.c_str(), 1);
    if (hadCxx) ::setenv("CXX", oldCxx.c_str(), 1); else ::unsetenv("CXX");
    fs::current_path(prior);
    require(inspected.exitStatus == 0 && inspected.standardOutput.find(life.bundle.parameterDigest) != std::string::npos,
            "relocated fresh-process bundle inspection failed");
    require(run.exitStatus == 0 && run.standardError.find("compiler_invocations=0") != std::string::npos &&
            run.standardError.find("fallback=NONE") != std::string::npos,
            "relocated compiler-absent CPU model execution failed: " + run.standardError);
    const auto expected = 5.0f * life.weight + life.bias;
    require(run.standardOutput.find(std::to_string(expected).substr(0,4)) != std::string::npos,
            "fresh CPU prediction does not contain learned output");

    auto ldd = tooling::runProcess({"ldd", {(fresh / "thiran-model").string()}});
    auto readelf = tooling::runProcess({"readelf", {"-d", (fresh / "thiran-model").string()}});
    auto symbols = tooling::runProcess({"nm", {"-C", (fresh / "thiran-model").string()}});
    auto strings = tooling::runProcess({"strings", {(fresh / "thiran-model").string()}});
    require(ldd.exitStatus == 0 && readelf.exitStatus == 0 && symbols.exitStatus == 0 && strings.exitStatus == 0,
            "deployed model runtime dependency audit command failed");
    const auto audit = ldd.standardOutput + readelf.standardOutput + symbols.standardOutput + strings.standardOutput;
    for (const char* forbidden : {"libpython", "libtorch", "PyTorch", "TensorFlow", "JAX", "Triton", "CuPy", "evaluateCall"})
        require(audit.find(forbidden) == std::string::npos,
                std::string("deployed runtime contains forbidden dependency/symbol: ") + forbidden);
}

} // namespace

int main() {
    const auto root = temporary();
    try {
        auto life = buildLifecycle(root);
        checkpointMatrix(root, life);
        resumeMatrix(root);
        bundleMatrix(root, life);
        freshProcessAndAudit(root, life);
        std::cout << std::setprecision(9)
                  << "V0ModelDeploymentTests PASS " << checks << " checks\n"
                  << "initial_loss=" << life.initialLoss << '\n'
                  << "final_loss=" << life.finalLoss << '\n'
                  << "weight=" << life.weight << '\n'
                  << "bias=" << life.bias << '\n'
                  << "checkpoint_step=" << life.restored.state->step << '\n'
                  << "parameter_digest=" << life.snapshot.parameterDigest << '\n'
                  << "cpu_plan_digest=" << life.cpuArtifact.manifest.planDigest << '\n'
                  << "fallback=NONE\n";
        fs::remove_all(root);
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << "V0ModelDeploymentTests FAIL: " << failure.what() << '\n';
        std::error_code ignored;
        fs::remove_all(root, ignored);
        return 1;
    }
}
