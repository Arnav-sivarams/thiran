#pragma once

#include "artifact/v0/NativeArtifacts.hpp"
#include "persistence/v0/Persistence.hpp"

#include <filesystem>
#include <optional>

namespace thiran::v0::model {

inline constexpr std::uint32_t modelBundleFormatVersion = 0;
inline constexpr std::uint32_t modelRuntimeAbiVersion = 1;

enum class ModelBackend : std::uint8_t { Cpu = 1, Gpu = 2 };
enum class ModelErrorCategory { Build, Load, Execution };

struct ModelError {
    ModelErrorCategory category = ModelErrorCategory::Load;
    std::string code;
    std::string message;
};

struct PublicInputBinding {
    std::uint32_t publicInputIndex = 0;
    std::uint32_t artifactInputIndex = 0;
    bool operator==(const PublicInputBinding&) const = default;
};

struct ParameterBinding {
    std::uint64_t parameterId = 0;
    std::uint32_t artifactInputIndex = 0;
    bool operator==(const ParameterBinding&) const = default;
};

// Deployment state is immutable by contract after successful construction. It
// contains no optimizer, gradient, backward-save, TrainingPlan, or step state.
struct DeploymentSnapshot {
    std::string modelIdentity;
    std::string inferenceEntry;
    std::vector<persistence::StoredParameter> parameters;
    std::vector<artifact::ArtifactType> publicInputs;
    artifact::ArtifactType result;
    std::vector<PublicInputBinding> publicBindings;
    std::vector<ParameterBinding> parameterBindings;
    std::string parameterDigest;
};

struct EmbeddedArtifact {
    std::string modelIdentity;
    artifact::NativeArtifact artifact;
};

struct ModelBundle {
    std::uint32_t formatVersion = modelBundleFormatVersion;
    std::uint32_t runtimeAbiVersion = modelRuntimeAbiVersion;
    std::string modelIdentity;
    std::string inferenceEntry;
    std::vector<persistence::StoredParameter> parameters;
    std::vector<artifact::ArtifactType> publicInputs;
    artifact::ArtifactType result;
    std::vector<PublicInputBinding> publicBindings;
    std::vector<ParameterBinding> parameterBindings;
    std::string parameterDigest;
    std::vector<EmbeddedArtifact> artifacts;
};

struct ModelBundleResult {
    std::optional<ModelBundle> bundle;
    std::optional<ModelError> error;
    bool ok() const noexcept { return bundle.has_value() && !error.has_value(); }
};

struct ModelWriteResult {
    bool success = false;
    std::optional<ModelError> error;
    bool ok() const noexcept { return success && !error.has_value(); }
};

struct ModelExecutionResult {
    std::optional<artifact::ArtifactValue> value;
    std::optional<ModelError> error;
    std::optional<backend::GpuExecutionEvidence> gpuEvidence;
    bool ok() const noexcept { return value.has_value() && !error.has_value(); }
};

struct ModelGpuSubmission {
    std::optional<backend::PendingGpuExecution> pending;
    std::optional<ModelError> error;
    std::optional<backend::GpuExecutionEvidence> gpuEvidence;
    bool ok() const noexcept { return pending.has_value() && !error.has_value(); }
};

std::optional<ModelError> validateModelBundle(const ModelBundle&);
ModelBundleResult createModelBundle(const DeploymentSnapshot&,
                                    std::vector<EmbeddedArtifact>);
ModelWriteResult writeModelBundle(const ModelBundle&, const std::filesystem::path&);
ModelBundleResult loadModelBundle(const std::filesystem::path&,
                                  std::optional<std::string> expectedModelIdentity = {});
std::string inspectModelBundle(const ModelBundle&);
ModelExecutionResult executeModel(const ModelBundle&, ModelBackend,
                                  const std::vector<artifact::ArtifactValue>& publicInputs,
                                  int device = 0);
ModelGpuSubmission submitModelGpu(const ModelBundle&,
                                  const std::vector<artifact::ArtifactValue>& publicInputs,
                                  int device = 0);

} // namespace thiran::v0::model
