#pragma once

#include "model/v0/ModelBundle.hpp"
#include "training/v0/Checkpoint.hpp"

namespace thiran::v0::model {

struct SnapshotParameterBindingRequest {
    std::size_t sourceParameterIndex = 0;
    std::uint32_t artifactInputIndex = 0;
};

struct DeploymentSnapshotRequest {
    std::string modelIdentity;
    std::string inferenceEntry;
    std::vector<artifact::ArtifactType> publicInputs;
    artifact::ArtifactType result;
    std::vector<PublicInputBinding> publicBindings;
    std::vector<SnapshotParameterBindingRequest> parameterBindings;
};

struct DeploymentSnapshotResult {
    std::optional<DeploymentSnapshot> snapshot;
    std::optional<ModelError> error;
    bool ok() const noexcept { return snapshot.has_value() && !error.has_value(); }
};

// The only public snapshot constructor consumes a successful checkpoint-load
// result, making the validated reload boundary explicit in the API.
DeploymentSnapshotResult createDeploymentSnapshot(
    const training::TrainingPlan&,
    const training::CheckpointLoadResult& restoredCheckpoint,
    const DeploymentSnapshotRequest&);

} // namespace thiran::v0::model
