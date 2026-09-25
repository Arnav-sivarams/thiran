#include "model/v0/DeploymentBuild.hpp"

#include <algorithm>
#include <set>

namespace thiran::v0::model {

DeploymentSnapshotResult createDeploymentSnapshot(
    const training::TrainingPlan& plan,
    const training::CheckpointLoadResult& restoredCheckpoint,
    const DeploymentSnapshotRequest& request) {
    DeploymentSnapshotResult result;
    auto fail = [&](std::string code, std::string message) {
        result.error = ModelError{ModelErrorCategory::Build, std::move(code), std::move(message)};
    };
    if (!restoredCheckpoint.ok()) {
        fail("MODEL-CHECKPOINT-REQUIRED", "deployment snapshot requires a successful validated checkpoint reload");
        return result;
    }
    if (request.modelIdentity.empty() || request.inferenceEntry.empty() ||
        restoredCheckpoint.metadata->modelIdentity != request.modelIdentity) {
        fail("MODEL-IDENTITY", "snapshot request disagrees with restored checkpoint identity");
        return result;
    }
    const auto stateErrors = training::verifyTrainingState(plan, *restoredCheckpoint.state);
    if (!stateErrors.empty()) {
        fail("MODEL-CHECKPOINT-STATE", stateErrors.front().code + ": " + stateErrors.front().message);
        return result;
    }
    DeploymentSnapshot snapshot;
    snapshot.modelIdentity = request.modelIdentity;
    snapshot.inferenceEntry = request.inferenceEntry;
    snapshot.publicInputs = request.publicInputs;
    snapshot.result = request.result;
    snapshot.publicBindings = request.publicBindings;
    try { snapshot.parameters = training::storedParameters(plan, *restoredCheckpoint.state); }
    catch (const std::exception& failure) {
        fail("MODEL-CHECKPOINT-STATE", failure.what());
        return result;
    }
    snapshot.parameterDigest = persistence::parameterDigest(snapshot.parameters);
    if (snapshot.parameterDigest != restoredCheckpoint.metadata->parameterDigest) {
        fail("MODEL-PARAMETER-DIGEST", "restored checkpoint and deployment snapshot parameter digests disagree");
        return result;
    }

    std::set<std::size_t> sourceIndices;
    std::set<std::uint32_t> artifactIndices;
    for (const auto& binding : request.parameterBindings) {
        const auto descriptor = std::find_if(plan.parameters.begin(), plan.parameters.end(),
            [&](const auto& candidate) { return candidate.sourceParameterIndex == binding.sourceParameterIndex; });
        if (descriptor == plan.parameters.end()) {
            fail("MODEL-UNEXPECTED-PARAMETER", "snapshot binding references a non-trainable source parameter");
            return result;
        }
        if (!sourceIndices.insert(binding.sourceParameterIndex).second ||
            !artifactIndices.insert(binding.artifactInputIndex).second) {
            fail("MODEL-DUPLICATE-BINDING", "snapshot contains a duplicate parameter binding");
            return result;
        }
        snapshot.parameterBindings.push_back({descriptor->id.value, binding.artifactInputIndex});
    }
    if (sourceIndices.size() != plan.parameters.size()) {
        fail("MODEL-MISSING-PARAMETER", "snapshot does not bind every restored trainable parameter");
        return result;
    }
    std::set<std::uint32_t> publicIndices;
    for (const auto& binding : snapshot.publicBindings) {
        if (binding.publicInputIndex >= snapshot.publicInputs.size() ||
            !publicIndices.insert(binding.publicInputIndex).second ||
            !artifactIndices.insert(binding.artifactInputIndex).second) {
            fail("MODEL-DUPLICATE-BINDING", "snapshot public-input binding is missing, duplicate, or out of range");
            return result;
        }
    }
    if (publicIndices.size() != snapshot.publicInputs.size()) {
        fail("MODEL-MISSING-BINDING", "snapshot does not bind every public input");
        return result;
    }
    result.snapshot = std::move(snapshot);
    return result;
}

} // namespace thiran::v0::model
