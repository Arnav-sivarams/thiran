#pragma once

#include "model/v0/DeploymentBuild.hpp"

namespace thiran::v0::model {

inline constexpr std::string_view referenceAffineModelIdentity = "thiran.reference.affine.v0";
inline constexpr std::string_view referenceAffineLossEntry = "loss";
inline constexpr std::string_view referenceAffineInferenceEntry = "infer";

struct ReferencePlanResult {
    std::optional<semantic::Module> module;
    std::optional<training::TrainingPlan> plan;
    std::string error;
    bool ok() const noexcept { return module.has_value() && plan.has_value() && error.empty(); }
};

std::string referenceAffineSource();
ReferencePlanResult createReferenceAffinePlan(
    training::OptimizerSpec optimizer = {training::OptimizerKind::SGD, 0.02f, 0.0f});
training::StateResult initializeReferenceAffineState(const training::TrainingPlan&);
std::vector<training::RuntimeInput> referenceAffineBatch();
backend::NativeResult referenceAffineInferenceRegion(const semantic::Module&,
                                                     backend::NativeTarget);
DeploymentSnapshotRequest referenceAffineSnapshotRequest();
artifact::ArtifactValue referenceAffinePublicInput(float);
semantic::Observation evaluateReferenceAffine(const semantic::Module&,
                                              const training::TrainingPlan&,
                                              const training::TrainingState&,
                                              float input);

} // namespace thiran::v0::model
