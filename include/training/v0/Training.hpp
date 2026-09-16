#pragma once

#include "autodiff/v0/Autodiff.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace thiran::v0::training {

struct ParameterId {
    std::uint64_t value = 0;
    bool operator==(const ParameterId&) const = default;
};

struct TrainingDiagnostic {
    std::string code;
    std::string message;
};

enum class OptimizerKind : std::uint8_t { SGD = 1, SGDMomentum = 2 };

struct OptimizerSpec {
    OptimizerKind kind = OptimizerKind::SGD;
    float learningRate = 0.0f;
    float momentum = 0.0f;
};

struct ParameterDescriptor {
    ParameterId id;
    std::size_t sourceParameterIndex = 0;
    std::string sourceParameterName;
    semantic::Type type;
    semantic::ShapeFact expectedShape;
    std::size_t position = 0;
};

struct TrainingPlanRequest {
    std::string function;
    std::vector<std::size_t> trainableParameters;
    OptimizerSpec optimizer;
};

// This is an internal developer object. A successfully constructed plan is
// immutable by contract; public fields permit independent adversarial audits.
struct TrainingPlan {
    semantic::Module source;
    analysis::OwnershipAnalysisResult ownership;
    semantic::FunctionId function = 0;
    std::vector<std::size_t> runtimeParameterIndices;
    std::vector<ParameterDescriptor> parameters;
    autodiff::ReverseModeRequest reverseRequest;
    autodiff::DifferentiationResult differentiated;
    analysis::OwnershipAnalysisResult generatedOwnership;
    OptimizerSpec optimizer;
    std::uint64_t signature = 0;
    std::uint64_t generationCount = 0;
};

struct PlanResult {
    std::optional<TrainingPlan> plan;
    std::vector<TrainingDiagnostic> diagnostics;
    bool ok() const { return plan.has_value() && diagnostics.empty(); }
};

struct ParameterValue {
    ParameterId id;
    semantic::Type type;
    std::vector<std::int64_t> shape;
    semantic::RuntimeValue value;
};

struct VelocityValue {
    ParameterId id;
    semantic::Type type;
    std::vector<std::int64_t> shape;
    semantic::RuntimeValue value;
};

struct OptimizerState {
    OptimizerKind kind = OptimizerKind::SGD;
    std::uint64_t step = 0;
    std::vector<VelocityValue> velocities;
};

struct TrainingState {
    std::uint64_t planSignature = 0;
    std::uint64_t step = 0;
    std::vector<ParameterValue> parameters;
    OptimizerState optimizer;
};

struct StateResult {
    std::optional<TrainingState> state;
    std::vector<TrainingDiagnostic> diagnostics;
    bool ok() const { return state.has_value() && diagnostics.empty(); }
};

struct RuntimeInput {
    std::size_t sourceParameterIndex = 0;
    semantic::RuntimeValue value;
};

struct GradientValue {
    ParameterId id;
    semantic::Type type;
    std::vector<std::int64_t> shape;
    semantic::RuntimeValue value;
};

struct OptimizerResult {
    std::optional<TrainingState> state;
    std::vector<TrainingDiagnostic> diagnostics;
    bool ok() const { return state.has_value() && diagnostics.empty(); }
};

struct TrainingStepResult {
    std::optional<float> loss;
    std::vector<GradientValue> gradients;
    std::optional<TrainingState> nextState;
    std::vector<TrainingDiagnostic> diagnostics;
    bool ok() const { return loss.has_value() && nextState.has_value() && diagnostics.empty(); }
};

PlanResult createTrainingPlan(const semantic::Module&, const TrainingPlanRequest&);
std::vector<TrainingDiagnostic> verifyTrainingPlan(const TrainingPlan&);
StateResult initializeTrainingState(const TrainingPlan&,
                                    const std::vector<semantic::RuntimeValue>& orderedParameters);
std::vector<TrainingDiagnostic> verifyOptimizerState(const TrainingPlan&, const TrainingState&);
OptimizerResult applyOptimizer(const TrainingPlan&, const TrainingState&,
                               const std::vector<GradientValue>& orderedGradients);
TrainingStepResult trainingStep(const TrainingPlan&, const TrainingState&,
                                const std::vector<RuntimeInput>& runtimeInputs);
std::string dumpTrainingStep(const TrainingPlan&, const TrainingStepResult&);

} // namespace thiran::v0::training
