#include "model/v0/ReferenceModel.hpp"

#include "frontend/v0/Parser.hpp"
#include "semantic/v0/Analyzer.hpp"
#include "semantic/v0/Evaluator.hpp"

#include <algorithm>

namespace thiran::v0::model {
namespace {

semantic::RuntimeValue tensor(std::vector<std::int64_t> shape, std::vector<float> values) {
    return semantic::RuntimeValue{semantic::RuntimeTensor{
        semantic::TypeKind::F32, std::move(shape), {}, std::move(values)}};
}

const training::ParameterValue* sourceParameter(const training::TrainingPlan& plan,
                                                const training::TrainingState& state,
                                                std::size_t sourceIndex) {
    const auto descriptor = std::find_if(plan.parameters.begin(), plan.parameters.end(),
        [&](const auto& candidate) { return candidate.sourceParameterIndex == sourceIndex; });
    if (descriptor == plan.parameters.end()) return nullptr;
    const auto parameter = std::find_if(state.parameters.begin(), state.parameters.end(),
        [&](const auto& candidate) { return candidate.id == descriptor->id; });
    return parameter == state.parameters.end() ? nullptr : &*parameter;
}

} // namespace

std::string referenceAffineSource() {
    return R"THIRAN(fn loss(
    x: Tensor<f32,2>,
    target: Tensor<f32,2>,
    weight: Tensor<f32,2>,
    bias: Tensor<f32,1>
) -> f32 {
    let prediction = x * weight + bias
    let error = prediction - target
    let squared = error .* error
    let reduced = sum(squared, 0)
    return sum(reduced, 0)
}

fn infer(
    x: Tensor<f32,1>,
    weight: Tensor<f32,2>,
    bias: Tensor<f32,1>
) -> Tensor<f32,1> {
    let row = 0
    let column = 0
    let selected_weight = weight[row,column]
    let projected_weight = [selected_weight]
    let scaled = x .* projected_weight
    return scaled + bias
}
)THIRAN";
}

ReferencePlanResult createReferenceAffinePlan(training::OptimizerSpec optimizer) {
    ReferencePlanResult result;
    const auto parsed = parse(referenceAffineSource(), "<th017-reference-affine>");
    if (!parsed.module) {
        result.error = parsed.diagnostics.empty() ? "reference model parse failed" : parsed.diagnostics.front().format();
        return result;
    }
    auto analyzed = semantic::analyze(*parsed.module);
    if (!analyzed.module) {
        result.error = analyzed.diagnostics.empty() ? "reference model analysis failed" : analyzed.diagnostics.front().format();
        return result;
    }
    auto planned = training::createTrainingPlan(*analyzed.module,
        {std::string(referenceAffineLossEntry), {3, 2}, optimizer});
    if (!planned.ok()) {
        result.error = planned.diagnostics.empty() ? "reference training plan failed" :
            planned.diagnostics.front().code + ": " + planned.diagnostics.front().message;
        return result;
    }
    result.module = *analyzed.module;
    result.plan = std::move(*planned.plan);
    return result;
}

training::StateResult initializeReferenceAffineState(const training::TrainingPlan& plan) {
    return training::initializeTrainingState(plan,
        {tensor({1}, {0.0f}), tensor({1,1}, {0.0f})});
}

std::vector<training::RuntimeInput> referenceAffineBatch() {
    return {{0, tensor({4,1}, {0,1,2,3})},
            {1, tensor({4,1}, {1,3,5,7})}};
}

backend::NativeResult referenceAffineInferenceRegion(const semantic::Module& module,
                                                     backend::NativeTarget target) {
    const auto ownership = analysis::analyze(module);
    return backend::extractStrictRegion(module, ownership,
                                        std::string(referenceAffineInferenceEntry), false, target);
}

DeploymentSnapshotRequest referenceAffineSnapshotRequest() {
    const artifact::ArtifactType value{artifact::ValueKind::Tensor, storage::DType::F32, 1,
                                       {std::uint64_t{1}}};
    DeploymentSnapshotRequest request;
    request.modelIdentity = referenceAffineModelIdentity;
    request.inferenceEntry = referenceAffineInferenceEntry;
    request.publicInputs = {value};
    request.result = value;
    request.publicBindings = {{0, 0}};
    request.parameterBindings = {{2, 1}, {3, 2}};
    return request;
}

artifact::ArtifactValue referenceAffinePublicInput(float value) {
    return storage::Tensor::materializeF32({1}, {value});
}

semantic::Observation evaluateReferenceAffine(const semantic::Module& module,
                                              const training::TrainingPlan& plan,
                                              const training::TrainingState& state,
                                              float input) {
    const auto* weight = sourceParameter(plan, state, 2);
    const auto* bias = sourceParameter(plan, state, 3);
    if (!weight || !bias) return {false, {}, "MODEL-MISSING-PARAMETER"};
    return semantic::evaluateCall(module, std::string(referenceAffineInferenceEntry),
                                  {tensor({1}, {input}), weight->value, bias->value});
}

} // namespace thiran::v0::model
