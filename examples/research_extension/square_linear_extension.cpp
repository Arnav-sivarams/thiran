#include "extension/v0/Extension.hpp"

using namespace thiran::v0::extension;

namespace {
constexpr ScalarNodeV0 forwardNodes[] = {
    {ScalarOpcode::Input, 0, 0, 0, 0.0f},
    {ScalarOpcode::Multiply, 0, 0, 0, 0.0f},
    {ScalarOpcode::Add, 1, 0, 0, 0.0f},
};

// Gradient inputs are the primal input, primal output, and output cotangent.
constexpr ScalarNodeV0 gradientNodes[] = {
    {ScalarOpcode::Input, 0, 0, 2, 0.0f},
    {ScalarOpcode::ConstantF32, 0, 0, 0, 2.0f},
    {ScalarOpcode::Input, 0, 0, 0, 0.0f},
    {ScalarOpcode::Multiply, 1, 2, 0, 0.0f},
    {ScalarOpcode::ConstantF32, 0, 0, 0, 1.0f},
    {ScalarOpcode::Add, 3, 4, 0, 0.0f},
    {ScalarOpcode::Multiply, 0, 5, 0, 0.0f},
};

constexpr ScalarRecipeV0 gradients[] = {
    {gradientNodes, 7, 6, 3},
};
constexpr DerivativeV0 derivative{1, 1, false, gradients, 1};
constexpr OperationDescriptorV0 operations[] = {
    {"research_square_linear", "1.0.0", 1, 1, 2, 0, true,
     EffectV0::Pure, BackendV0::Reference | BackendV0::Cpu | BackendV0::Gpu,
     true, {forwardNodes, 3, 2, 1}, &derivative},
};
constexpr ExtensionDescriptorV0 descriptor{
    extensionAbiVersion, "org.thiran.examples.square-linear", "1.0.0",
    operations, 1, nullptr,
};
} // namespace

extern "C" const ExtensionDescriptorV0* thiran_extension_v0() {
    return &descriptor;
}
