#include "extension/v0/Extension.hpp"

using namespace thiran::v0::extension;

namespace {
constexpr ScalarNodeV0 squareLinear[] = {
    {ScalarOpcode::Input,0,0,0,0},
    {ScalarOpcode::Multiply,0,0,0,0},
    {ScalarOpcode::Add,1,0,0,0}
};
constexpr ScalarNodeV0 pairMix[] = {
    {ScalarOpcode::Input,0,0,0,0},
    {ScalarOpcode::Input,0,0,1,0},
    {ScalarOpcode::Multiply,0,1,0,0},
    {ScalarOpcode::Add,2,0,0,0}
};
constexpr ScalarNodeV0 cpuOnly[] = {
    {ScalarOpcode::Input,0,0,0,0},
    {ScalarOpcode::Add,0,0,0,0}
};
constexpr ScalarNodeV0 checkedRatio[] = {
    {ScalarOpcode::Input,0,0,0,0},
    {ScalarOpcode::Input,0,0,1,0},
    {ScalarOpcode::Divide,0,1,0,0}
};
// Inputs are x, primal output, and output cotangent.
constexpr ScalarNodeV0 squareLinearGradient[] = {
    {ScalarOpcode::Input,0,0,2,0},
    {ScalarOpcode::ConstantF32,0,0,0,2.0f},
    {ScalarOpcode::Input,0,0,0,0},
    {ScalarOpcode::Multiply,1,2,0,0},
    {ScalarOpcode::ConstantF32,0,0,0,1.0f},
    {ScalarOpcode::Add,3,4,0,0},
    {ScalarOpcode::Multiply,0,5,0,0}
};
constexpr ScalarRecipeV0 gradients[] = {
    {squareLinearGradient,7,6,3}
};
constexpr DerivativeV0 derivative{1,1,false,gradients,1};
constexpr OperationDescriptorV0 operations[] = {
    {"research_square_linear","1.0.0",1,1,2,0,true,EffectV0::Pure,
     BackendV0::Reference|BackendV0::Cpu|BackendV0::Gpu,true,
     {squareLinear,3,2,1},&derivative},
    {"research_pair_mix","1.0.0",2,1,2,0,true,EffectV0::Pure,
     BackendV0::Reference|BackendV0::Cpu|BackendV0::Gpu,true,
     {pairMix,4,3,2},nullptr},
    {"research_cpu_only","1.0.0",1,1,2,0,true,EffectV0::Pure,
     BackendV0::Reference|BackendV0::Cpu,true,{cpuOnly,2,1,1},nullptr},
    {"research_checked_ratio","1.0.0",2,1,2,0,true,EffectV0::MayTrap,
     BackendV0::Reference|BackendV0::Cpu|BackendV0::Gpu,false,
     {checkedRatio,3,2,2},nullptr}
};
constexpr ExtensionDescriptorV0 descriptor{
    extensionAbiVersion,"org.thiran.research.fixture","1.0.0",operations,4,nullptr
};
}

extern "C" const ExtensionDescriptorV0* thiran_extension_v0() { return &descriptor; }
