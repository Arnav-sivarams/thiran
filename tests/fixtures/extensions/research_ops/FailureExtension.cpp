#include "extension/v0/Extension.hpp"
using namespace thiran::v0::extension;

#if defined(TH021_NO_ENTRY)
extern "C" int not_the_thiran_entry() { return 1; }
#else
namespace {
constexpr ScalarNodeV0 nodes[]={{ScalarOpcode::Input,0,0,0,0}};
constexpr OperationDescriptorV0 good{"bad_fixture_op","1.0.0",1,1,2,0,true,EffectV0::Pure,
    BackendV0::Reference|BackendV0::Cpu,false,{nodes,1,0,1},nullptr};
#if defined(TH021_MALFORMED)
constexpr OperationDescriptorV0 operations[]={good,{nullptr,"1.0.0",1,1,2,0,true,EffectV0::Pure,
    BackendV0::Reference,false,{nodes,1,0,1},nullptr}};
constexpr ExtensionDescriptorV0 descriptor{extensionAbiVersion,"org.thiran.bad.transaction","1.0.0",operations,2,nullptr};
#elif defined(TH021_INIT_FAIL)
constexpr ExtensionDescriptorV0 descriptor{extensionAbiVersion,"org.thiran.bad.init","1.0.0",&good,1,"fixture initialization failure"};
#elif defined(TH021_ZERO_OPS)
constexpr ExtensionDescriptorV0 descriptor{extensionAbiVersion,"org.thiran.bad.zero","1.0.0",nullptr,0,nullptr};
#else
constexpr ExtensionDescriptorV0 descriptor{999,"org.thiran.bad.abi","1.0.0",&good,1,nullptr};
#endif
}
extern "C" const ExtensionDescriptorV0* thiran_extension_v0() { return &descriptor; }
#endif
