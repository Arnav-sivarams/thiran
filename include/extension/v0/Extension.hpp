#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace thiran::v0::extension {

inline constexpr std::uint32_t extensionAbiVersion = 1;
inline constexpr const char* extensionEntrySymbol = "thiran_extension_v0";

enum class ScalarOpcode : std::uint8_t { Input = 1, ConstantF32 = 2, Add = 3, Subtract = 4,
                                         Multiply = 5, Divide = 6, Negate = 7 };
struct ScalarNodeV0 {
    ScalarOpcode opcode;
    std::uint32_t left;
    std::uint32_t right;
    std::uint32_t input;
    float constant;
};
struct ScalarRecipeV0 {
    const ScalarNodeV0* nodes;
    std::size_t nodeCount;
    std::uint32_t root;
    std::uint32_t inputCount;
};
enum class EffectV0 : std::uint8_t { Pure = 0, MayTrap = 1 };
enum BackendV0 : std::uint32_t { Reference = 1, Cpu = 2, Gpu = 4 };
struct DerivativeV0 {
    std::uint64_t differentiableInputs;
    std::uint64_t savedInputs;
    bool savesOutput;
    const ScalarRecipeV0* gradients;
    std::size_t gradientCount;
};
struct OperationDescriptorV0 {
    const char* name;
    const char* semanticVersion;
    std::uint32_t arity;
    std::uint32_t minimumRank;
    std::uint32_t maximumRank;
    std::uint32_t resultLikeInput;
    bool sameShape;
    EffectV0 effect;
    std::uint32_t backends;
    bool fusible;
    ScalarRecipeV0 forward;
    const DerivativeV0* derivative;
};
struct ExtensionDescriptorV0 {
    std::uint32_t abiVersion;
    const char* extensionId;
    const char* extensionVersion;
    const OperationDescriptorV0* operations;
    std::size_t operationCount;
    const char* initializationError;
};
using ExtensionEntryV0 = const ExtensionDescriptorV0* (*)();

struct ScalarNode {
    ScalarOpcode opcode = ScalarOpcode::Input;
    std::uint32_t left = 0, right = 0, input = 0;
    float constant = 0;
    bool operator==(const ScalarNode&) const = default;
};
struct ScalarRecipe {
    std::vector<ScalarNode> nodes;
    std::uint32_t root = 0, inputCount = 0;
    bool operator==(const ScalarRecipe&) const = default;
};
struct Derivative {
    std::uint64_t differentiableInputs = 0, savedInputs = 0;
    bool savesOutput = false;
    std::vector<ScalarRecipe> gradients;
    bool operator==(const Derivative&) const = default;
};
struct Operation {
    std::string extensionId, extensionVersion, name, semanticVersion;
    std::uint32_t abiVersion = extensionAbiVersion, arity = 0;
    std::uint32_t minimumRank = 1, maximumRank = 2, resultLikeInput = 0;
    bool sameShape = true;
    EffectV0 effect = EffectV0::Pure;
    std::uint32_t backends = 0;
    bool fusible = false;
    ScalarRecipe forward;
    std::optional<Derivative> derivative;
    std::string canonicalIdentity, descriptorDigest;
    bool operator==(const Operation&) const = default;
};

std::optional<std::string> validateRecipe(const ScalarRecipe&, std::uint32_t inputs,
                                          bool allowDivide);
std::optional<std::string> validateOperation(const Operation&);
std::string operationDigest(const Operation&);
float evaluateRecipe(const ScalarRecipe&, const std::vector<float>&);

struct RegistryResult {
    bool success = false;
    std::string code, message;
    bool ok() const noexcept { return success; }
};

class ExtensionRegistry {
public:
    ExtensionRegistry() = default;
    ~ExtensionRegistry();
    ExtensionRegistry(const ExtensionRegistry&) = delete;
    ExtensionRegistry& operator=(const ExtensionRegistry&) = delete;
    RegistryResult load(const std::string& path);
    RegistryResult registerExtension(const ExtensionDescriptorV0&);
    RegistryResult freeze();
    bool frozen() const noexcept { return frozen_; }
    const Operation* find(std::string_view sourceName) const noexcept;
    std::vector<std::string> operationIdentities() const;
    std::string digest() const;
private:
    RegistryResult prepare(const ExtensionDescriptorV0&, std::vector<Operation>&) const;
    std::map<std::string, Operation> operationsByName_;
    std::map<std::string, std::string> extensionVersions_;
    std::vector<void*> handles_;
    bool frozen_ = false;
};

}

extern "C" {
const thiran::v0::extension::ExtensionDescriptorV0* thiran_extension_v0();
}
