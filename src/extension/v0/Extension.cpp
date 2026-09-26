#include "extension/v0/Extension.hpp"

#include <bit>
#include <cmath>
#include <dlfcn.h>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace thiran::v0::extension {
namespace {
bool text(const char* value) { return value && *value; }
bool version(std::string_view value) {
    if (value.empty()) return false;
    for (char c : value) if (!(c == '.' || c == '-' || c == '+' || c == '_' ||
        (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) return false;
    return true;
}
void hashByte(std::uint64_t& hash, std::uint8_t value) { hash ^= value; hash *= 1099511628211ULL; }
template<class T> void hashValue(std::uint64_t& hash, T value) {
    for (std::size_t index = 0; index < sizeof(T); ++index)
        hashByte(hash, static_cast<std::uint8_t>(value >> (index * 8)));
}
void hashText(std::uint64_t& hash, std::string_view value) {
    hashValue(hash, static_cast<std::uint64_t>(value.size()));
    for (unsigned char c : value) hashByte(hash, c);
}
void hashRecipe(std::uint64_t& hash, const ScalarRecipe& recipe) {
    hashValue(hash, recipe.inputCount); hashValue(hash, recipe.root);
    hashValue(hash, static_cast<std::uint64_t>(recipe.nodes.size()));
    for (const auto& node : recipe.nodes) {
        hashByte(hash, static_cast<std::uint8_t>(node.opcode)); hashValue(hash, node.left);
        hashValue(hash, node.right); hashValue(hash, node.input);
        hashValue(hash, std::bit_cast<std::uint32_t>(node.constant));
    }
}
std::string finish(std::uint64_t hash) {
    std::ostringstream out; out << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}
ScalarRecipe copyRecipe(const ScalarRecipeV0& recipe) {
    ScalarRecipe result; result.root = recipe.root; result.inputCount = recipe.inputCount;
    if (recipe.nodes && recipe.nodeCount)
        for (std::size_t index = 0; index < recipe.nodeCount; ++index) {
            const auto& node = recipe.nodes[index];
            result.nodes.push_back({node.opcode,node.left,node.right,node.input,node.constant});
        }
    return result;
}
}

std::optional<std::string> validateRecipe(const ScalarRecipe& recipe, std::uint32_t inputs,
                                          bool allowDivide) {
    if (recipe.inputCount != inputs) return "recipe input count disagrees with schema";
    if (recipe.nodes.empty() || recipe.nodes.size() > 256 || recipe.root >= recipe.nodes.size())
        return "recipe node/root count is invalid";
    for (std::size_t index = 0; index < recipe.nodes.size(); ++index) {
        const auto& node = recipe.nodes[index];
        switch (node.opcode) {
        case ScalarOpcode::Input:
            if (node.input >= inputs) return "recipe input reference is out of range";
            break;
        case ScalarOpcode::ConstantF32:
            if (!std::isfinite(node.constant)) return "recipe constant must be finite";
            break;
        case ScalarOpcode::Negate:
            if (node.left >= index) return "recipe unary dependency is not earlier";
            break;
        case ScalarOpcode::Add: case ScalarOpcode::Subtract: case ScalarOpcode::Multiply:
            if (node.left >= index || node.right >= index) return "recipe dependency is not earlier";
            break;
        case ScalarOpcode::Divide:
            if (!allowDivide) return "division requires MayTrap";
            if (node.left >= index || node.right >= index) return "recipe dependency is not earlier";
            break;
        default: return "recipe opcode is invalid";
        }
    }
    return {};
}

std::optional<std::string> validateOperation(const Operation& op) {
    if (op.abiVersion != extensionAbiVersion) return "extension ABI mismatch";
    if (op.extensionId.empty() || op.name.empty() || !version(op.extensionVersion) || !version(op.semanticVersion))
        return "canonical identity field is missing or malformed";
    const auto canonical=op.extensionId+"::"+op.name+"@"+op.semanticVersion+"/abi"+std::to_string(op.abiVersion);
    if (op.canonicalIdentity!=canonical) return "canonical operation identity disagrees with descriptor fields";
    if (op.arity == 0 || op.arity > 8 || op.resultLikeInput >= op.arity) return "operation arity/result rule is invalid";
    if (op.minimumRank < 1 || op.maximumRank > 2 || op.minimumRank > op.maximumRank)
        return "operation rank contract is invalid";
    if (!op.sameShape) return "V0 requires same-shape operands";
    if ((op.backends & Reference) == 0 || (op.backends & ~(Reference|Cpu|Gpu)) != 0)
        return "operation backend contract is invalid";
    if (op.effect != EffectV0::Pure && op.effect != EffectV0::MayTrap) return "operation effect is invalid";
    if (op.fusible && (op.effect != EffectV0::Pure || (op.backends & (Cpu|Gpu)) == 0))
        return "fusibility requires pure native lowering";
    if (auto invalid = validateRecipe(op.forward, op.arity, op.effect == EffectV0::MayTrap)) return invalid;
    if (op.derivative) {
        const auto mask = op.arity == 64 ? ~0ULL : ((1ULL << op.arity) - 1ULL);
        if ((op.derivative->differentiableInputs & ~mask) || (op.derivative->savedInputs & ~mask))
            return "derivative input/save mask is invalid";
        if (op.derivative->gradients.size() != op.arity) return "derivative gradient count disagrees with arity";
        const auto derivativeInputs = op.arity + 2; // primals, result, output cotangent
        for (std::size_t index = 0; index < op.derivative->gradients.size(); ++index) {
            const auto& gradient = op.derivative->gradients[index];
            const bool differentiable = (op.derivative->differentiableInputs & (1ULL << index)) != 0;
            if (differentiable && gradient.nodes.empty()) return "differentiable input has no gradient recipe";
            if (!differentiable && !gradient.nodes.empty()) return "non-differentiable input has a gradient recipe";
            if (!gradient.nodes.empty()) {
                if (auto invalid = validateRecipe(gradient, derivativeInputs, false)) return invalid;
                std::vector<std::uint64_t> dependencies(gradient.nodes.size());
                for (std::size_t nodeIndex=0;nodeIndex<gradient.nodes.size();++nodeIndex) {
                    const auto& node=gradient.nodes[nodeIndex];
                    if (node.opcode==ScalarOpcode::Input) dependencies[nodeIndex]=1ULL<<node.input;
                    else if (node.opcode==ScalarOpcode::Negate) dependencies[nodeIndex]=dependencies[node.left];
                    else if (node.opcode!=ScalarOpcode::ConstantF32)
                        dependencies[nodeIndex]=dependencies[node.left]|dependencies[node.right];
                }
                if (!(dependencies[gradient.root]&(1ULL<<(op.arity+1))))
                    return "gradient recipe must depend on the output cotangent";
                for (const auto& node : gradient.nodes) if (node.opcode == ScalarOpcode::Input) {
                    if (node.input < op.arity && !(op.derivative->savedInputs & (1ULL << node.input)))
                        return "gradient references an undeclared saved primal input";
                    if (node.input == op.arity && !op.derivative->savesOutput)
                        return "gradient references an undeclared saved primal output";
                }
            }
        }
    }
    return {};
}

std::string operationDigest(const Operation& op) {
    std::uint64_t hash = 1469598103934665603ULL;
    hashText(hash,op.extensionId); hashText(hash,op.extensionVersion); hashText(hash,op.name);
    hashText(hash,op.semanticVersion); hashValue(hash,op.abiVersion); hashValue(hash,op.arity);
    hashValue(hash,op.minimumRank); hashValue(hash,op.maximumRank); hashValue(hash,op.resultLikeInput);
    hashByte(hash,op.sameShape); hashByte(hash,static_cast<std::uint8_t>(op.effect));
    hashValue(hash,op.backends); hashByte(hash,op.fusible); hashRecipe(hash,op.forward);
    hashByte(hash,op.derivative.has_value());
    if (op.derivative) {
        hashValue(hash,op.derivative->differentiableInputs); hashValue(hash,op.derivative->savedInputs);
        hashByte(hash,op.derivative->savesOutput); hashValue(hash,static_cast<std::uint64_t>(op.derivative->gradients.size()));
        for (const auto& recipe : op.derivative->gradients) hashRecipe(hash,recipe);
    }
    return finish(hash);
}

float evaluateRecipe(const ScalarRecipe& recipe, const std::vector<float>& inputs) {
    if (auto invalid = validateRecipe(recipe, static_cast<std::uint32_t>(inputs.size()), true))
        throw std::invalid_argument(*invalid);
    std::vector<float> values; values.reserve(recipe.nodes.size());
    for (const auto& node : recipe.nodes) {
        switch (node.opcode) {
        case ScalarOpcode::Input: values.push_back(inputs.at(node.input)); break;
        case ScalarOpcode::ConstantF32: values.push_back(node.constant); break;
        case ScalarOpcode::Add: values.push_back(values[node.left] + values[node.right]); break;
        case ScalarOpcode::Subtract: values.push_back(values[node.left] - values[node.right]); break;
        case ScalarOpcode::Multiply: values.push_back(values[node.left] * values[node.right]); break;
        case ScalarOpcode::Divide:
            if (values[node.right] == 0.0f) throw std::domain_error("TH021-DIVIDE-BY-ZERO");
            values.push_back(values[node.left] / values[node.right]); break;
        case ScalarOpcode::Negate: values.push_back(-values[node.left]); break;
        }
    }
    return values.at(recipe.root);
}

ExtensionRegistry::~ExtensionRegistry() { for (auto it=handles_.rbegin();it!=handles_.rend();++it) if (*it) dlclose(*it); }

RegistryResult ExtensionRegistry::prepare(const ExtensionDescriptorV0& descriptor,
                                          std::vector<Operation>& prepared) const {
    if (descriptor.abiVersion != extensionAbiVersion) return {false,"TH021-ABI","extension ABI mismatch"};
    if (!text(descriptor.extensionId) || !text(descriptor.extensionVersion) || !version(descriptor.extensionVersion))
        return {false,"TH021-DESCRIPTOR","extension identity/version is missing or malformed"};
    if (descriptor.initializationError && *descriptor.initializationError)
        return {false,"TH021-INITIALIZATION",descriptor.initializationError};
    if (!descriptor.operations || descriptor.operationCount == 0 || descriptor.operationCount > 256)
        return {false,"TH021-DESCRIPTOR","extension must declare one to 256 operations"};
    if (extensionVersions_.contains(descriptor.extensionId))
        return {false,"TH021-DUPLICATE-EXTENSION","duplicate extension identity " + std::string(descriptor.extensionId)};
    std::set<std::string> localNames, localIdentities;
    for (std::size_t index=0;index<descriptor.operationCount;++index) {
        const auto& raw=descriptor.operations[index];
        if (!text(raw.name) || !text(raw.semanticVersion))
            return {false,"TH021-DESCRIPTOR","operation identity field is missing"};
        Operation op; op.extensionId=descriptor.extensionId; op.extensionVersion=descriptor.extensionVersion;
        op.name=raw.name; op.semanticVersion=raw.semanticVersion; op.arity=raw.arity;
        op.minimumRank=raw.minimumRank; op.maximumRank=raw.maximumRank; op.resultLikeInput=raw.resultLikeInput;
        op.sameShape=raw.sameShape; op.effect=raw.effect; op.backends=raw.backends; op.fusible=raw.fusible;
        op.forward=copyRecipe(raw.forward);
        if (raw.derivative) {
            Derivative derivative; derivative.differentiableInputs=raw.derivative->differentiableInputs;
            derivative.savedInputs=raw.derivative->savedInputs; derivative.savesOutput=raw.derivative->savesOutput;
            if (!raw.derivative->gradients && raw.derivative->gradientCount)
                return {false,"TH021-DERIVATIVE","gradient recipe table is missing"};
            for (std::size_t g=0;g<raw.derivative->gradientCount;++g)
                derivative.gradients.push_back(copyRecipe(raw.derivative->gradients[g]));
            op.derivative=std::move(derivative);
        }
        op.canonicalIdentity=op.extensionId+"::"+op.name+"@"+op.semanticVersion+"/abi"+std::to_string(op.abiVersion);
        if (auto invalid=validateOperation(op)) return {false,"TH021-DESCRIPTOR",*invalid};
        op.descriptorDigest=operationDigest(op);
        if (!localNames.insert(op.name).second || operationsByName_.contains(op.name))
            return {false,"TH021-DUPLICATE-OP","duplicate source operation name "+op.name};
        if (!localIdentities.insert(op.canonicalIdentity).second)
            return {false,"TH021-DUPLICATE-OP","duplicate canonical operation identity "+op.canonicalIdentity};
        for (const auto& [_,existing]:operationsByName_) if (existing.canonicalIdentity==op.canonicalIdentity)
            return {false,"TH021-DUPLICATE-OP","duplicate canonical operation identity "+op.canonicalIdentity};
        prepared.push_back(std::move(op));
    }
    return {true,{},{}};
}

RegistryResult ExtensionRegistry::registerExtension(const ExtensionDescriptorV0& descriptor) {
    if (frozen_) return {false,"TH021-REGISTRY-FROZEN","late extension registration is rejected"};
    std::vector<Operation> prepared;
    if (auto result=prepare(descriptor,prepared);!result.ok()) return result;
    extensionVersions_.emplace(descriptor.extensionId,descriptor.extensionVersion);
    for (auto& op:prepared) operationsByName_.emplace(op.name,std::move(op));
    return {true,{},{}};
}

RegistryResult ExtensionRegistry::load(const std::string& path) {
    if (frozen_) return {false,"TH021-REGISTRY-FROZEN","late extension loading is rejected"};
    dlerror();
    void* handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!handle) { const char* failure=dlerror(); return {false,"TH021-LOAD",failure?failure:"cannot load extension shared library"}; }
    dlerror();
    auto entry=reinterpret_cast<ExtensionEntryV0>(dlsym(handle,extensionEntrySymbol));
    if (const char* failure=dlerror();failure || !entry) {
        std::string message=failure?failure:"missing extension entry symbol"; dlclose(handle);
        return {false,"TH021-ENTRY",std::move(message)};
    }
    const ExtensionDescriptorV0* descriptor=nullptr;
    try { descriptor=entry(); } catch (...) { dlclose(handle); return {false,"TH021-INITIALIZATION","extension entry threw"}; }
    if (!descriptor) { dlclose(handle); return {false,"TH021-INITIALIZATION","extension entry returned null"}; }
    std::vector<Operation> prepared;
    auto result=prepare(*descriptor,prepared);
    if (!result.ok()) { dlclose(handle); return result; }
    extensionVersions_.emplace(descriptor->extensionId,descriptor->extensionVersion);
    for (auto& op:prepared) operationsByName_.emplace(op.name,std::move(op));
    handles_.push_back(handle);
    return {true,{},{}};
}

RegistryResult ExtensionRegistry::freeze() { frozen_=true; return {true,{},{}}; }
const Operation* ExtensionRegistry::find(std::string_view name) const noexcept {
    auto found=operationsByName_.find(std::string(name)); return found==operationsByName_.end()?nullptr:&found->second;
}
std::vector<std::string> ExtensionRegistry::operationIdentities() const {
    std::vector<std::string> result; for (const auto& [_,op]:operationsByName_) result.push_back(op.canonicalIdentity); return result;
}
std::string ExtensionRegistry::digest() const {
    std::uint64_t hash=1469598103934665603ULL;
    for (const auto& [_,op]:operationsByName_) { hashText(hash,op.canonicalIdentity); hashText(hash,op.descriptorDigest); }
    return finish(hash);
}
}
