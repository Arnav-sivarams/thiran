#include "interop/v0/NativeLibrary.hpp"

#include <cctype>
#include <set>

namespace thiran::v0::interop {
namespace {
bool resourceType(const semantic::Type& type) {
    return type.kind == semantic::TypeKind::Tensor || type.kind == semantic::TypeKind::Buffer;
}
bool symbolName(const std::string& value) {
    if (value.empty() || !(std::isalpha(static_cast<unsigned char>(value[0])) || value[0] == '_')) return false;
    for (char c : value)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    return true;
}
bool libraryName(const std::string& value) {
    if (value.empty()) return false;
    for (char c : value)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '+' || c == '.' || c == '-')) return false;
    return value != "." && value != "..";
}
bool safePathText(const std::string& value) {
    if (value.empty() || value.find('\0') != std::string::npos) return false;
    for (unsigned char c : value) if (c < 0x20 || c == 0x7f) return false;
    return true;
}
bool effect(analysis::EffectSet set, analysis::EffectKind kind) {
    return (set & static_cast<analysis::EffectSet>(kind)) != 0;
}
}

ContractVerification verifyContracts(const std::vector<NativeLibraryContract>& contracts) {
    ContractVerification out;
    std::set<std::string> ids, names;
    constexpr analysis::EffectSet knownEffects =
        static_cast<analysis::EffectSet>(analysis::EffectKind::MayTrap) |
        static_cast<analysis::EffectSet>(analysis::EffectKind::Mutates) |
        static_cast<analysis::EffectSet>(analysis::EffectKind::RNG) |
        static_cast<analysis::EffectSet>(analysis::EffectKind::IO) |
        static_cast<analysis::EffectSet>(analysis::EffectKind::Transfer) |
        static_cast<analysis::EffectSet>(analysis::EffectKind::Async);
    for (const auto& contract : contracts) {
        if (contract.id.empty() || !ids.insert(contract.id).second)
            out.errors.push_back("LIBV01 duplicate or empty contract ID");
        const auto qualified = contract.logicalLibrary + "::" + contract.symbol;
        if (contract.logicalLibrary.empty() || !names.insert(qualified).second)
            out.errors.push_back("LIBV01 duplicate or empty logical contract name");
        if (!symbolName(contract.symbol)) out.errors.push_back("LIBV02 empty or invalid native symbol");
        if (!semantic::validType(contract.result)) out.errors.push_back("LIBV03 invalid result semantic type");
        if (contract.executable && !semantic::executableType(contract.result))
            out.errors.push_back("LIBV04 unsupported result falsely marked executable");
        if ((contract.effects & ~knownEffects) != 0) out.errors.push_back("LIBV05 unknown effect bit");
        bool mutableResource = false;
        for (const auto& parameter : contract.parameters) {
            if (!semantic::validType(parameter.type)) out.errors.push_back("LIBV03 invalid parameter semantic type");
            if (parameter.access == semantic::AccessMode::MutableBorrow && !resourceType(parameter.type))
                out.errors.push_back("LIBV06 MutableBorrow scalar is unsupported");
            if ((parameter.access == semantic::AccessMode::MutableBorrow ||
                 parameter.access == semantic::AccessMode::Consume) && resourceType(parameter.type))
                mutableResource = true;
            if (parameter.access == semantic::AccessMode::MutableBorrow &&
                !effect(contract.effects, analysis::EffectKind::Mutates))
                out.errors.push_back("LIBV07 mutable access lacks Mutates effect");
        }
        if (effect(contract.effects, analysis::EffectKind::Mutates) && !mutableResource)
            out.errors.push_back("LIBV08 Mutates effect lacks mutable or consuming resource parameter");
        if (contract.mayTrap != effect(contract.effects, analysis::EffectKind::MayTrap))
            out.errors.push_back("LIBV09 may-trap flag/effect inconsistency");
        for (const auto& item : contract.linkItems) {
            bool valid = false;
            if (item.kind == LinkItemKind::SharedLibraryName) valid = libraryName(item.value);
            else if (safePathText(item.value)) valid = std::filesystem::path(item.value).is_absolute();
            if (!valid) out.errors.push_back("LIBV10 malformed structured link item");
        }
    }
    return out;
}

} // namespace thiran::v0::interop
