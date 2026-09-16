#pragma once

#include "analysis/v0/Ownership.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace thiran::v0::interop {

enum class LinkItemKind { StaticArchive, SharedLibraryName, SearchPath };
struct LinkItem {
    LinkItemKind kind = LinkItemKind::StaticArchive;
    std::string value;
    bool operator==(const LinkItem&) const = default;
};

enum class HostApplicability { PosixNativeCpu, FutureHost };
struct ParameterContract {
    semantic::Type type;
    semantic::AccessMode access = semantic::AccessMode::Read;
};
struct NativeLibraryContract {
    std::string id;
    std::string logicalLibrary;
    std::string symbol;
    std::vector<ParameterContract> parameters;
    semantic::Type result;
    analysis::EffectSet effects = 0;
    bool mayTrap = false;
    bool executable = false;
    std::vector<LinkItem> linkItems;
    HostApplicability applicability = HostApplicability::PosixNativeCpu;
};

struct ContractVerification {
    std::vector<std::string> errors;
    bool ok() const { return errors.empty(); }
};

ContractVerification verifyContracts(const std::vector<NativeLibraryContract>& contracts);

} // namespace thiran::v0::interop
