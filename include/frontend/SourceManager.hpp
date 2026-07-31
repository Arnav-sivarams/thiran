#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "frontend/SourceLocation.hpp"

namespace thiran::frontend
{
struct SourceFile final
{
    SourceId id;
    std::filesystem::path canonicalPath;
    std::string displayPath;
    std::string contents;
};

class SourceManager final
{
public:
    std::optional<SourceId> loadEntry(const std::filesystem::path& path, std::string& error);
    std::optional<SourceId> resolveImport(SourceId importer, const std::string& requested, std::string& error);
    const SourceFile* source(SourceId id) const noexcept;
    SourceLocation location(SourceId id, std::size_t offset) const noexcept;
    const std::filesystem::path& projectRoot() const noexcept { return projectRoot_; }

private:
    std::optional<SourceId> loadCanonical(const std::filesystem::path& path, std::string& error);
    std::filesystem::path projectRoot_;
    std::vector<SourceFile> sources_;
};
}
