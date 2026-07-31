#include "frontend/SourceManager.hpp"

#include <fstream>
#include <sstream>

namespace thiran::frontend
{
namespace
{
bool contained(const std::filesystem::path& root, const std::filesystem::path& candidate)
{
    auto r = root.begin();
    auto c = candidate.begin();
    for(; r != root.end(); ++r, ++c)
        if(c == candidate.end() || *r != *c) return false;
    return true;
}
}

std::optional<SourceId> SourceManager::loadEntry(const std::filesystem::path& path, std::string& error)
{
    std::error_code ec;
    const auto canonical = std::filesystem::canonical(path, ec);
    if(ec || !std::filesystem::is_regular_file(canonical))
    {
        error = "could not open source file '" + path.string() + "'";
        return std::nullopt;
    }
    projectRoot_ = canonical.parent_path();
    return loadCanonical(canonical, error);
}

std::optional<SourceId> SourceManager::loadCanonical(const std::filesystem::path& path, std::string& error)
{
    for(const auto& item : sources_) if(item.canonicalPath == path) return item.id;
    std::ifstream input(path, std::ios::binary);
    if(!input)
    {
        error = "could not read source file '" + path.string() + "'";
        return std::nullopt;
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    const SourceId id = sources_.size();
    std::error_code ec;
    auto relative = std::filesystem::relative(path, projectRoot_, ec);
    sources_.push_back({id, path, ec ? path.filename().string() : relative.generic_string(), contents.str()});
    return id;
}

std::optional<SourceId> SourceManager::resolveImport(SourceId importer, const std::string& requested, std::string& error)
{
    const auto* owner = source(importer);
    if(owner == nullptr) { error = "invalid source identifier"; return std::nullopt; }
    const std::filesystem::path lexical(requested);
    if(lexical.is_absolute()) { error = "absolute import paths are not allowed: '" + requested + "'"; return std::nullopt; }
    if(lexical.extension() != ".th") { error = "import path must have .th extension: '" + requested + "'"; return std::nullopt; }
    std::error_code ec;
    const auto candidate = std::filesystem::weakly_canonical(owner->canonicalPath.parent_path() / lexical, ec);
    if(ec) { error = "could not resolve import '" + requested + "'"; return std::nullopt; }
    if(!contained(projectRoot_, candidate)) { error = "import escapes project root: '" + requested + "'"; return std::nullopt; }
    if(!std::filesystem::exists(candidate)) { error = "could not resolve import '" + requested + "'"; return std::nullopt; }
    if(!std::filesystem::is_regular_file(candidate)) { error = "import is not a regular file: '" + requested + "'"; return std::nullopt; }
    return loadCanonical(candidate, error);
}

const SourceFile* SourceManager::source(SourceId id) const noexcept
{
    return id < sources_.size() ? &sources_[id] : nullptr;
}

SourceLocation SourceManager::location(SourceId id, std::size_t offset) const noexcept
{
    const auto* file = source(id);
    if(file == nullptr) return {};
    offset = std::min(offset, file->contents.size());
    std::size_t line = 1, column = 1;
    for(std::size_t i = 0; i < offset; ++i)
    {
        if(file->contents[i] == '\n') { ++line; column = 1; }
        else { ++column; }
    }
    return {id, offset, line, column};
}
}
