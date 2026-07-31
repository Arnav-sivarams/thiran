#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "frontend/SourceManager.hpp"

namespace
{
int failures = 0;
#define CHECK(value) do { if(!(value)) { ++failures; std::cerr << "Failure at line " << __LINE__ << "\n"; } } while(false)

void write(const std::filesystem::path& path, const std::string& value)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output << value;
}
}

int main()
{
    namespace fs = std::filesystem;
    using thiran::frontend::SourceManager;
    const auto root = fs::temp_directory_path() / "thiran-source-manager-tests";
    fs::remove_all(root);
    write(root / "main.th", "A\t= Input()\r\nB = Output(A)\n");
    write(root / "sub" / "value.th", "export C = Constant(1)\n");
    fs::create_directories(root / "folder.th");
    SourceManager manager;
    std::string error;
    const auto entry = manager.loadEntry(root / "main.th", error);
    CHECK(entry && *entry == 0);
    CHECK(manager.source(*entry)->contents == "A\t= Input()\r\nB = Output(A)\n");
    CHECK(manager.source(*entry)->displayPath == "main.th");
    CHECK(manager.location(*entry, 0).line == 1 && manager.location(*entry, 0).column == 1);
    CHECK(manager.location(*entry, 1).column == 2);
    CHECK(manager.location(*entry, 2).column == 3); // tabs advance one byte-column
    CHECK(manager.location(*entry, 14).line == 2);
    CHECK(manager.location(*entry, manager.source(*entry)->contents.size()).offset == manager.source(*entry)->contents.size());
    CHECK(manager.location(999, 0).source == thiran::frontend::invalidSourceId);
    auto imported = manager.resolveImport(*entry, "sub/value.th", error);
    CHECK(imported && *imported == 1);
    CHECK(manager.resolveImport(*entry, "sub/./value.th", error) == imported);
    CHECK(manager.resolveImport(*entry, "sub/../sub/value.th", error) == imported);
    CHECK(!manager.resolveImport(*entry, "/tmp/value.th", error));
    CHECK(!manager.resolveImport(*entry, "../escape.th", error));
    CHECK(!manager.resolveImport(*entry, "missing.th", error));
    CHECK(!manager.resolveImport(*entry, "sub", error));
    CHECK(!manager.resolveImport(*entry, "folder.th", error) && error.find("regular file") != std::string::npos);
    std::error_code ec;
    fs::create_symlink(root / "sub" / "value.th", root / "alias.th", ec);
    if(!ec) CHECK(manager.resolveImport(*entry, "alias.th", error) == imported);
    for(int i = 0; i < 50; ++i) CHECK(manager.resolveImport(*entry, "sub/value.th", error) == imported);
    CHECK(manager.source(*entry)->displayPath.find("0x") == std::string::npos);
    SourceManager empty;
    write(root / "empty.th", "");
    const auto emptyId = empty.loadEntry(root / "empty.th", error);
    CHECK(emptyId && empty.location(*emptyId, 0).line == 1 && empty.location(*emptyId, 0).column == 1);
    fs::remove_all(root);
    if(failures) return 1;
    std::cout << "All Source Manager tests passed\n";
    return 0;
}
