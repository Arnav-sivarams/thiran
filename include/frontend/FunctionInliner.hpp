#pragma once

#include <string>

namespace thiran::frontend
{
class FunctionInliner final
{
public:
    static std::string localName(std::size_t callOrdinal, std::size_t functionOrdinal,
                                 std::size_t localOrdinal, const std::string& local);
};
}
