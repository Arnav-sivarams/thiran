#include "frontend/FunctionInliner.hpp"

namespace thiran::frontend
{
std::string FunctionInliner::localName(std::size_t callOrdinal, std::size_t functionOrdinal,
                                       std::size_t localOrdinal, const std::string& local)
{
    return "__thiran_call_" + std::to_string(callOrdinal) + "_" +
           std::to_string(functionOrdinal) + "_" + std::to_string(localOrdinal) + "_" + local;
}
}
