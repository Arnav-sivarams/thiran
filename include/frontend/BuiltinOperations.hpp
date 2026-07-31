#pragma once

#include <string>

#include "ir/Operation.hpp"

namespace thiran::frontend
{
Operation builtinOperation(const std::string& name);
bool isBuiltinOperation(const std::string& name);
}
