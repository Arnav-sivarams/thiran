#pragma once

#include <string>

#include "backend/BackendIR.hpp"
#include "backend/PythonExecutorEmitter.hpp"

namespace thiran
{

class TritonCodeGenerator
{
public:

    std::string generate(
        BackendIR& backend
    );

};

}