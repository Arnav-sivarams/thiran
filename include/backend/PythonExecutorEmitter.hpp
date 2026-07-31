#pragma once

#include <string>

#include "backend/BackendIR.hpp"

namespace thiran
{

class PythonExecutorEmitter
{
public:

    static std::string emit(
        BackendIR& backend
    );

};

}