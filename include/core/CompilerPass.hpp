#pragma once

#include <string>

#include "core/CompilationContext.hpp"

namespace thiran
{

class CompilerPass
{
public:

    virtual ~CompilerPass() = default;

    virtual std::string name() = 0;

    virtual void run(
        CompilationContext& context
    ) = 0;

};

}