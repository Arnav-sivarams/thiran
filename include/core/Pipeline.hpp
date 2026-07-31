#pragma once

#include <memory>
#include <vector>

#include "core/CompilerPass.hpp"

namespace thiran
{

class Pipeline
{
public:

    void add(
        std::shared_ptr<CompilerPass> pass
    );

    void run(
        CompilationContext& context
    );

private:

    std::vector<
        std::shared_ptr<CompilerPass>
    > passes;

};

}