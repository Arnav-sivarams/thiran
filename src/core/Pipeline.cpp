#include "core/Pipeline.hpp"

#include <iostream>

namespace thiran
{

void Pipeline::add(
    std::shared_ptr<CompilerPass> pass
)
{
    passes.push_back(pass);
}

void Pipeline::run(
    CompilationContext& context
)
{
    for(auto& pass : passes)
    {
        std::cout
            << "Running "
            << pass->name()
            << "\n";

        pass->run(context);
    }
}

}