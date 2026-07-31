#include "backend/TritonCodeGenerator.hpp"

#include "backend/TritonEmitter.hpp"
#include "backend/SymbolBuilder.hpp"
#include "backend/SignatureBuilder.hpp"
#include "backend/PythonExecutorEmitter.hpp"

#include <sstream>

namespace thiran
{

std::string TritonCodeGenerator::generate(
    BackendIR& backend
)
{
    std::stringstream out;

    SymbolBuilder symbolBuilder;
    auto symbols = symbolBuilder.build(backend);

    SignatureBuilder signatureBuilder;

    out << "import torch\n";
    out << "try:\n";
    out << "    import triton\n";
    out << "    import triton.language as tl\n";
    out << "except ImportError:\n";
    out << "    triton = None\n";
    out << "    tl = None\n\n";

    for(auto& kernel : backend.kernels)
    {
        auto signature =
            signatureBuilder.build(*kernel);

        out << "if triton is not None:\n";
        out << "    @triton.jit\n";

        out
            << "    def "
            << kernel->name
            << "(\n";

        for(size_t i = 0; i < signature.parameters.size(); i++)
        {
            out
                << "        "
                << signature.parameters[i];

            if(i + 1 != signature.parameters.size())
            {
                out << ",";
            }

            out << "\n";
        }

        out
            << "    ):\n";

        for(auto op : kernel->operations)
        {
            out << "        "
                << TritonEmitter::emit(
                    op,
                    symbols
                );
        }

        out
            << "        pass\n\n";
    }

    // Emit the runtime / executor
    out
        << PythonExecutorEmitter::emit(
            backend
        );

    return out.str();
}

}
