#include "backend/FileEmitter.hpp"

#include <fstream>

namespace thiran
{

bool FileEmitter::write(
    const std::string& path,
    const std::string& code
)
{
    std::ofstream file(path);

    if(!file.is_open())
    {
        return false;
    }

    file << code;

    return true;
}

}