#pragma once

#include <string>

namespace thiran
{

class FileEmitter
{
public:

    static bool write(
        const std::string& path,
        const std::string& code
    );

};

}