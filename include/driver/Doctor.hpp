#pragma once

#include <string>

namespace thiran::driver
{

struct DoctorProbeResult final
{
    bool supportedPlatform;
    bool temporaryDirectoryAvailable;
    bool pythonAvailable;
    bool torchAvailable;
    bool cudaVisible;
};

class Doctor final
{
public:
    static DoctorProbeResult probe();
    static std::string format(const DoctorProbeResult& result);
    static int exitCode(const DoctorProbeResult& result) noexcept;
};

}
