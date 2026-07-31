#pragma once

#include <string>

namespace thiran
{

enum class Device
{
    CPU,
    GPU,
    TPU,
    Unknown
};

std::string toString(Device device);

}