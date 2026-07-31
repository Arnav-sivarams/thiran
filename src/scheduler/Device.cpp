#include "scheduler/Device.hpp"

namespace thiran
{

std::string toString(Device device)
{
    switch(device)
    {
        case Device::CPU:
            return "CPU";

        case Device::GPU:
            return "GPU";

        case Device::TPU:
            return "TPU";

        default:
            return "Unknown";
    }
}

}