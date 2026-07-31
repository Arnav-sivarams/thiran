#include "ir/DataType.hpp"

namespace thiran
{

std::string toString(DataType type)
{
    switch(type)
    {
        case DataType::Float16:
            return "float16";

        case DataType::Float32:
            return "float32";

        case DataType::Float64:
            return "float64";

        case DataType::Int32:
            return "int32";

        case DataType::Int64:
            return "int64";

        case DataType::Bool:
            return "bool";
    }

    return "unknown";
}

}