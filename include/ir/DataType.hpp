#pragma once

#include <string>

namespace thiran
{

enum class DataType
{
    Float16,
    Float32,
    Float64,
    Int32,
    Int64,
    Bool
};

std::string toString(DataType type);

}