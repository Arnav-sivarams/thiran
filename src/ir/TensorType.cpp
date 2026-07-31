#include "ir/TensorType.hpp"

namespace thiran
{

TensorType::TensorType()
{
    dtype = DataType::Float32;
}

TensorType::TensorType(
    DataType type,
    std::initializer_list<int> dims
)
{
    dtype = type;

    shape = dims;
}

}