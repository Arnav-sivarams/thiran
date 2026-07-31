#include "analysis/TensorShape.hpp"

namespace thiran
{

TensorShape::TensorShape()
{
}

TensorShape::TensorShape(std::initializer_list<int> dims)
{
    dimensions = dims;
}

int TensorShape::rank() const
{
    return dimensions.size();
}

}