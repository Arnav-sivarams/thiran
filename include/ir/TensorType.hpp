#pragma once

#include <vector>

#include "ir/DataType.hpp"

namespace thiran
{

class TensorType
{
public:

    DataType dtype;

    std::vector<int> shape;

    TensorType();

    TensorType(
        DataType type,
        std::initializer_list<int> dims
    );

};

}