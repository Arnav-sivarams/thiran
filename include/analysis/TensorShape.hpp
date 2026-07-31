#pragma once

#include <vector>

namespace thiran
{

class TensorShape
{
public:

    std::vector<int> dimensions;

    TensorShape();

    TensorShape(std::initializer_list<int> dims);

    int rank() const;

};

}