#pragma once

#include "analysis/ShapeInference.hpp"

namespace thiran
{

class ShapePrinter
{
public:

    static void print(
        Graph& graph,
        ShapeInference& inference
    );

};

}