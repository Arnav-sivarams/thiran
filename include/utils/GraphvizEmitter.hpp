#pragma once

#include <string>

#include "ir/Graph.hpp"

namespace thiran
{

class GraphvizEmitter
{
public:
    static bool write(const Graph& graph, const std::string& path);
};

}
