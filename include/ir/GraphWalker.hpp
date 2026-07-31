#pragma once

#include <functional>

#include "ir/Graph.hpp"

namespace thiran
{

class GraphWalker
{
public:

    static void bfs(
        Graph& graph,
        std::function<void(Node*)> visit
    );

    static void dfs(
        Graph& graph,
        std::function<void(Node*)> visit
    );

};

}