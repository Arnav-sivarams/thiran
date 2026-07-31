#include "analysis/DependencyAnalysis.hpp"

#include <queue>

namespace thiran
{

std::unordered_map<Node*,int>
DependencyAnalysis::indegree(Graph& graph)
{
    std::unordered_map<Node*,int> degree;

    for(auto& node : graph.nodes)
    {
        degree[node.get()] = node->inputs.size();
    }

    return degree;
}

std::vector<std::vector<Node*>>
DependencyAnalysis::executionLevels(Graph& graph)
{
    auto degree = indegree(graph);

    std::queue<Node*> q;

    for(auto& node : graph.nodes)
    {
        if(degree[node.get()] == 0)
        {
            q.push(node.get());
        }
    }

    std::vector<std::vector<Node*>> levels;

    while(!q.empty())
    {
        int count = q.size();

        std::vector<Node*> level;

        while(count--)
        {
            Node* current = q.front();

            q.pop();

            level.push_back(current);

            for(auto next : current->outputs)
            {
                degree[next]--;

                if(degree[next] == 0)
                {
                    q.push(next);
                }
            }
        }

        levels.push_back(level);
    }

    return levels;
}

}