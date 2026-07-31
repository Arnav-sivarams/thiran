#include "analysis/GraphStatistics.hpp"

#include <queue>
#include <unordered_map>

namespace thiran
{

int GraphStatistics::nodeCount(Graph& graph)
{
    return graph.nodes.size();
}

int GraphStatistics::edgeCount(Graph& graph)
{
    return graph.edges.size();
}

int GraphStatistics::inputCount(Graph& graph)
{
    int count = 0;

    for(auto& node : graph.nodes)
    {
        if(node->inputs.empty())
        {
            count++;
        }
    }

    return count;
}

int GraphStatistics::outputCount(Graph& graph)
{
    int count = 0;

    for(auto& node : graph.nodes)
    {
        if(node->outputs.empty())
        {
            count++;
        }
    }

    return count;
}

int GraphStatistics::maxDepth(Graph& graph)
{
    std::unordered_map<Node*, int> depth;

    std::queue<Node*> q;

    for(auto& node : graph.nodes)
    {
        if(node->inputs.empty())
        {
            depth[node.get()] = 1;
            q.push(node.get());
        }
    }

    int answer = 0;

    while(!q.empty())
    {
        Node* current = q.front();

        q.pop();

        answer = std::max(answer, depth[current]);

        for(auto next : current->outputs)
        {
            if(depth[next] < depth[current] + 1)
            {
                depth[next] = depth[current] + 1;
            }

            q.push(next);
        }
    }

    return answer;
}

}