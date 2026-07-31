#include "ir/GraphWalker.hpp"

#include <queue>
#include <stack>
#include <unordered_set>

namespace thiran
{

void GraphWalker::bfs(
    Graph& graph,
    std::function<void(Node*)> visit
)
{
    std::queue<Node*> q;

    std::unordered_set<Node*> seen;

    for(auto& node : graph.nodes)
    {
        if(node->inputs.empty())
        {
            q.push(node.get());
            seen.insert(node.get());
        }
    }

    while(!q.empty())
    {
        Node* current = q.front();

        q.pop();

        visit(current);

        for(auto next : current->outputs)
        {
            if(seen.count(next))
            {
                continue;
            }

            seen.insert(next);

            q.push(next);
        }
    }
}

void GraphWalker::dfs(
    Graph& graph,
    std::function<void(Node*)> visit
)
{
    std::stack<Node*> s;

    std::unordered_set<Node*> seen;

    for(auto& node : graph.nodes)
    {
        if(node->inputs.empty())
        {
            s.push(node.get());
        }
    }

    while(!s.empty())
    {
        Node* current = s.top();

        s.pop();

        if(seen.count(current))
        {
            continue;
        }

        seen.insert(current);

        visit(current);

        for(auto next : current->outputs)
        {
            s.push(next);
        }
    }
}

}