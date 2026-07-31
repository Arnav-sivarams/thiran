#include "utils/GraphvizEmitter.hpp"

#include <fstream>

namespace thiran
{

bool GraphvizEmitter::write(const Graph& graph, const std::string& path)
{
    std::ofstream output(path);
    if(!output.is_open())
    {
        return false;
    }

    output << "digraph ThiranGraph {\n";
    output << "  rankdir=LR;\n";
    output << "  node [shape=box, style=rounded, fontname=Helvetica];\n";

    for(const auto& holder : graph.nodes)
    {
        const Node* node = holder.get();
        output << "  n" << node->id << " [label=\"" << node->name << "\\n"
               << toString(node->op) << "\"];\n";
    }

    for(const auto& edge : graph.edges)
    {
        output << "  n" << edge.source->id << " -> n" << edge.destination->id << ";\n";
    }

    output << "}\n";
    return true;
}

}
