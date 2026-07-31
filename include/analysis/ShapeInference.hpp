#pragma once

#include <unordered_map>
#include <string>
#include <vector>

#include "analysis/TensorShape.hpp"
#include "ir/Graph.hpp"

namespace thiran
{

class ShapeInference
{
public:

    void infer(Graph& graph);

    TensorShape shape(Node* node);

    bool hasErrors() const;

    const std::vector<std::string>& diagnostics() const;

private:

    std::unordered_map<Node*, TensorShape> shapes;

    std::vector<std::string> errors;

    void report(const std::string& message);

};

}
