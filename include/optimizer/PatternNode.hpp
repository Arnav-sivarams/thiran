#pragma once

#include <vector>

#include "ir/Operation.hpp"

namespace thiran
{

class PatternNode
{
public:

    Operation op;

    std::vector<PatternNode*> outputs;

    PatternNode(Operation op);

    void connect(PatternNode* node);
};

}