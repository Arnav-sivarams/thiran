#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ir/Graph.hpp"
#include "optimizer/CanonicalizationPass.hpp"
#include "optimizer/ConstantFold.hpp"
#include "optimizer/DCEPass.hpp"
#include "optimizer/FusionRule.hpp"
#include "optimizer/GraphVerifier.hpp"
#include "optimizer/RemoveDoubleReluRule.hpp"
#include "optimizer/RewriteEngine.hpp"
#include "optimizer/RewriteRule.hpp"

namespace
{

int failures = 0;
#define CHECK(value) do { if(!(value)) { ++failures; std::cerr << "Failure at line " << __LINE__ << ": " #value "\n"; } } while(false)

using thiran::Graph;
using thiran::Node;
using thiran::Operation;

class InsertDoubleReluRule : public thiran::RewriteRule
{
public:
    bool inserted = false;
    std::string getName() override { return "InsertDoubleRelu"; }
    bool match(Node* node) override { return !inserted && node->name == "X"; }
    bool rewrite(Graph& graph, Node* node) override
    {
        Node* output = graph.findNode("O");
        Node* first = graph.createNode("R1", Operation::ReLU);
        Node* second = graph.createNode("R2", Operation::ReLU);
        graph.disconnect(node, output);
        graph.connect(node, first);
        graph.connect(first, second);
        graph.connect(second, output);
        inserted = true;
        return true;
    }
};

std::string signature(const Graph& graph)
{
    std::string result;
    for(const auto& holder : graph.nodes)
    {
        result += holder->name + ":" + std::to_string(holder->id) + "(";
        for(Node* input : holder->inputs)
        {
            result += input->name + ",";
        }
        result += ");";
    }
    result += " edges:";
    for(const auto& edge : graph.edges)
    {
        result += edge.source->name + ">" + edge.destination->name + ";";
    }
    return result;
}

void rewrite(Graph& graph)
{
    thiran::RewriteEngine engine;
    engine.addRule(std::make_shared<thiran::RemoveDoubleReluRule>());
    engine.run(graph);
}

void doubleRelu()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* o = graph.createNode("O", Operation::Output);
    graph.connect(x, a);
    graph.connect(a, b);
    graph.connect(b, o);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    rewrite(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(signature(graph) == "X:0();A:1(X,);O:3(A,); edges:X>A;A>O;");
    CHECK(graph.findNode("B") == nullptr);
    CHECK(!graph.containsNode(b));
    CHECK(o->inputs.size() == 1 && o->inputs[0] == a);
    CHECK(a->outputs.size() == 1 && a->outputs[0] == o);
}

void tripleRelu()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* c = graph.createNode("C", Operation::ReLU);
    Node* o = graph.createNode("O", Operation::Output);
    graph.connect(x, a);
    graph.connect(a, b);
    graph.connect(b, c);
    graph.connect(c, o);
    rewrite(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(signature(graph) == "X:0();A:1(X,);O:4(A,); edges:X>A;A>O;");
    CHECK(!graph.containsNode(b) && !graph.containsNode(c));
    CHECK(o->inputs[0] == a);
}

void independentRewrites()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* o1 = graph.createNode("O1", Operation::Output);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* c = graph.createNode("C", Operation::ReLU);
    Node* d = graph.createNode("D", Operation::ReLU);
    Node* o2 = graph.createNode("O2", Operation::Output);
    graph.connect(x, a);
    graph.connect(a, b);
    graph.connect(b, o1);
    graph.connect(y, c);
    graph.connect(c, d);
    graph.connect(d, o2);
    rewrite(graph);
    const std::string expected = "X:0();A:1(X,);O1:3(A,);Y:4();C:5(Y,);O2:7(C,); edges:X>A;A>O1;Y>C;C>O2;";
    CHECK(signature(graph) == expected);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(!graph.containsNode(b) && !graph.containsNode(d));
    CHECK(o1->inputs[0] == a && o2->inputs[0] == c);
    rewrite(graph);
    CHECK(signature(graph) == expected);
}

void noMatch()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* o = graph.createNode("O", Operation::Output);
    graph.connect(x, a);
    graph.connect(a, o);
    const auto before = signature(graph);
    const auto owners = graph.nodes;
    rewrite(graph);
    CHECK(signature(graph) == before);
    CHECK(graph.nodes == owners);
    CHECK(thiran::GraphVerifier{}.verify(graph));
}

void createdNodesJoinNextRound()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* o = graph.createNode("O", Operation::Output);
    graph.connect(x, o);
    auto insert = std::make_shared<InsertDoubleReluRule>();
    thiran::RewriteEngine engine;
    engine.addRule(insert);
    engine.addRule(std::make_shared<thiran::RemoveDoubleReluRule>());
    engine.run(graph);
    CHECK(insert->inserted);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(signature(graph) == "X:0();O:1(R1,);R1:2(X,); edges:X>R1;R1>O;");
    CHECK(graph.findNode("R2") == nullptr);
}

void preserveOperandOrder()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* s = graph.createNode("S", Operation::Subtract);
    Node* o = graph.createNode("O", Operation::Output);
    graph.connect(x, a);
    graph.connect(a, b);
    graph.connect(b, s);
    graph.connect(y, s);
    graph.connect(s, o);
    rewrite(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(s->inputs.size() == 2 && s->inputs[0] == a && s->inputs[1] == y);
    CHECK(graph.edges[1].source == a && graph.edges[1].destination == s);
}

void collisionIsNotRewritten()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* s = graph.createNode("S", Operation::Subtract);
    Node* o = graph.createNode("O", Operation::Output);
    graph.connect(x, a);
    graph.connect(a, b);
    graph.connect(a, s);
    graph.connect(b, s);
    graph.connect(s, o);
    const auto before = signature(graph);
    rewrite(graph);
    CHECK(signature(graph) == before);
    CHECK(s->inputs.size() == 2 && s->inputs[0] == a && s->inputs[1] == b);
    CHECK(thiran::GraphVerifier{}.verify(graph));
}

void constantFoldSlotZero()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* zero = graph.createNode("Zero", Operation::Constant);
    Node* folded = graph.createNode("Folded", Operation::Add);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* subtract = graph.createNode("Subtract", Operation::Subtract);
    Node* output = graph.createNode("Output", Operation::Output);
    zero->constantValue = 0.0f;
    graph.connect(x, folded);
    graph.connect(zero, folded);
    graph.connect(folded, subtract);
    graph.connect(y, subtract);
    graph.connect(subtract, output);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    thiran::ConstantFold{}.run(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(!graph.containsNode(folded));
    CHECK(subtract->inputs.size() == 2 && subtract->inputs[0] == x && subtract->inputs[1] == y);
    CHECK(output->inputs.size() == 1 && output->inputs[0] == subtract);
    CHECK(signature(graph) == "X:0();Zero:1();Y:3();Subtract:4(X,Y,);Output:5(Subtract,); edges:X>Subtract;Y>Subtract;Subtract>Output;");
    const auto after = signature(graph);
    thiran::ConstantFold{}.run(graph);
    CHECK(signature(graph) == after);
    CHECK(thiran::GraphVerifier{}.verify(graph));
}

void constantFoldSlotOne()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* one = graph.createNode("One", Operation::Constant);
    Node* folded = graph.createNode("Folded", Operation::Multiply);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* divide = graph.createNode("Divide", Operation::Divide);
    Node* output = graph.createNode("Output", Operation::Output);
    one->constantValue = 1.0f;
    graph.connect(x, folded);
    graph.connect(one, folded);
    graph.connect(y, divide);
    graph.connect(folded, divide);
    graph.connect(divide, output);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    thiran::ConstantFold{}.run(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(!graph.containsNode(folded));
    CHECK(divide->inputs.size() == 2 && divide->inputs[0] == y && divide->inputs[1] == x);
    CHECK(output->inputs.size() == 1 && output->inputs[0] == divide);
    CHECK(signature(graph) == "X:0();One:1();Y:3();Divide:4(Y,X,);Output:5(Divide,); edges:Y>Divide;X>Divide;Divide>Output;");
}

void constantFoldMultipleConsumers()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* zero = graph.createNode("Zero", Operation::Constant);
    Node* folded = graph.createNode("Folded", Operation::Add);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* first = graph.createNode("First", Operation::Subtract);
    Node* second = graph.createNode("Second", Operation::Divide);
    Node* output = graph.createNode("Output", Operation::Output);
    zero->constantValue = 0.0f;
    graph.connect(x, folded);
    graph.connect(zero, folded);
    graph.connect(folded, first);
    graph.connect(y, first);
    graph.connect(y, second);
    graph.connect(folded, second);
    graph.connect(folded, output);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    thiran::ConstantFold{}.run(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(!graph.containsNode(folded));
    CHECK(first->inputs.size() == 2 && first->inputs[0] == x && first->inputs[1] == y);
    CHECK(second->inputs.size() == 2 && second->inputs[0] == y && second->inputs[1] == x);
    CHECK(output->inputs.size() == 1 && output->inputs[0] == x);
    CHECK(signature(graph) == "X:0();Zero:1();Y:3();First:4(X,Y,);Second:5(Y,X,);Output:6(X,); edges:X>First;Y>First;Y>Second;X>Second;X>Output;");
}

void constantFoldCollisionRefusal()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* zero = graph.createNode("Zero", Operation::Constant);
    Node* folded = graph.createNode("Folded", Operation::Add);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* legal = graph.createNode("Legal", Operation::Subtract);
    Node* collision = graph.createNode("Collision", Operation::Subtract);
    Node* output = graph.createNode("Output", Operation::Output);
    zero->constantValue = 0.0f;
    graph.connect(x, folded);
    graph.connect(zero, folded);
    graph.connect(folded, legal);
    graph.connect(y, legal);
    graph.connect(folded, collision);
    graph.connect(x, collision);
    graph.connect(collision, output);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    const auto before = signature(graph);
    const auto owners = graph.nodes;
    thiran::ConstantFold{}.run(graph);
    CHECK(signature(graph) == before);
    CHECK(graph.nodes == owners);
    CHECK(graph.containsNode(folded));
    CHECK(legal->inputs.size() == 2 && legal->inputs[0] == folded && legal->inputs[1] == y);
    CHECK(collision->inputs.size() == 2 && collision->inputs[0] == folded && collision->inputs[1] == x);
    CHECK(output->inputs.size() == 1 && output->inputs[0] == collision);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    thiran::ConstantFold{}.run(graph);
    CHECK(signature(graph) == before);
    CHECK(thiran::GraphVerifier{}.verify(graph));
}

void constantFoldFullPipeline()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* w = graph.createNode("W", Operation::Input);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* zero = graph.createNode("Zero", Operation::Constant);
    Node* matmul = graph.createNode("MatMul", Operation::MatMul);
    Node* first = graph.createNode("FirstRelu", Operation::ReLU);
    Node* second = graph.createNode("SecondRelu", Operation::ReLU);
    Node* folded = graph.createNode("Folded", Operation::Add);
    Node* subtract = graph.createNode("Subtract", Operation::Subtract);
    Node* output = graph.createNode("Output", Operation::Output);
    zero->constantValue = 0.0f;
    graph.connect(x, matmul);
    graph.connect(w, matmul);
    graph.connect(matmul, first);
    graph.connect(first, second);
    graph.connect(second, folded);
    graph.connect(zero, folded);
    graph.connect(folded, subtract);
    graph.connect(y, subtract);
    graph.connect(subtract, output);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    thiran::CanonicalizationPass{}.run(graph);
    thiran::ConstantFold{}.run(graph);
    CHECK(subtract->inputs.size() == 2 && subtract->inputs[0] == second && subtract->inputs[1] == y);
    rewrite(graph);
    CHECK(subtract->inputs.size() == 2 && subtract->inputs[0] == first && subtract->inputs[1] == y);
    CHECK(thiran::FusionRule{}.run(graph));
    thiran::DCEPass{}.run(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(matmul->op == Operation::FusedMatMulRelu);
    CHECK(subtract->inputs.size() == 2 && subtract->inputs[0] == matmul && subtract->inputs[1] == y);
    CHECK(output->inputs.size() == 1 && output->inputs[0] == subtract);
    CHECK(!graph.containsNode(first) && !graph.containsNode(second) &&
          !graph.containsNode(folded) && !graph.containsNode(zero));
    CHECK(signature(graph) == "X:0();W:1();Y:2();MatMul:4(X,W,);Subtract:8(MatMul,Y,);Output:9(Subtract,); edges:X>MatMul;W>MatMul;MatMul>Subtract;Y>Subtract;Subtract>Output;");
}

void pipelineAfterRewrite()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* w = graph.createNode("W", Operation::Input);
    Node* m = graph.createNode("M", Operation::MatMul);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* o = graph.createNode("O", Operation::Output);
    Node* dead = graph.createNode("Dead", Operation::ReLU);
    graph.connect(x, m);
    graph.connect(w, m);
    graph.connect(m, a);
    graph.connect(a, b);
    graph.connect(b, o);
    graph.connect(x, dead);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    thiran::CanonicalizationPass{}.run(graph);
    thiran::ConstantFold{}.run(graph);
    rewrite(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(thiran::FusionRule{}.run(graph));
    thiran::DCEPass{}.run(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(m->op == Operation::FusedMatMulRelu);
    CHECK(o->inputs.size() == 1 && o->inputs[0] == m);
    CHECK(!graph.containsNode(a) && !graph.containsNode(b) && !graph.containsNode(dead));
    CHECK(signature(graph) == "X:0();W:1();M:2(X,W,);O:5(M,); edges:X>M;W>M;M>O;");
}

void multipleFusionOpportunities()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* w = graph.createNode("W", Operation::Input);
    Node* m = graph.createNode("M", Operation::MatMul);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* o1 = graph.createNode("O1", Operation::Output);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* z = graph.createNode("Z", Operation::Input);
    Node* n = graph.createNode("N", Operation::MatMul);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* o2 = graph.createNode("O2", Operation::Output);
    graph.connect(x, m);
    graph.connect(w, m);
    graph.connect(m, a);
    graph.connect(a, o1);
    graph.connect(y, n);
    graph.connect(z, n);
    graph.connect(n, b);
    graph.connect(b, o2);
    CHECK(thiran::FusionRule{}.run(graph));
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(graph.nodes.size() == 8);
    CHECK(m->op == Operation::FusedMatMulRelu && n->op == Operation::FusedMatMulRelu);
    CHECK(o1->inputs[0] == m && o2->inputs[0] == n);
    CHECK(!graph.containsNode(a) && !graph.containsNode(b));
    const auto after = signature(graph);
    CHECK(!thiran::FusionRule{}.run(graph));
    CHECK(signature(graph) == after);
}

void pipelinePreservesSubtractionOrder()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* w = graph.createNode("W", Operation::Input);
    Node* y = graph.createNode("Y", Operation::Input);
    Node* m = graph.createNode("M", Operation::MatMul);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* b = graph.createNode("B", Operation::ReLU);
    Node* s = graph.createNode("S", Operation::Subtract);
    Node* o = graph.createNode("O", Operation::Output);
    graph.connect(x, m);
    graph.connect(w, m);
    graph.connect(m, a);
    graph.connect(a, b);
    graph.connect(b, s);
    graph.connect(y, s);
    graph.connect(s, o);
    thiran::CanonicalizationPass{}.run(graph);
    thiran::ConstantFold{}.run(graph);
    rewrite(graph);
    CHECK(s->inputs.size() == 2 && s->inputs[0] == a && s->inputs[1] == y);
    CHECK(thiran::FusionRule{}.run(graph));
    thiran::DCEPass{}.run(graph);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    CHECK(m->op == Operation::FusedMatMulRelu);
    CHECK(s->inputs.size() == 2 && s->inputs[0] == m && s->inputs[1] == y);
    CHECK(o->inputs[0] == s);
    CHECK(!graph.containsNode(a) && !graph.containsNode(b));
}

void verifierDetectsStaleAdjacency()
{
    Graph graph;
    Node* x = graph.createNode("X", Operation::Input);
    Node* a = graph.createNode("A", Operation::ReLU);
    Node* o = graph.createNode("O", Operation::Output);
    graph.connect(x, a);
    graph.connect(a, o);
    CHECK(thiran::GraphVerifier{}.verify(graph));
    x->outputs.push_back(o);
    CHECK(!thiran::GraphVerifier{}.verify(graph));
    x->outputs.pop_back();
    graph.edges.pop_back();
    CHECK(!thiran::GraphVerifier{}.verify(graph));
}

}

int main()
{
    doubleRelu();
    tripleRelu();
    independentRewrites();
    noMatch();
    createdNodesJoinNextRound();
    preserveOperandOrder();
    collisionIsNotRewritten();
    constantFoldSlotZero();
    constantFoldSlotOne();
    constantFoldMultipleConsumers();
    constantFoldCollisionRefusal();
    constantFoldFullPipeline();
    pipelineAfterRewrite();
    multipleFusionOpportunities();
    pipelinePreservesSubtractionOrder();
    verifierDetectsStaleAdjacency();
    if(failures != 0)
    {
        return 1;
    }
    std::cout << "Rewrite regression tests passed\n";
    return 0;
}
