#include "parser/Parser.hpp"

#include <sstream>
#include <unordered_map>

namespace thiran
{

Parser::Parser(
const std::vector<Token>& tokens
)
{
    this->tokens = tokens;
    position = 0;
}

Token Parser::currentToken()
{
    if(position >= tokens.size())
    {
        return Token(
            TokenType::EndOfFile,
            ""
        );
    }

    return tokens[position];
}

void Parser::advance()
{
    if(position < tokens.size())
    {
        position++;
    }
}

bool Parser::match(TokenType type)
{
    if(currentToken().type == type)
    {
        advance();
        return true;
    }

    return false;
}

void Parser::parseStatement(Graph& graph)
{
    if(currentToken().type != TokenType::Identifier)
    {
        errors.push_back("Expected a node name");
        advance();
        return;
    }

    std::string variableName = currentToken().text;

    advance();

    if(!match(TokenType::Equals))
    {
        errors.push_back("Expected '=' after " + variableName);
        return;
    }

    if(currentToken().type != TokenType::Identifier)
    {
        errors.push_back("Expected an operation after " + variableName + " =");
        return;
    }

    std::string operationName = currentToken().text;

    advance();

    Operation operation = Operation::Unknown;

    static const std::unordered_map<std::string, Operation> operations = {
        {"Input", Operation::Input}, {"Output", Operation::Output},
        {"Constant", Operation::Constant}, {"Add", Operation::Add},
        {"Subtract", Operation::Subtract}, {"Multiply", Operation::Multiply},
        {"Divide", Operation::Divide}, {"MatMul", Operation::MatMul},
        {"Conv2D", Operation::Conv2D}, {"ReLU", Operation::ReLU},
        {"Sigmoid", Operation::Sigmoid}, {"Tanh", Operation::Tanh},
        {"Softmax", Operation::Softmax}, {"MaxPool", Operation::MaxPool},
        {"AvgPool", Operation::AvgPool}, {"Reshape", Operation::Reshape},
        {"Transpose", Operation::Transpose}, {"Transfer", Operation::Transfer}
    };
    if(auto it = operations.find(operationName); it != operations.end())
        operation = it->second;
    else
        errors.push_back("Unknown operation '" + operationName + "'");

    Node* node = graph.createNode(
        variableName,
        operation
    );

    std::vector<std::string> arguments;
    if(match(TokenType::LeftParen))
    {
        while(currentToken().type != TokenType::RightParen &&
              currentToken().type != TokenType::EndOfFile)
        {
            if(currentToken().type == TokenType::Identifier ||
               currentToken().type == TokenType::Number ||
               currentToken().type == TokenType::Minus)
            {
                std::string argument = currentToken().text;
                if(currentToken().type == TokenType::Minus)
                {
                    advance();
                    if(currentToken().type != TokenType::Number)
                    {
                        errors.push_back("Expected a number after '-'");
                        continue;
                    }
                    argument += currentToken().text;
                }
                arguments.push_back(argument);
            }

            advance();
        }

        match(TokenType::RightParen);
    }

    if(operation == Operation::Constant)
    {
        node->isConstant = true;
        if(arguments.empty())
            errors.push_back("Constant '" + variableName + "' requires a numeric value");
        else
        {
            try { node->constantValue = std::stof(arguments[0]); }
            catch(...) { errors.push_back("Constant '" + variableName + "' requires a numeric value"); }
        }
    }

    std::vector<int> dimensions;
    for(size_t index = 0; index < arguments.size(); ++index)
    {
        const auto& argument = arguments[index];
        Node* input = graph.findNode(argument);
        if(input && operation != Operation::Input && operation != Operation::Constant)
        {
            graph.connect(input, node);
            continue;
        }

        try
        {
            const bool isShapeArgument = operation == Operation::Input ||
                                         operation == Operation::Reshape ||
                                         (operation == Operation::Constant && index > 0);
            if(isShapeArgument)
            {
                dimensions.push_back(std::stoi(argument));
            }
        }
        catch(...)
        {
            if(operation == Operation::Transfer &&
               (argument == "CPU" || argument == "GPU" || argument == "TPU"))
                node->attributes["target"] = argument;
            else if(operation != Operation::Input && operation != Operation::Constant)
                errors.push_back("Unknown input '" + argument + "' for " + variableName);
        }
    }

    if(operation == Operation::Input || operation == Operation::Reshape || operation == Operation::Constant)
    {
        if(!dimensions.empty())
        {
            node->shape = dimensions;
            std::ostringstream shape;
            for(size_t i = 0; i < dimensions.size(); ++i)
            {
                if(i) shape << ',';
                shape << dimensions[i];
            }
            node->attributes["shape"] = shape.str();
        }
    }
}

Graph Parser::parse()
{
    Graph graph;

    while(currentToken().type != TokenType::EndOfFile)
    {
        parseStatement(graph);
    }

    return graph;
}

bool Parser::hasErrors() const
{
    return !errors.empty();
}

const std::vector<std::string>& Parser::diagnostics() const
{
    return errors;
}

}
