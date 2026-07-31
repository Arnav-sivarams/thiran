#pragma once

#include <string>
#include <vector>

#include "frontend/SourceLocation.hpp"

namespace thiran::frontend
{
struct ValueReference final
{
    std::string alias;
    std::string name;
    SourceSpan span;
    bool qualified() const noexcept { return !alias.empty(); }
};

struct Argument final
{
    std::string spelling;
    bool numeric = false;
    ValueReference reference;
    SourceSpan span;
};

struct AssignmentStmt final
{
    std::string name;
    std::string operation;
    std::vector<Argument> arguments;
    bool exported = false;
    SourceSpan span;
};

struct FunctionParameter final
{
    std::string name;
    SourceSpan span;
};

struct ReturnStmt final
{
    std::string name;
    SourceSpan span;
    SourceSpan valueSpan;
};

struct FunctionDecl final
{
    std::string name;
    std::vector<FunctionParameter> parameters;
    std::vector<AssignmentStmt> locals;
    std::vector<ReturnStmt> returns;
    bool exported = false;
    SourceSpan fnSpan;
    SourceSpan nameSpan;
    SourceSpan bodySpan;
    SourceSpan span;
};

struct ImportDecl final
{
    std::string path;
    std::string alias;
    SourceSpan span;
};

struct ModuleAst final
{
    SourceId source;
    std::vector<ImportDecl> imports;
    std::vector<FunctionDecl> functions;
    std::vector<AssignmentStmt> assignments;
};
}
