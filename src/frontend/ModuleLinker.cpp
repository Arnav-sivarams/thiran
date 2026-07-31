#include "frontend/ModuleLinker.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

#include "frontend/BuiltinOperations.hpp"
#include "frontend/CallGraph.hpp"
#include "frontend/FunctionInliner.hpp"
#include "frontend/Parser.hpp"
#include "frontend/SemanticAnalyzer.hpp"

namespace thiran::frontend
{
namespace
{
struct ResolvedModule
{
    ModuleAst ast;
    std::vector<ResolvedImport> imports;
};

struct FunctionSymbol
{
    SourceId module;
    std::size_t declaration;
    std::size_t ordinal;
    const FunctionDecl* ast;
};

struct ResolvedCall
{
    const FunctionSymbol* target;
    SourceSpan span;
};

bool reserved(const std::string& name)
{
    return name.rfind("__thiran_module_", 0) == 0 || name.rfind("__thiran_call_", 0) == 0;
}

class Compiler
{
public:
    FrontendResult compile(const std::filesystem::path& path)
    {
        std::string error;
        const auto entry = sources_.loadEntry(path, error);
        if(!entry) { add(error); return std::move(result_); }
        entry_ = *entry; visit(*entry, {}, {});
        if(!result_.diagnostics.empty()) return std::move(result_);
        collectAndValidate();
        if(!result_.diagnostics.empty()) return std::move(result_);
        buildCallGraph();
        if(!result_.diagnostics.empty()) return std::move(result_);
        LinkedProgram linked{entry_, {}};
        for(const auto id : order_) linked.modules.push_back({id, modules_.at(id).ast, modules_.at(id).imports});
        result_.program = std::move(linked);
        lower();
        return std::move(result_);
    }

private:
    std::string where(const SourceSpan& span) const
    {
        const auto* file = sources_.source(span.begin.source);
        return (file ? file->displayPath : "<invalid>") + ":" + std::to_string(span.begin.line) + ":" + std::to_string(span.begin.column);
    }
    std::string diagnostic(const SourceSpan& span, const std::string& message) const
    { return where(span) + ": error: " + message; }
    void add(std::string message)
    {
        if(result_.diagnostics.size() < FunctionLimits::diagnostics) result_.diagnostics.push_back(std::move(message));
        else if(result_.diagnostics.size() == FunctionLimits::diagnostics)
            result_.diagnostics.push_back("<frontend>:1:1: error: diagnostic limit of 100 exceeded");
    }
    void error(const SourceSpan& span, const std::string& message) { add(diagnostic(span, message)); }

    void visit(SourceId id, std::vector<SourceId> stack, std::vector<SourceSpan> importStack)
    {
        if(stack.size() >= 256)
        {
            const auto* file = sources_.source(id);
            add((file ? file->displayPath : "<invalid>") + ":1:1: error: import depth exceeds 256 modules"); return;
        }
        if(done_.count(id)) return;
        if(active_.count(id)) return;
        active_.insert(id); stack.push_back(id);
        std::vector<std::string> parseDiagnostics;
        auto ast = Parser::parse(sources_, id, parseDiagnostics);
        for(auto& item : parseDiagnostics) add(std::move(item));
        if(!ast) { active_.erase(id); return; }
        ResolvedModule module{std::move(*ast), {}};
        std::map<std::string, SourceId> aliases; std::map<SourceId, std::string> importedSources;
        for(const auto& import : module.ast.imports)
        {
            std::string resolveError; const auto target = sources_.resolveImport(id, import.path, resolveError);
            if(!target) { error(import.span, resolveError); continue; }
            if(*target == id) { error(import.span, "module imports itself"); continue; }
            if(active_.count(*target))
            {
                std::string chain;
                for(const auto source : stack) { if(!chain.empty()) chain += " -> "; chain += sources_.source(source)->displayPath; }
                chain += " -> " + sources_.source(*target)->displayPath;
                std::string cycle = diagnostic(import.span, "import cycle: " + chain);
                for(const auto& edge : importStack) cycle += "\n  imported from " + where(edge);
                cycle += "\n  imported from " + where(import.span); add(std::move(cycle)); continue;
            }
            const auto existing = aliases.find(import.alias);
            if(existing != aliases.end()) { if(existing->second != *target) error(import.span, "duplicate import alias '" + import.alias + "'"); continue; }
            const auto canonical = importedSources.find(*target);
            if(canonical != importedSources.end()) { error(import.span, "module already imported as '" + canonical->second + "'"); continue; }
            aliases.emplace(import.alias, *target); importedSources.emplace(*target, import.alias);
            module.imports.push_back({import.alias, *target, import.span}); auto child = importStack; child.push_back(import.span); visit(*target, stack, std::move(child));
        }
        active_.erase(id);
        if(result_.diagnostics.empty() || modules_.find(id) == modules_.end())
        { modules_.emplace(id, std::move(module)); order_.push_back(id); done_.insert(id); }
    }

    const FunctionSymbol* resolveFunction(SourceId owner, const std::string& spelling, const SourceSpan& span, bool report)
    {
        const auto dot = spelling.find('.'); SourceId targetModule = owner; std::string name = spelling;
        if(dot != std::string::npos)
        {
            const std::string alias = spelling.substr(0, dot); name = spelling.substr(dot + 1);
            const auto& imports = modules_.at(owner).imports;
            const auto found = std::find_if(imports.begin(), imports.end(), [&](const ResolvedImport& item){ return item.alias == alias; });
            if(found == imports.end()) { if(report) error(span, "unresolved import alias '" + alias + "'"); return nullptr; }
            targetModule = found->target;
        }
        const auto found = functionsByName_.find({targetModule, name});
        if(found == functionsByName_.end())
        {
            if(report) error(span, dot == std::string::npos ? "unresolved function '" + name + "'" : "unresolved imported function '" + spelling + "'");
            return nullptr;
        }
        if(dot != std::string::npos && !found->second->ast->exported)
        { if(report) error(span, "function '" + spelling + "' is not exported"); return nullptr; }
        return found->second;
    }

    void collectAndValidate()
    {
        std::size_t ordinal = 0;
        for(const auto id : order_)
        {
            const auto& module = modules_.at(id); std::set<std::string> aliases;
            for(const auto& item : module.imports) aliases.insert(item.alias);
            if(module.ast.functions.size() > FunctionLimits::functionsPerModule)
                error(module.ast.functions[FunctionLimits::functionsPerModule].span, "module exceeds 4096 functions");
            std::set<std::string> names = aliases;
            for(std::size_t index = 0; index < module.ast.functions.size(); ++index)
            {
                const auto& function = module.ast.functions[index];
                if(!names.insert(function.name).second)
                {
                    if(aliases.count(function.name)) error(function.nameSpan, "function '" + function.name + "' collides with import alias");
                    else error(function.nameSpan, "duplicate function '" + function.name + "'");
                }
                if(isBuiltinOperation(function.name)) error(function.nameSpan, "function '" + function.name + "' collides with built-in operation");
                auto symbol = FunctionSymbol{id, index, ordinal++, &function}; symbols_.push_back(symbol);
            }
            for(const auto& assignment : module.ast.assignments)
            {
                if(!names.insert(assignment.name).second)
                {
                    if(aliases.count(assignment.name)) error(assignment.span, "declaration collides with import alias '" + assignment.name + "'");
                    else error(assignment.span, "duplicate declaration '" + assignment.name + "'");
                }
            }
        }
        for(auto& symbol : symbols_) functionsByName_[{symbol.module, symbol.ast->name}] = &symbol;

        for(const auto id : order_)
        {
            const auto& module = modules_.at(id); std::set<std::string> exports; std::set<std::string> moduleValues;
            for(const auto& assignment : module.ast.assignments) moduleValues.insert(assignment.name);
            for(const auto& function : module.ast.functions)
            {
                if(function.exported && !exports.insert(function.name).second) error(function.span, "duplicate export '" + function.name + "'");
                if(function.parameters.size() > FunctionLimits::parametersPerFunction) error(function.parameters[FunctionLimits::parametersPerFunction].span, "function '" + function.name + "' exceeds 256 parameters");
                if(function.locals.size() > FunctionLimits::statementsPerFunction) error(function.locals[FunctionLimits::statementsPerFunction].span, "function '" + function.name + "' exceeds 4096 body statements");
                std::set<std::string> values; std::set<std::string> allLocals;
                for(const auto& local : function.locals) allLocals.insert(local.name);
                for(const auto& parameter : function.parameters)
                {
                    if(reserved(parameter.name)) error(parameter.span, "parameter '" + parameter.name + "' uses reserved internal prefix");
                    if(!values.insert(parameter.name).second) error(parameter.span, "duplicate parameter '" + parameter.name + "' in function '" + function.name + "'");
                }
                for(std::size_t localIndex = 0; localIndex < function.locals.size(); ++localIndex)
                {
                    const auto& local = function.locals[localIndex];
                    if(reserved(local.name)) error(local.span, "identifier uses reserved internal call prefix");
                    if(values.count(local.name)) error(local.span, "local '" + local.name + "' shadows parameter or previous local '" + local.name + "'");
                    const auto op = builtinOperation(local.operation);
                    if(op == Operation::Input) error(local.span, "Input is not allowed inside function '" + function.name + "'");
                    else if(op == Operation::Output) error(local.span, "Output is not allowed inside function '" + function.name + "'");
                    const FunctionSymbol* callee = nullptr;
                    if(op == Operation::Unknown) callee = resolveFunction(id, local.operation, local.span, true);
                    if(callee != nullptr && local.arguments.size() != callee->ast->parameters.size())
                        error(local.span, "function '" + local.operation + "' expects " + std::to_string(callee->ast->parameters.size()) + " arguments but received " + std::to_string(local.arguments.size()));
                    for(const auto& argument : local.arguments)
                    {
                        if(callee != nullptr && (argument.numeric || argument.reference.qualified()))
                        { error(argument.span, "function-call arguments must be local tensor value references"); continue; }
                        if(argument.numeric) continue;
                        if(argument.reference.qualified())
                        { error(argument.span, "function '" + function.name + "' cannot capture imported module value '" + argument.spelling + "'; pass it as a parameter"); continue; }
                        if(values.count(argument.reference.name)) continue;
                        if(moduleValues.count(argument.reference.name)) error(argument.span, "function '" + function.name + "' cannot capture module value '" + argument.reference.name + "'; pass it as a parameter");
                        else if(allLocals.count(argument.reference.name)) error(argument.span, "local value '" + argument.reference.name + "' used before its declaration");
                        else if(op != Operation::Transfer || (argument.spelling != "CPU" && argument.spelling != "GPU" && argument.spelling != "TPU")) error(argument.span, "unresolved local tensor value '" + argument.reference.name + "'");
                    }
                    values.insert(local.name);
                }
                if(function.returns.empty()) error(function.span, "function '" + function.name + "' requires exactly one return");
                else
                {
                    if(function.returns.size() > 1) error(function.returns[1].span, "multiple return statements in function '" + function.name + "'");
                    const auto& returned = function.returns.front();
                    if(!values.count(returned.name)) error(returned.valueSpan, "unresolved local tensor value '" + returned.name + "'");
                }
            }
            std::set<std::string> declarations;
            for(const auto& assignment : module.ast.assignments)
            {
                if(!declarations.insert(assignment.name).second) error(assignment.span, "duplicate declaration '" + assignment.name + "'");
                const auto op = builtinOperation(assignment.operation);
                if(op == Operation::Unknown)
                {
                    const auto* callee = resolveFunction(id, assignment.operation, assignment.span, true);
                    if(callee && assignment.arguments.size() != callee->ast->parameters.size()) error(assignment.span, "function '" + assignment.operation + "' expects " + std::to_string(callee->ast->parameters.size()) + " arguments but received " + std::to_string(assignment.arguments.size()));
                    if(callee) for(const auto& argument : assignment.arguments)
                        if(argument.numeric || argument.reference.qualified()) error(argument.span, "function-call arguments must be tensor value references");
                }
                if(id != entry_ && (op == Operation::Input || op == Operation::Output)) error(assignment.span, assignment.operation + " is only allowed in the entry module");
            }
        }
    }

    void buildCallGraph()
    {
        callGraph_.functions.reserve(symbols_.size());
        for(const auto& symbol : symbols_) callGraph_.functions.push_back({symbol.module, symbol.declaration});
        for(const auto& symbol : symbols_)
            for(std::size_t index = 0; index < symbol.ast->locals.size(); ++index)
                if(builtinOperation(symbol.ast->locals[index].operation) == Operation::Unknown)
                    if(const auto* callee = resolveFunction(symbol.module, symbol.ast->locals[index].operation, symbol.ast->locals[index].span, false))
                        callGraph_.edges.push_back({symbol.ordinal, callee->ordinal, index, symbol.ast->locals[index].span});
        std::vector<std::vector<const CallGraphEdge*>> adjacency(symbols_.size());
        for(const auto& edge : callGraph_.edges) adjacency[edge.callerOrdinal].push_back(&edge);
        std::vector<int> color(symbols_.size()); std::vector<std::size_t> stack; std::vector<const CallGraphEdge*> edgeStack;
        const auto dfs = [&](auto&& self, std::size_t current) -> bool
        {
            color[current] = 1; stack.push_back(current);
            for(const auto* edge : adjacency[current])
            {
                if(color[edge->calleeOrdinal] == 0) { edgeStack.push_back(edge); if(self(self, edge->calleeOrdinal)) return true; edgeStack.pop_back(); }
                else if(color[edge->calleeOrdinal] == 1)
                {
                    const auto begin = std::find(stack.begin(), stack.end(), edge->calleeOrdinal); std::string chain;
                    for(auto it = begin; it != stack.end(); ++it) { if(!chain.empty()) chain += " -> "; chain += display(symbols_[*it]); }
                    chain += " -> " + display(symbols_[edge->calleeOrdinal]);
                    std::string message = diagnostic(edge->callSite, "recursive function cycle: " + chain);
                    const std::size_t offset = static_cast<std::size_t>(begin - stack.begin());
                    for(std::size_t i = offset; i < edgeStack.size(); ++i) message += "\n  called from " + where(edgeStack[i]->callSite);
                    message += "\n  called from " + where(edge->callSite); add(std::move(message)); return true;
                }
            }
            stack.pop_back(); color[current] = 2; return false;
        };
        for(std::size_t ordinal = 0; ordinal < symbols_.size(); ++ordinal) if(color[ordinal] == 0 && dfs(dfs, ordinal)) break;
    }

    std::string display(const FunctionSymbol& symbol) const
    { return sources_.source(symbol.module)->displayPath + "." + symbol.ast->name; }

    Node* createOperation(Graph& graph, const AssignmentStmt& statement, const std::string& name,
                          const std::map<std::string, Node*>& values, SourceId module,
                          const std::vector<InlineFrame>& chain, const FunctionDecl* function)
    {
        if(graph.nodes.size() >= FunctionLimits::expandedNodes) { error(statement.span, "function expansion exceeds 1000000 Graph nodes"); return nullptr; }
        const auto op = builtinOperation(statement.operation); Node* node = graph.createNode(name, op); std::vector<int> dimensions;
        for(std::size_t index = 0; index < statement.arguments.size(); ++index)
        {
            const auto& argument = statement.arguments[index];
            if(!argument.numeric)
            {
                const auto found = values.find(argument.spelling);
                if(found != values.end() && op != Operation::Input && op != Operation::Constant) graph.connect(found->second, node);
                else if(op == Operation::Transfer && (argument.spelling == "CPU" || argument.spelling == "GPU" || argument.spelling == "TPU")) node->attributes["target"] = argument.spelling;
                else error(argument.span, "unresolved value '" + argument.spelling + "'");
            }
            else try
            {
                const bool shape = op == Operation::Input || op == Operation::Reshape || (op == Operation::Constant && index > 0);
                if(shape) dimensions.push_back(std::stoi(argument.spelling));
            }
            catch(...) { error(argument.span, "invalid numeric argument '" + argument.spelling + "'"); }
        }
        if(op == Operation::Constant)
        {
            node->isConstant = true;
            if(statement.arguments.empty() || !statement.arguments.front().numeric) error(statement.span, "Constant '" + statement.name + "' requires a numeric value");
            else try { node->constantValue = std::stof(statement.arguments.front().spelling); } catch(...) { error(statement.span, "Constant '" + statement.name + "' requires a numeric value"); }
        }
        if(!dimensions.empty())
        {
            node->shape = dimensions; std::ostringstream shape;
            for(std::size_t i = 0; i < dimensions.size(); ++i) { if(i) shape << ','; shape << dimensions[i]; }
            node->attributes["shape"] = shape.str();
        }
        const auto* file = sources_.source(module); LoweringProvenance provenance;
        provenance.valueId = nextValueId_++; provenance.definingModule = file ? file->displayPath : "<invalid>";
        provenance.localDeclaration = statement.span; provenance.inlineChain = chain;
        if(function) provenance.functionDeclaration = function->span;
        if(!chain.empty()) provenance.immediateCallSite = chain.back().callSite;
        result_.provenance[name] = std::move(provenance); return node;
    }

    Node* expand(Graph& graph, const FunctionSymbol& symbol, const std::vector<Node*>& arguments,
                 const std::string& targetName, std::vector<InlineFrame> chain, const SourceSpan& callSite, std::size_t depth)
    {
        if(depth >= FunctionLimits::expansionDepth) { error(callSite, "function expansion depth exceeds 256: " + display(symbol)); return nullptr; }
        const std::size_t callOrdinal = nextCallOrdinal_++;
        const auto* callerFile = sources_.source(callSite.begin.source);
        chain.push_back({display(symbol), {callerFile ? callerFile->displayPath : "<invalid>", callSite}});
        std::map<std::string, Node*> values;
        for(std::size_t i = 0; i < symbol.ast->parameters.size(); ++i) values[symbol.ast->parameters[i].name] = arguments[i];
        const std::string returned = symbol.ast->returns.front().name;
        for(std::size_t index = 0; index < symbol.ast->locals.size(); ++index)
        {
            const auto& local = symbol.ast->locals[index];
            std::string localName = local.name == returned ? targetName : FunctionInliner::localName(callOrdinal, symbol.ordinal, index, local.name);
            if(localName.size() > FunctionLimits::generatedNameLength) { error(local.span, "generated internal name exceeds 1024 bytes"); return nullptr; }
            Node* value = nullptr;
            if(builtinOperation(local.operation) != Operation::Unknown) value = createOperation(graph, local, localName, values, symbol.module, chain, symbol.ast);
            else
            {
                const auto* callee = resolveFunction(symbol.module, local.operation, local.span, false); std::vector<Node*> callArguments;
                for(const auto& argument : local.arguments) callArguments.push_back(values.at(argument.reference.name));
                value = expand(graph, *callee, callArguments, localName, chain, local.span, depth + 1);
            }
            values[local.name] = value;
        }
        return values.at(returned);
    }

    void lower()
    {
        Graph graph; std::map<std::pair<SourceId, std::string>, Node*> symbols; std::map<SourceId, std::set<std::string>> exports;
        for(const auto id : order_) for(const auto& statement : modules_.at(id).ast.assignments) if(statement.exported) exports[id].insert(statement.name);
        for(const auto id : order_)
        {
            const auto& module = modules_.at(id); std::map<std::string, SourceId> aliases;
            for(const auto& item : module.imports) aliases.emplace(item.alias, item.target);
            std::size_t declaration = 0;
            for(const auto& statement : module.ast.assignments)
            {
                const std::string name = id == entry_ ? statement.name : "__thiran_module_" + std::to_string(id) + "_" + std::to_string(declaration) + "_" + statement.name;
                std::map<std::string, Node*> values;
                for(const auto& item : symbols) if(item.first.first == id) values[item.first.second] = item.second;
                for(const auto& argument : statement.arguments) if(argument.reference.qualified())
                {
                    const auto alias = aliases.find(argument.reference.alias);
                    if(alias == aliases.end()) { error(argument.span, "unresolved import alias '" + argument.reference.alias + "'"); continue; }
                    if(!exports[alias->second].count(argument.reference.name)) { error(argument.span, "symbol '" + argument.spelling + "' is not exported"); continue; }
                    const auto found = symbols.find({alias->second, argument.reference.name}); if(found != symbols.end()) values[argument.spelling] = found->second;
                }
                Node* node = nullptr;
                if(builtinOperation(statement.operation) != Operation::Unknown) node = createOperation(graph, statement, name, values, id, {}, nullptr);
                else
                {
                    const auto* callee = resolveFunction(id, statement.operation, statement.span, false); std::vector<Node*> arguments;
                    for(const auto& argument : statement.arguments)
                    {
                        auto found = values.find(argument.spelling);
                        if(found == values.end()) { error(argument.span, "unresolved value '" + argument.spelling + "'"); arguments.push_back(nullptr); }
                        else arguments.push_back(found->second);
                    }
                    if(callee) node = expand(graph, *callee, arguments, name, {}, statement.span, 0);
                }
                symbols[{id, statement.name}] = node; ++declaration;
            }
        }
        if(result_.diagnostics.empty()) result_.graph = std::move(graph);
    }

    SourceManager sources_; FrontendResult result_; SourceId entry_ = invalidSourceId;
    std::map<SourceId, ResolvedModule> modules_; std::vector<SourceId> order_; std::set<SourceId> active_, done_;
    std::vector<FunctionSymbol> symbols_; std::map<std::pair<SourceId, std::string>, FunctionSymbol*> functionsByName_;
    CallGraph callGraph_; std::size_t nextCallOrdinal_ = 0, nextValueId_ = 0;
};
}

FrontendResult ModuleLinker::compile(const std::filesystem::path& entryPath)
{
    return Compiler{}.compile(entryPath);
}
}
