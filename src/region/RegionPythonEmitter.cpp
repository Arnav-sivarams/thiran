#include "region/RegionPythonEmitter.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "region/RegionPlanVerifier.hpp"

namespace thiran::region
{
namespace
{

using NodeIndexMap = std::unordered_map<const Node*, std::size_t>;

void addDiagnostic(
    RegionPythonEmissionResult& result,
    std::string code,
    std::string message,
    std::optional<RegionId> regionId = std::nullopt,
    std::optional<std::string> nodeName = std::nullopt
)
{
    result.diagnostics.push_back({
        std::move(code),
        std::move(message),
        regionId,
        std::move(nodeName)
    });
}

std::optional<std::size_t> requiredArity(Operation operation)
{
    switch(operation)
    {
        case Operation::Input:
        case Operation::Constant:
            return 0;
        case Operation::Output:
        case Operation::ReLU:
        case Operation::Sigmoid:
        case Operation::Tanh:
        case Operation::Softmax:
        case Operation::MaxPool:
        case Operation::AvgPool:
        case Operation::Reshape:
        case Operation::Transpose:
        case Operation::Transfer:
            return 1;
        case Operation::Add:
        case Operation::Subtract:
        case Operation::Multiply:
        case Operation::Divide:
        case Operation::MatMul:
        case Operation::Conv2D:
        case Operation::FusedMatMulRelu:
            return 2;
        case Operation::Unknown:
            return std::nullopt;
    }
    return std::nullopt;
}

std::string valueName(std::size_t index)
{
    return "v_" + std::to_string(index);
}

std::string pythonString(const std::string& value)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string result = "'";
    for(const unsigned char character : value)
    {
        switch(character)
        {
            case '\\': result += "\\\\"; break;
            case '\'': result += "\\'"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if(character < 0x20 || character == 0x7f)
                {
                    result += "\\x";
                    result += digits[(character >> 4) & 0x0f];
                    result += digits[character & 0x0f];
                }
                else
                {
                    result += static_cast<char>(character);
                }
                break;
        }
    }
    result += "'";
    return result;
}

std::string shapeTuple(const std::vector<int>& shape)
{
    std::ostringstream output;
    output << "(";
    for(std::size_t index = 0; index < shape.size(); ++index)
    {
        if(index != 0)
        {
            output << ", ";
        }
        output << shape[index];
    }
    if(shape.size() == 1)
    {
        output << ",";
    }
    output << ")";
    return output.str();
}

std::string floatLiteral(float value)
{
    char buffer[64];
    const auto conversion = std::to_chars(
        buffer,
        buffer + sizeof(buffer),
        value,
        std::chars_format::general,
        std::numeric_limits<float>::max_digits10
    );
    if(conversion.ec != std::errc())
    {
        return {};
    }
    std::string result(buffer, conversion.ptr);
    if(result.find_first_of(".eE") == std::string::npos)
    {
        result += ".0";
    }
    return result;
}

bool invalidShape(
    const std::vector<int>& shape,
    bool allowDynamic
)
{
    std::size_t dynamicDimensions = 0;
    for(const int dimension : shape)
    {
        if(dimension == -1)
        {
            ++dynamicDimensions;
        }
        else if(dimension <= 0)
        {
            return true;
        }
    }
    return (!allowDynamic && dynamicDimensions != 0) ||
           dynamicDimensions > 1;
}

std::optional<std::vector<const Region*>> regionOrder(
    const RegionGraph& graph
)
{
    std::unordered_map<RegionId, const Region*> regions;
    std::unordered_map<RegionId, std::size_t> indegrees;
    std::unordered_map<RegionId, std::vector<RegionId>> consumers;
    for(const auto& holder : graph.regions())
    {
        if(holder == nullptr || regions.contains(holder->id()))
        {
            return std::nullopt;
        }
        regions.emplace(holder->id(), holder.get());
        indegrees.emplace(holder->id(), 0);
    }
    for(const auto& dependency : graph.dependencies())
    {
        if(!regions.contains(dependency.source) ||
           !regions.contains(dependency.destination))
        {
            return std::nullopt;
        }
        ++indegrees[dependency.destination];
        consumers[dependency.source].push_back(dependency.destination);
    }

    std::priority_queue<
        RegionId,
        std::vector<RegionId>,
        std::greater<RegionId>
    > ready;
    for(const auto& holder : graph.regions())
    {
        if(indegrees[holder->id()] == 0)
        {
            ready.push(holder->id());
        }
    }

    std::vector<const Region*> order;
    std::unordered_set<RegionId> visited;
    while(!ready.empty())
    {
        const RegionId id = ready.top();
        ready.pop();
        if(!visited.insert(id).second)
        {
            return std::nullopt;
        }
        order.push_back(regions.at(id));
        for(const RegionId consumer : consumers[id])
        {
            auto& indegree = indegrees[consumer];
            if(indegree == 0)
            {
                return std::nullopt;
            }
            --indegree;
            if(indegree == 0)
            {
                ready.push(consumer);
            }
        }
    }
    if(order.size() != regions.size())
    {
        return std::nullopt;
    }
    return order;
}

std::string operationSource(
    const Node& node,
    std::size_t nodeIndex,
    const NodeIndexMap& indices
)
{
    const std::string output = valueName(nodeIndex);
    const auto input = [&](std::size_t operand)
    {
        return valueName(indices.at(node.inputs[operand]));
    };
    switch(node.op)
    {
        case Operation::Input:
            return output + " = _require_input(inputs, " +
                   pythonString(node.name) + ")";
        case Operation::Output:
            return output + " = " + input(0);
        case Operation::Constant:
        {
            const std::string value = floatLiteral(node.constantValue);
            if(node.shape.empty())
            {
                return output + " = torch.tensor(" + value +
                       ", dtype=torch.float32, device=runtime_device)";
            }
            return output + " = torch.full(" + shapeTuple(node.shape) +
                   ", " + value +
                   ", dtype=torch.float32, device=runtime_device)";
        }
        case Operation::Add:
            return output + " = " + input(0) + " + " + input(1);
        case Operation::Subtract:
            return output + " = " + input(0) + " - " + input(1);
        case Operation::Multiply:
            return output + " = " + input(0) + " * " + input(1);
        case Operation::Divide:
            return output + " = " + input(0) + " / " + input(1);
        case Operation::MatMul:
            return output + " = torch.matmul(" + input(0) + ", " +
                   input(1) + ")";
        case Operation::Conv2D:
            return output + " = torch.nn.functional.conv2d(" + input(0) +
                   ", " + input(1) + ")";
        case Operation::ReLU:
            return output + " = torch.relu(" + input(0) + ")";
        case Operation::Sigmoid:
            return output + " = torch.sigmoid(" + input(0) + ")";
        case Operation::Tanh:
            return output + " = torch.tanh(" + input(0) + ")";
        case Operation::Softmax:
            return output + " = torch.softmax(" + input(0) + ", dim=-1)";
        case Operation::MaxPool:
            return output +
                   " = torch.nn.functional.max_pool2d(" + input(0) + ", 2)";
        case Operation::AvgPool:
            return output +
                   " = torch.nn.functional.avg_pool2d(" + input(0) + ", 2)";
        case Operation::Reshape:
            return output + " = torch.reshape(" + input(0) + ", " +
                   shapeTuple(node.shape) + ")";
        case Operation::Transpose:
            return output + " = torch.transpose(" + input(0) + ", 0, 1)";
        case Operation::Transfer:
            return output + " = " + input(0) + ".to(runtime_device)";
        case Operation::FusedMatMulRelu:
            return output + " = torch.relu(torch.matmul(" + input(0) +
                   ", " + input(1) + "))";
        case Operation::Unknown:
            break;
    }
    return {};
}

}

bool RegionPythonEmissionResult::succeeded() const noexcept
{
    return artifact.has_value() && diagnostics.empty();
}

RegionPythonEmissionResult RegionPythonEmitter::emit(
    const Graph& graph,
    const RegionPlan& plan
)
{
    RegionPythonEmissionResult result;
    const auto verification = RegionPlanVerifier::verify(graph, plan);
    if(&graph != &plan.sourceGraph() || !verification.valid)
    {
        for(const auto& diagnostic : verification.diagnostics)
        {
            addDiagnostic(
                result,
                "RPE001",
                diagnostic.code + ": " + diagnostic.message,
                diagnostic.regionId,
                diagnostic.nodeName
            );
        }
        if(result.diagnostics.empty())
        {
            addDiagnostic(
                result,
                "RPE001",
                "RP001: RegionPlan belongs to another Graph."
            );
        }
        return result;
    }

    NodeIndexMap indices;
    indices.reserve(graph.nodes.size());
    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        if(graph.nodes[index] != nullptr)
        {
            indices.emplace(graph.nodes[index].get(), index);
        }
    }

    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node& node = *graph.nodes[index];
        if(!requiredArity(node.op).has_value())
        {
            const Region* owner = plan.regionFor(node);
            addDiagnostic(
                result,
                "RPE002",
                "Unsupported Operation value " +
                    std::to_string(static_cast<int>(node.op)) + ".",
                owner == nullptr ? std::nullopt :
                    std::optional<RegionId>(owner->id()),
                node.name
            );
        }
    }
    if(!result.diagnostics.empty()) return result;

    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node& node = *graph.nodes[index];
        const auto arity = requiredArity(node.op);
        if(arity.has_value() && node.inputs.size() != *arity)
        {
            const Region* owner = plan.regionFor(node);
            addDiagnostic(
                result,
                "RPE003",
                "Expected " + std::to_string(*arity) +
                    " inputs but found " +
                    std::to_string(node.inputs.size()) + ".",
                owner == nullptr ? std::nullopt :
                    std::optional<RegionId>(owner->id()),
                node.name
            );
        }
    }
    if(!result.diagnostics.empty()) return result;

    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        const Node& node = *graph.nodes[index];
        const Region* owner = plan.regionFor(node);
        bool invalid = !indices.contains(&node) || owner == nullptr;
        if(node.op == Operation::Input)
        {
            for(const int dimension : node.shape)
            {
                invalid = invalid || dimension == 0 || dimension < -1;
            }
        }
        else if(node.op == Operation::Constant)
        {
            invalid = invalid || invalidShape(node.shape, false) ||
                      !std::isfinite(node.constantValue);
        }
        else if(node.op == Operation::Reshape)
        {
            invalid = invalid || invalidShape(node.shape, true);
        }
        if(invalid)
        {
            addDiagnostic(
                result,
                "RPE004",
                "Invalid metadata required for Python emission.",
                owner == nullptr ? std::nullopt :
                    std::optional<RegionId>(owner->id()),
                node.name
            );
        }
    }
    std::unordered_set<std::string> inputNames;
    std::unordered_set<std::string> outputNames;
    for(const auto& holder : graph.nodes)
    {
        const Node& node = *holder;
        auto& names = node.op == Operation::Input ? inputNames : outputNames;
        if((node.op == Operation::Input || node.op == Operation::Output) &&
           !names.insert(node.name).second)
        {
            const Region* owner = plan.regionFor(node);
            addDiagnostic(
                result,
                "RPE004",
                node.op == Operation::Input ?
                    "Duplicate Graph Input name." :
                    "Duplicate semantic Output name.",
                owner == nullptr ? std::nullopt :
                    std::optional<RegionId>(owner->id()),
                node.name
            );
        }
    }
    if(!result.diagnostics.empty()) return result;

    const auto order = regionOrder(plan.regionGraph());
    if(!order.has_value())
    {
        addDiagnostic(
            result,
            "RPE005",
            "Region dependencies do not form a valid deterministic topology."
        );
        return result;
    }

    std::unordered_set<const Node*> available;
    for(const Region* region : *order)
    {
        for(const Node* input : region->inputs())
        {
            if(input == nullptr || !available.contains(input))
            {
                addDiagnostic(
                    result,
                    "RPE006",
                    "Region input value is unavailable before execution.",
                    region->id(),
                    input == nullptr ? std::nullopt :
                        std::optional<std::string>(input->name)
                );
            }
        }
        for(const Node* output : region->outputs())
        {
            if(output != nullptr)
            {
                available.insert(output);
            }
        }
    }
    if(!result.diagnostics.empty()) return result;

    for(const Region* region : *order)
    {
        std::unordered_set<const Node*> local(
            region->inputs().begin(),
            region->inputs().end()
        );
        for(const Node* node : region->nodes())
        {
            local.insert(node);
        }
        for(const Node* output : region->outputs())
        {
            if(output == nullptr || !local.contains(output))
            {
                addDiagnostic(
                    result,
                    "RPE007",
                    "Region output has no produced or aliased value.",
                    region->id(),
                    output == nullptr ? std::nullopt :
                        std::optional<std::string>(output->name)
                );
            }
        }
    }
    if(!result.diagnostics.empty()) return result;

    for(const Region* region : *order)
    {
        std::unordered_set<const Node*> local(
            region->inputs().begin(),
            region->inputs().end()
        );
        for(const Node* node : region->nodes())
        {
            if(node->op != Operation::Input &&
               node->op != Operation::Constant)
            {
                for(const Node* input : node->inputs)
                {
                    if(input == nullptr || !local.contains(input))
                    {
                        addDiagnostic(
                            result,
                            "RPE008",
                            "Operation input is unavailable inside Region.",
                            region->id(),
                            node->name
                        );
                        break;
                    }
                }
            }
            local.insert(node);
        }
    }
    if(!result.diagnostics.empty()) return result;

    std::ostringstream source;
    source << "# Generated by Thiran RegionPythonEmitter\n"
           << "import os\n"
           << "import pathlib\n"
           << "import sys\n"
           << "import tempfile\n"
           << "import torch\n"
           << "import torch.nn.functional\n\n"
           << "_THIRAN_INPUT_SPEC = (\n";
    for(const auto& holder : graph.nodes)
    {
        if(holder->op != Operation::Input) continue;
        source << "    (" << pythonString(holder->name) << ", ";
        if(holder->shape.empty()) source << "None";
        else source << shapeTuple(holder->shape);
        source << "),\n";
    }
    source << ")\n\n"
           << "_THIRAN_OUTPUT_NAMES = (\n";
    for(const auto& holder : graph.nodes)
    {
        if(holder->op == Operation::Output)
            source << "    (" << pythonString(holder->name) << "),\n";
    }
    source << ")\n\n"
           << "_THIRAN_USAGE = \"Usage:\\n\" \\\n"
           << "    \"  ThiranRegionExecutor --describe\\n\" \\\n"
           << "    \"  ThiranRegionExecutor --run <input-bundle.pt> <output-bundle.pt>\\n\" \\\n"
           << "    \"  ThiranRegionExecutor --help\\n\"\n\n"
           << "def thiran_input_spec():\n"
           << "    return _THIRAN_INPUT_SPEC\n\n"
           << "def thiran_output_names():\n"
           << "    return _THIRAN_OUTPUT_NAMES\n\n"
           << "def _require_input(inputs, name):\n"
           << "    if name not in inputs:\n"
           << "        raise KeyError(f\"Missing graph input: {name}\")\n"
           << "    value = inputs[name]\n"
           << "    if not isinstance(value, torch.Tensor):\n"
           << "        raise TypeError(f\"Graph input is not a torch.Tensor: {name}\")\n"
           << "    return value\n\n"
           << "def _validate_inputs(inputs, strict):\n"
           << "    if not isinstance(inputs, dict):\n"
           << "        raise ValueError(\"input bundle is not a dictionary\")\n"
           << "    for key in inputs:\n"
           << "        if not isinstance(key, str):\n"
           << "            raise ValueError(\"input bundle contains a non-string key\")\n"
           << "    for name, value in inputs.items():\n"
           << "        if not isinstance(value, torch.Tensor):\n"
           << "            raise ValueError(f\"input {name!r} is not a torch.Tensor\")\n"
           << "    required_names = {name for name, _ in _THIRAN_INPUT_SPEC}\n"
           << "    devices = []\n"
           << "    for name, shape in _THIRAN_INPUT_SPEC:\n"
           << "        if name not in inputs:\n"
           << "            raise ValueError(f\"missing required input {name!r}\")\n"
           << "        value = inputs[name]\n"
           << "        if shape is not None:\n"
           << "            if value.dim() != len(shape):\n"
           << "                raise ValueError(f\"input {name!r} expected rank {len(shape)} but found {value.dim()}\")\n"
           << "            for index, expected in enumerate(shape):\n"
           << "                actual = value.shape[index]\n"
           << "                if expected != -1 and actual != expected:\n"
           << "                    raise ValueError(f\"input {name!r} dimension {index} expected {expected} but found {actual}\")\n"
           << "        devices.append(value.device)\n"
           << "    if devices and any(device != devices[0] for device in devices[1:]):\n"
           << "        raise ValueError(\"graph inputs must use one device\")\n"
           << "    if strict:\n"
           << "        unexpected = sorted(name for name in inputs if name not in required_names)\n"
           << "        if unexpected:\n"
           << "            raise ValueError(f\"unexpected input {unexpected[0]!r}\")\n\n"
           << "def _load_input_bundle(input_path):\n"
           << "    return torch.load(input_path, map_location='cpu', weights_only=True)\n\n"
           << "def _write_output_bundle(output_path, outputs):\n"
           << "    if not isinstance(outputs, dict):\n"
           << "        raise TypeError(\"Region outputs are not a dictionary\")\n"
           << "    serialized = {}\n"
           << "    for key, value in outputs.items():\n"
           << "        if not isinstance(key, str) or not isinstance(value, torch.Tensor):\n"
           << "            raise TypeError(\"Region outputs are not string-to-Tensor values\")\n"
           << "        serialized[key] = value.detach().cpu()\n"
           << "    temporary_path = None\n"
           << "    try:\n"
           << "        with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as temporary:\n"
           << "            temporary_path = pathlib.Path(temporary.name)\n"
           << "        torch.save(serialized, temporary_path)\n"
           << "        os.replace(temporary_path, output_path)\n"
           << "        temporary_path = None\n"
           << "    finally:\n"
           << "        if temporary_path is not None and temporary_path.exists():\n"
           << "            temporary_path.unlink()\n\n"
           << "def _shape_description(shape):\n"
           << "    if shape is None:\n"
           << "        return '<unknown-rank>'\n"
           << "    values = [('?' if value == -1 else str(value)) for value in shape]\n"
           << "    if len(values) == 1:\n"
           << "        return f\"({values[0]},)\"\n"
           << "    return '(' + ', '.join(values) + ')'\n\n"
           << "def _print_description():\n"
           << "    print(\"Thiran Region Executor\")\n"
           << "    print(\"Inputs:\")\n"
           << "    if not _THIRAN_INPUT_SPEC:\n"
           << "        print(\"  <none>\")\n"
           << "    for name, shape in _THIRAN_INPUT_SPEC:\n"
           << "        print(f\"  {name!r} shape={_shape_description(shape)}\")\n"
           << "    print(\"Outputs:\")\n"
           << "    if not _THIRAN_OUTPUT_NAMES:\n"
           << "        print(\"  <none>\")\n"
           << "    for name in _THIRAN_OUTPUT_NAMES:\n"
           << "        print(f\"  {name!r}\")\n\n";

    for(const auto& holder : plan.regionGraph().regions())
    {
        const Region& region = *holder;
        source << "# Region " << region.id() << " strategy: "
               << toString(region.strategy()) << "\n"
               << "def _region_" << region.id()
               << "(inputs, runtime_device";
        for(const Node* input : region.inputs())
        {
            source << ", " << valueName(indices.at(input));
        }
        source << "):\n";
        for(const Node* node : region.nodes())
        {
            source << "    "
                   << operationSource(*node, indices.at(node), indices)
                   << "\n";
        }
        source << "    return (";
        for(std::size_t index = 0; index < region.outputs().size(); ++index)
        {
            if(index != 0) source << ", ";
            source << valueName(indices.at(region.outputs()[index]));
        }
        if(region.outputs().size() == 1) source << ",";
        source << ")\n\n";
    }

    source << "def run_region_plan(inputs):\n"
           << "    if not isinstance(inputs, dict):\n"
           << "        raise TypeError(\"inputs must be a dictionary\")\n"
           << "    _validate_inputs(inputs, False)\n";
    std::vector<std::size_t> inputIndices;
    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        if(graph.nodes[index]->op == Operation::Input)
        {
            inputIndices.push_back(index);
            source << "    _input_" << index << " = _require_input(inputs, "
                   << pythonString(graph.nodes[index]->name) << ")\n";
        }
    }
    if(inputIndices.empty())
    {
        source << "    runtime_device = torch.device('cpu')\n";
    }
    else
    {
        source << "    runtime_device = _input_" << inputIndices.front()
               << ".device\n";
    }
    source << "    values = {}\n";
    for(const Region* region : *order)
    {
        source << "    ";
        if(!region->outputs().empty())
        {
            source << "_region_result_" << region->id() << " = ";
        }
        source << "_region_" << region->id()
               << "(inputs, runtime_device";
        for(const Node* input : region->inputs())
        {
            source << ", values[" << indices.at(input) << "]";
        }
        source << ")\n";
        for(std::size_t index = 0; index < region->outputs().size(); ++index)
        {
            source << "    values[" << indices.at(region->outputs()[index])
                   << "] = _region_result_" << region->id()
                   << "[" << index << "]\n";
        }
    }
    source << "    outputs = {}\n";
    for(std::size_t index = 0; index < graph.nodes.size(); ++index)
    {
        if(graph.nodes[index]->op == Operation::Output)
        {
            source << "    outputs[" << pythonString(graph.nodes[index]->name)
                   << "] = values[" << index << "]\n";
        }
    }
    source << "    return outputs\n\n"
           << "def main(argv=None):\n"
           << "    argv = sys.argv[1:] if argv is None else argv\n"
           << "    if argv == ['--help'] or argv == ['-h']:\n"
           << "        sys.stdout.write(_THIRAN_USAGE)\n"
           << "        return 0\n"
           << "    if argv == ['--describe']:\n"
           << "        _print_description()\n"
           << "        return 0\n"
           << "    if len(argv) != 3 or argv[0] != '--run':\n"
           << "        sys.stderr.write(\"Invalid Region executor command\\n\" + _THIRAN_USAGE)\n"
           << "        return 2\n"
           << "    input_path = pathlib.Path(argv[1]).resolve(strict=False)\n"
           << "    output_path = pathlib.Path(argv[2]).resolve(strict=False)\n"
           << "    if input_path == output_path:\n"
           << "        sys.stderr.write(f\"Refusing to overwrite input tensor bundle: {argv[2]}\\n\")\n"
           << "        return 5\n"
           << "    if not output_path.parent.exists() or not output_path.parent.is_dir():\n"
           << "        sys.stderr.write(f\"Failed to write output tensor bundle: {argv[2]}\\n\")\n"
           << "        return 5\n"
           << "    try:\n"
           << "        inputs = _load_input_bundle(input_path)\n"
           << "    except Exception:\n"
           << "        sys.stderr.write(f\"Failed to load input tensor bundle: {argv[1]}\\n\")\n"
           << "        return 3\n"
           << "    try:\n"
           << "        _validate_inputs(inputs, True)\n"
           << "    except Exception as error:\n"
           << "        sys.stderr.write(f\"Invalid input tensor bundle: {error}\\n\")\n"
           << "        return 3\n"
           << "    try:\n"
           << "        outputs = run_region_plan(inputs)\n"
           << "    except Exception as error:\n"
           << "        sys.stderr.write(f\"Region execution failed: {type(error).__name__}: {error}\\n\")\n"
           << "        return 4\n"
           << "    try:\n"
           << "        _write_output_bundle(output_path, outputs)\n"
           << "    except Exception:\n"
           << "        sys.stderr.write(f\"Failed to write output tensor bundle: {argv[2]}\\n\")\n"
           << "        return 5\n"
           << "    sys.stdout.write(f\"Wrote Region outputs: {argv[2]}\\n\")\n"
           << "    return 0\n\n"
           << "if __name__ == '__main__':\n"
           << "    raise SystemExit(main())\n";
    result.artifact = RegionPythonArtifact{source.str()};
    return result;
}

}
