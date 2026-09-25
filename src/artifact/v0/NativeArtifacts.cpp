#include "artifact/v0/NativeArtifacts.hpp"

#include "backend/v0/NativeCpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace thiran::v0::artifact {
namespace {

constexpr std::array<char, 8> magic{'T','H','I','R','A','N','1','6'};
constexpr std::uint64_t maxStringBytes = 1U << 20;
constexpr std::uint64_t maxRegionBytes = 64U << 20;
constexpr std::uint64_t maxPayloadBytes = 512U << 20;
constexpr std::uint64_t maxItems = 1U << 20;

ArtifactError error(ArtifactErrorCategory category, std::string code, std::string message) {
    return {category, std::move(code), std::move(message)};
}

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(std::string_view prefix = "thiran-artifact-") {
        auto pattern = (std::filesystem::temp_directory_path() /
                        (std::string(prefix) + "XXXXXX")).string();
        std::vector<char> bytes(pattern.begin(), pattern.end());
        bytes.push_back('\0');
        if (auto* created = ::mkdtemp(bytes.data())) path_ = created;
    }
    ~TemporaryDirectory() {
        if (!path_.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
        }
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    const std::filesystem::path& path() const noexcept { return path_; }
    bool valid() const noexcept { return !path_.empty(); }
private:
    std::filesystem::path path_;
};

class Writer {
public:
    void u8(std::uint8_t value) { bytes_.push_back(static_cast<std::byte>(value)); }
    void u32(std::uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8)
            u8(static_cast<std::uint8_t>(value >> shift));
    }
    void u64(std::uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8)
            u8(static_cast<std::uint8_t>(value >> shift));
    }
    void i64(std::int64_t value) { u64(std::bit_cast<std::uint64_t>(value)); }
    void text(std::string_view value) {
        u64(value.size());
        raw(reinterpret_cast<const std::byte*>(value.data()), value.size());
    }
    void raw(const std::byte* data, std::size_t size) {
        if (size != 0) bytes_.insert(bytes_.end(), data, data + size);
    }
    const std::vector<std::byte>& bytes() const noexcept { return bytes_; }
    std::vector<std::byte> take() { return std::move(bytes_); }
private:
    std::vector<std::byte> bytes_;
};

class Reader {
public:
    explicit Reader(const std::vector<std::byte>& bytes) : bytes_(bytes) {}
    std::uint8_t u8() {
        need(1);
        return std::to_integer<std::uint8_t>(bytes_[offset_++]);
    }
    std::uint32_t u32() {
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift != 32; shift += 8)
            value |= static_cast<std::uint32_t>(u8()) << shift;
        return value;
    }
    std::uint64_t u64() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift != 64; shift += 8)
            value |= static_cast<std::uint64_t>(u8()) << shift;
        return value;
    }
    std::int64_t i64() { return std::bit_cast<std::int64_t>(u64()); }
    std::string text() {
        const auto count = u64();
        if (count > maxStringBytes) throw std::runtime_error("ARTIFACT-LENGTH: string exceeds V0 limit");
        need(count);
        std::string value(reinterpret_cast<const char*>(bytes_.data() + offset_),
                          static_cast<std::size_t>(count));
        offset_ += static_cast<std::size_t>(count);
        return value;
    }
    std::vector<std::byte> raw(std::uint64_t count, std::uint64_t limit) {
        if (count > limit) throw std::runtime_error("ARTIFACT-LENGTH: payload exceeds V0 limit");
        need(count);
        std::vector<std::byte> value(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                                     bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += static_cast<std::size_t>(count);
        return value;
    }
    bool done() const noexcept { return offset_ == bytes_.size(); }
private:
    void need(std::uint64_t count) {
        if (count > bytes_.size() || offset_ > bytes_.size() - static_cast<std::size_t>(count))
            throw std::runtime_error("ARTIFACT-TRUNCATED: checked read exceeds artifact size");
    }
    const std::vector<std::byte>& bytes_;
    std::size_t offset_ = 0;
};

std::uint8_t dtypeCode(storage::DType dtype) {
    if (dtype == storage::DType::I64) return 1;
    if (dtype == storage::DType::F32) return 2;
    return 0;
}
storage::DType decodeDtype(std::uint8_t code) {
    if (code == 1) return storage::DType::I64;
    if (code == 2) return storage::DType::F32;
    return storage::DType::Invalid;
}

void writeType(Writer& writer, const semantic::Type& type) {
    writer.u8(static_cast<std::uint8_t>(type.kind));
    writer.u32(type.rank);
    writer.u32(static_cast<std::uint32_t>(type.elements.size()));
    for (const auto& element : type.elements) writeType(writer, element);
}
semantic::Type readType(Reader& reader, unsigned depth = 0) {
    if (depth > 8) throw std::runtime_error("ARTIFACT-REGION: type nesting exceeds V0 limit");
    semantic::Type type;
    type.kind = static_cast<semantic::TypeKind>(reader.u8());
    type.rank = reader.u32();
    const auto count = reader.u32();
    if (count > 16) throw std::runtime_error("ARTIFACT-REGION: type arity exceeds V0 limit");
    for (std::uint32_t index = 0; index < count; ++index)
        type.elements.push_back(readType(reader, depth + 1));
    if (!semantic::validType(type)) throw std::runtime_error("ARTIFACT-REGION: invalid type");
    return type;
}
void writeShape(Writer& writer, const semantic::ShapeFact& shape) {
    writer.u32(static_cast<std::uint32_t>(shape.extents.size()));
    for (const auto& extent : shape.extents) {
        writer.u8(extent.has_value());
        if (extent) writer.i64(*extent);
    }
}
semantic::ShapeFact readShape(Reader& reader) {
    semantic::ShapeFact shape;
    const auto count = reader.u32();
    if (count > 64) throw std::runtime_error("ARTIFACT-REGION: rank exceeds V0 limit");
    for (std::uint32_t index = 0; index < count; ++index) {
        if (reader.u8()) shape.extents.emplace_back(reader.i64());
        else shape.extents.emplace_back(std::nullopt);
    }
    return shape;
}
template<class T>
void writeIds(Writer& writer, const std::vector<T>& values) {
    writer.u32(static_cast<std::uint32_t>(values.size()));
    for (auto value : values) writer.u64(static_cast<std::uint64_t>(value));
}
template<class T>
std::vector<T> readIds(Reader& reader) {
    const auto count = reader.u32();
    if (count > maxItems) throw std::runtime_error("ARTIFACT-REGION: collection exceeds V0 limit");
    std::vector<T> values;
    values.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto value = reader.u64();
        if (value > std::numeric_limits<T>::max())
            throw std::runtime_error("ARTIFACT-REGION: identifier overflow");
        values.push_back(static_cast<T>(value));
    }
    return values;
}

std::vector<std::byte> serializeRegion(const backend::TensorRegion& region) {
    Writer writer;
    writer.u32(region.function);
    writer.text(region.name);
    writer.u32(static_cast<std::uint32_t>(region.nodes.size()));
    for (const auto& node : region.nodes) {
        writer.u32(node.id);
        writer.u8(static_cast<std::uint8_t>(node.op));
        writeType(writer, node.type);
        writeShape(writer, node.shape);
        writeIds(writer, node.dependencies);
        writeIds(writer, node.indices);
        writer.u8(node.integer.has_value());
        if (node.integer) writer.i64(*node.integer);
        writer.u8(node.floating.has_value());
        if (node.floating) writer.u32(std::bit_cast<std::uint32_t>(*node.floating));
        writer.u32(static_cast<std::uint32_t>(node.checks.size()));
        for (const auto& check : node.checks) {
            writer.u8(static_cast<std::uint8_t>(check.kind));
            writeIds(writer, check.operands);
            writer.text(check.failureId);
            writer.u32(check.axis);
            writer.u8(static_cast<std::uint8_t>(check.effect));
        }
        writer.u8(static_cast<std::uint8_t>(node.provenance));
        writer.u32(static_cast<std::uint32_t>(node.resources.size()));
        for (auto value : node.resources) writer.u64(value);
        writer.u32(static_cast<std::uint32_t>(node.viewRoots.size()));
        for (auto value : node.viewRoots) writer.u32(value);
    }
    writeIds(writer, region.inputs);
    writer.u32(region.output);
    writeType(writer, region.outputType);
    return writer.take();
}

backend::TensorRegion deserializeRegion(const std::vector<std::byte>& bytes) {
    Reader reader(bytes);
    backend::TensorRegion region;
    region.function = reader.u32();
    region.name = reader.text();
    const auto nodeCount = reader.u32();
    if (nodeCount == 0 || nodeCount > maxItems)
        throw std::runtime_error("ARTIFACT-REGION: invalid node count");
    region.nodes.reserve(nodeCount);
    for (std::uint32_t index = 0; index < nodeCount; ++index) {
        backend::RegionNode node;
        node.id = reader.u32();
        node.op = static_cast<backend::RegionOp>(reader.u8());
        node.type = readType(reader);
        node.shape = readShape(reader);
        node.dependencies = readIds<semantic::ValueId>(reader);
        node.indices = readIds<semantic::ValueId>(reader);
        if (reader.u8()) node.integer = reader.i64();
        if (reader.u8()) node.floating = std::bit_cast<float>(reader.u32());
        const auto checkCount = reader.u32();
        if (checkCount > maxItems) throw std::runtime_error("ARTIFACT-REGION: invalid check count");
        for (std::uint32_t checkIndex = 0; checkIndex < checkCount; ++checkIndex) {
            semantic::Check check;
            check.kind = static_cast<semantic::CheckKind>(reader.u8());
            check.operands = readIds<semantic::ValueId>(reader);
            check.failureId = reader.text();
            check.axis = reader.u32();
            check.effect = static_cast<semantic::EffectClass>(reader.u8());
            node.checks.push_back(std::move(check));
        }
        node.provenance = static_cast<analysis::ProvenanceKind>(reader.u8());
        const auto resourceCount = reader.u32();
        if (resourceCount > maxItems) throw std::runtime_error("ARTIFACT-REGION: invalid resource count");
        for (std::uint32_t resource = 0; resource < resourceCount; ++resource)
            node.resources.insert(reader.u64());
        const auto rootCount = reader.u32();
        if (rootCount > maxItems) throw std::runtime_error("ARTIFACT-REGION: invalid view-root count");
        for (std::uint32_t root = 0; root < rootCount; ++root) node.viewRoots.insert(reader.u32());
        region.nodes.push_back(std::move(node));
    }
    region.inputs = readIds<semantic::ValueId>(reader);
    region.output = reader.u32();
    region.outputType = readType(reader);
    if (!reader.done()) throw std::runtime_error("ARTIFACT-REGION: trailing region metadata");
    const auto verification = backend::verifyRegion(region);
    if (!verification.ok())
        throw std::runtime_error("ARTIFACT-REGION: " + verification.errors.front());
    return region;
}

void writeArtifactType(Writer& writer, const ArtifactType& type) {
    writer.u8(static_cast<std::uint8_t>(type.kind));
    writer.u8(dtypeCode(type.dtype));
    writer.u32(type.rank);
    writer.u32(static_cast<std::uint32_t>(type.extents.size()));
    for (const auto& extent : type.extents) {
        writer.u8(extent.has_value());
        if (extent) writer.u64(*extent);
    }
}
ArtifactType readArtifactType(Reader& reader) {
    ArtifactType type;
    type.kind = static_cast<ValueKind>(reader.u8());
    type.dtype = decodeDtype(reader.u8());
    type.rank = reader.u32();
    const auto count = reader.u32();
    if (count > 64) throw std::runtime_error("ARTIFACT-SIGNATURE: rank exceeds V0 limit");
    for (std::uint32_t index = 0; index < count; ++index) {
        if (reader.u8()) type.extents.emplace_back(reader.u64());
        else type.extents.emplace_back(std::nullopt);
    }
    return type;
}

std::optional<std::string> validateType(const ArtifactType& type) {
    if (type.dtype != storage::DType::I64 && type.dtype != storage::DType::F32)
        return "unsupported dtype";
    if (type.kind == ValueKind::Scalar) {
        if (type.rank != 0 || !type.extents.empty()) return "scalar has rank/extents";
    } else if (type.kind == ValueKind::Tensor) {
        if ((type.rank != 1 && type.rank != 2) || type.extents.size() != type.rank)
            return "tensor rank/extents mismatch";
    } else return "unknown value kind";
    return {};
}

const backend::RegionNode& node(const backend::TensorRegion& region, semantic::ValueId id) {
    const auto found = std::find_if(region.nodes.begin(), region.nodes.end(),
        [&](const auto& candidate) { return candidate.id == id; });
    if (found == region.nodes.end()) throw std::invalid_argument("entry references missing input");
    return *found;
}

bool compatible(const ArtifactType& artifact, const semantic::Type& semanticType,
                const semantic::ShapeFact& semanticShape) {
    if (artifact != artifactType(semanticType, semanticShape)) {
        const auto base = artifactType(semanticType, semanticShape);
        if (artifact.kind != base.kind || artifact.dtype != base.dtype || artifact.rank != base.rank ||
            artifact.extents.size() != base.extents.size()) return false;
        for (std::size_t index = 0; index < base.extents.size(); ++index)
            if (base.extents[index] && artifact.extents[index] != base.extents[index]) return false;
    }
    return true;
}

std::optional<ArtifactError> validateSignature(const backend::TensorRegion& region,
                                               const EntryPoint& entry,
                                               bool requireConcrete) {
    if (entry.name.empty() || entry.name != region.name)
        return error(ArtifactErrorCategory::Load, "ARTIFACT-ENTRY", "entry name does not match lowered region");
    if (entry.parameters.size() != region.inputs.size())
        return error(ArtifactErrorCategory::Load, "ARTIFACT-SIGNATURE", "parameter count mismatch");
    for (std::size_t index = 0; index < entry.parameters.size(); ++index) {
        if (const auto invalid = validateType(entry.parameters[index]))
            return error(ArtifactErrorCategory::Load, "ARTIFACT-SIGNATURE", *invalid);
        const auto& input = node(region, region.inputs[index]);
        if (!compatible(entry.parameters[index], input.type, input.shape))
            return error(ArtifactErrorCategory::Load, "ARTIFACT-SIGNATURE", "parameter type conflicts with region");
    }
    if (const auto invalid = validateType(entry.result))
        return error(ArtifactErrorCategory::Load, "ARTIFACT-SIGNATURE", *invalid);
    const auto& output = node(region, region.output);
    if (!compatible(entry.result, output.type, output.shape))
        return error(ArtifactErrorCategory::Load, "ARTIFACT-SIGNATURE", "result type conflicts with region");
    if (requireConcrete) {
        auto concrete = [](const ArtifactType& type) {
            return type.kind == ValueKind::Scalar ||
                   std::all_of(type.extents.begin(), type.extents.end(), [](const auto& value) { return value.has_value(); });
        };
        for (const auto& parameter : entry.parameters)
            if (!concrete(parameter))
                return error(ArtifactErrorCategory::Compilation, "ARTIFACT-CPU-SPECIALIZATION",
                             "CPU native ABI requires concrete tensor extents");
        if (!concrete(entry.result))
            return error(ArtifactErrorCategory::Compilation, "ARTIFACT-CPU-SPECIALIZATION",
                         "CPU native ABI requires concrete result extents");
    }
    return {};
}

std::string typeText(const ArtifactType& type) {
    std::ostringstream out;
    out << (type.kind == ValueKind::Scalar ? "scalar<" : "tensor<")
        << storage::dtypeName(type.dtype);
    if (type.kind == ValueKind::Tensor) {
        out << ',' << type.rank << ",shape=[";
        for (std::size_t index = 0; index < type.extents.size(); ++index) {
            if (index) out << ',';
            if (type.extents[index]) out << *type.extents[index]; else out << '?';
        }
        out << ']';
    }
    return out.str() + '>';
}

std::vector<std::byte> readFile(const std::filesystem::path& path, std::uint64_t limit) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open file");
    const auto end = input.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > limit)
        throw std::runtime_error("file size exceeds V0 limit");
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), end);
    if (!input) throw std::runtime_error("short file read");
    return bytes;
}

void writeFile(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create file");
    if (!bytes.empty())
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("short file write");
}

std::vector<std::byte> serializeArtifact(const NativeArtifact& artifact) {
    const auto region = serializeRegion(artifact.region);
    Writer writer;
    writer.raw(reinterpret_cast<const std::byte*>(magic.data()), magic.size());
    writer.u32(artifact.manifest.formatVersion);
    writer.u32(artifact.manifest.compilerAbiVersion);
    writer.u32(artifact.manifest.runtimeAbiVersion);
    writer.u8(static_cast<std::uint8_t>(artifact.manifest.backend));
    writer.u8(static_cast<std::uint8_t>(artifact.manifest.payloadKind));
    writer.u8(artifact.manifest.planning.enableReuse);
    writer.u8(artifact.manifest.planning.enableFusion);
    writer.text(artifact.manifest.target);
    writer.text(artifact.manifest.runtimeRequirement);
    writer.text(artifact.manifest.entry.name);
    writer.u32(static_cast<std::uint32_t>(artifact.manifest.entry.parameters.size()));
    for (const auto& parameter : artifact.manifest.entry.parameters) writeArtifactType(writer, parameter);
    writeArtifactType(writer, artifact.manifest.entry.result);
    writer.text(artifact.manifest.planDigest);
    writer.u64(region.size());
    writer.text(artifact.manifest.regionDigest);
    writer.u64(artifact.payload.size());
    writer.text(artifact.manifest.payloadDigest);
    writer.raw(region.data(), region.size());
    writer.raw(artifact.payload.data(), artifact.payload.size());
    return writer.take();
}

std::optional<ArtifactError> validateArtifact(const NativeArtifact& artifact) {
    const auto& manifest = artifact.manifest;
    if (manifest.formatVersion != artifactFormatVersion)
        return error(ArtifactErrorCategory::Load, "ARTIFACT-FORMAT-VERSION", "unsupported artifact format version");
    if (manifest.compilerAbiVersion != compilerArtifactAbiVersion)
        return error(ArtifactErrorCategory::Load, "ARTIFACT-COMPILER-ABI", "unsupported compiler/artifact ABI");
    if (manifest.runtimeAbiVersion != nativeRuntimeAbiVersion)
        return error(ArtifactErrorCategory::Load, "ARTIFACT-RUNTIME-ABI", "unsupported native runtime ABI");
    if (manifest.backend != NativeBackend::Cpu && manifest.backend != NativeBackend::Gpu)
        return error(ArtifactErrorCategory::Load, "ARTIFACT-BACKEND", "unsupported artifact backend");
    const auto expectedKind = manifest.backend == NativeBackend::Cpu ?
        PayloadKind::ElfSharedObject : PayloadKind::Ptx;
    if (manifest.payloadKind != expectedKind)
        return error(ArtifactErrorCategory::Load, "ARTIFACT-PAYLOAD-KIND", "payload kind/backend mismatch");
    if (manifest.target != nativeTarget(manifest.backend))
        return error(ArtifactErrorCategory::Load, "ARTIFACT-TARGET", "artifact target is incompatible with this runtime");
    const std::string expectedRuntime = manifest.backend == NativeBackend::Cpu ?
        "native-runtime-abi=1;system-cxx-runtime" :
        "native-runtime-abi=1;libcuda.so.1;driver-ptx-jit;min-cc=5.0";
    if (manifest.runtimeRequirement != expectedRuntime)
        return error(ArtifactErrorCategory::Load, "ARTIFACT-RUNTIME-REQUIREMENT",
                     "artifact runtime requirement is unsupported");
    if (const auto signature = validateSignature(artifact.region, manifest.entry,
                                                 manifest.backend == NativeBackend::Cpu))
        return signature;
    const auto region = serializeRegion(artifact.region);
    if (manifest.regionSize != region.size() || manifest.regionDigest != digestBytes(region.data(), region.size()))
        return error(ArtifactErrorCategory::Load, "ARTIFACT-REGION-INTEGRITY", "lowered-region size/digest mismatch");
    if (manifest.payloadSize != artifact.payload.size())
        return error(ArtifactErrorCategory::Load, "ARTIFACT-PAYLOAD-SIZE", "payload size mismatch");
    if (artifact.payload.empty())
        return error(ArtifactErrorCategory::Load, "ARTIFACT-MISSING-PAYLOAD", "artifact payload is empty");
    if (manifest.payloadDigest != digestBytes(artifact.payload.data(), artifact.payload.size()))
        return error(ArtifactErrorCategory::Load, "ARTIFACT-PAYLOAD-INTEGRITY", "payload digest mismatch");
    if (manifest.backend == NativeBackend::Cpu) {
        if (artifact.payload.size() < 4 || artifact.payload[0] != std::byte{0x7f} ||
            artifact.payload[1] != std::byte{'E'} || artifact.payload[2] != std::byte{'L'} ||
            artifact.payload[3] != std::byte{'F'})
            return error(ArtifactErrorCategory::Load, "ARTIFACT-MALFORMED-ELF", "CPU payload is not ELF");
    } else {
        const std::string_view ptx(reinterpret_cast<const char*>(artifact.payload.data()), artifact.payload.size());
        if (!ptx.starts_with(".version ") || ptx.find("\n.target ") == std::string_view::npos ||
            ptx.find("\n.address_size 64") == std::string_view::npos ||
            ptx.find(".visible .entry ") == std::string_view::npos ||
            ptx.find('\0') != std::string_view::npos)
            return error(ArtifactErrorCategory::Load, "ARTIFACT-MALFORMED-PTX", "GPU payload is not PTX text");
    }
    auto planned = backend::buildPhysicalPlan(artifact.region,
        manifest.backend == NativeBackend::Cpu ? backend::PhysicalDevice::Host : backend::PhysicalDevice::Gpu,
        manifest.planning);
    if (!planned.ok())
        return error(ArtifactErrorCategory::Load, "ARTIFACT-PLAN", "persistent plan cannot be reconstructed");
    if (digestString(planned.plan->dump()) != manifest.planDigest)
        return error(ArtifactErrorCategory::Load, "ARTIFACT-PLAN-INTEGRITY", "physical plan identity mismatch");
    return {};
}

std::string cpuWrapper(const backend::TensorRegion& region, const EntryPoint& entry) {
    std::ostringstream out;
    out << "\n#include <cstring>\n"
           "extern \"C\" {\n"
           "struct ThiranAbiArg { std::uint32_t kind,dtype,rank,reserved; const std::uint64_t* shape; const void* data; std::uint64_t count; };\n"
           "static void thiran_abi_error(char* out,std::uint64_t cap,const char* text){if(!out||cap==0)return;std::strncpy(out,text,cap-1);out[cap-1]='\\0';}\n"
           "int thiran_v0_entry(const ThiranAbiArg* args,std::uint64_t argc,void* output,std::uint64_t output_count,char* failure,std::uint64_t failure_cap) noexcept {\n"
           " try {\n";
    out << "  if(argc!=" << entry.parameters.size() << "ULL){thiran_abi_error(failure,failure_cap,\"ARTIFACT-ARGUMENT-COUNT\");return 1;}\n";
    for (std::size_t index = 0; index < entry.parameters.size(); ++index) {
        const auto& type = entry.parameters[index];
        out << "  if(args[" << index << "].kind!=" << static_cast<unsigned>(type.kind)
            << "U||args[" << index << "].dtype!=" << static_cast<unsigned>(dtypeCode(type.dtype))
            << "U||args[" << index << "].rank!=" << type.rank << "U)"
               "{thiran_abi_error(failure,failure_cap,\"ARTIFACT-ARGUMENT-TYPE\");return 1;}\n";
        if (type.kind == ValueKind::Tensor) {
            std::uint64_t count = 1;
            for (std::size_t axis = 0; axis < type.extents.size(); ++axis) {
                count = storage::checkedMultiply(count, *type.extents[axis]);
                out << "  if(!args[" << index << "].shape||args[" << index << "].shape[" << axis
                    << "]!=" << *type.extents[axis]
                    << "ULL){thiran_abi_error(failure,failure_cap,\"ARTIFACT-ARGUMENT-SHAPE\");return 1;}\n";
            }
            out << "  if(args[" << index << "].count!=" << count
                << "ULL){thiran_abi_error(failure,failure_cap,\"ARTIFACT-ARGUMENT-SHAPE\");return 1;}\n";
            const char* scalar = type.dtype == storage::DType::I64 ? "std::int64_t" : "float";
            out << "  std::vector<" << scalar << "> input_values_" << index << ";"
                << "if(args[" << index << "].count){auto p=static_cast<const " << scalar
                << "*>(args[" << index << "].data);input_values_" << index
                << ".assign(p,p+args[" << index << "].count);}\n"
                << "  Tensor input_" << index << "=Tensor::materialize"
                << (type.dtype == storage::DType::I64 ? "I64" : "F32") << "({";
            for (std::size_t axis = 0; axis < type.extents.size(); ++axis) {
                if (axis) out << ',';
                out << *type.extents[axis] << "ULL";
            }
            out << "},input_values_" << index << ");\n";
        } else {
            const char* scalar = type.dtype == storage::DType::I64 ? "std::int64_t" : "float";
            out << "  if(args[" << index << "].count!=1||!args[" << index
                << "].data){thiran_abi_error(failure,failure_cap,\"ARTIFACT-ARGUMENT-SCALAR\");return 1;}\n"
                << "  " << scalar << " input_" << index << "=*static_cast<const " << scalar
                << "*>(args[" << index << "].data);\n";
        }
    }
    out << "  auto result=native_f" << region.function << '(';
    for (std::size_t index = 0; index < entry.parameters.size(); ++index) {
        if (index) out << ',';
        out << "input_" << index;
    }
    out << ");\n";
    if (entry.result.kind == ValueKind::Tensor) {
        const char* scalar = entry.result.dtype == storage::DType::I64 ? "std::int64_t" : "float";
        out << "  auto result_values=result."
            << (entry.result.dtype == storage::DType::I64 ? "logicalI64Values" : "logicalF32Values")
            << "();if(result_values.size()!=output_count){thiran_abi_error(failure,failure_cap,\"ARTIFACT-RESULT-SHAPE\");return 1;}"
               "if(output_count)std::memcpy(output,result_values.data(),output_count*sizeof("
            << scalar << "));\n";
    } else {
        const char* scalar = entry.result.dtype == storage::DType::I64 ? "std::int64_t" : "float";
        out << "  if(output_count!=1||!output){thiran_abi_error(failure,failure_cap,\"ARTIFACT-RESULT-SCALAR\");return 1;}"
               "*static_cast<" << scalar << "*>(output)=result;\n";
    }
    out << "  return 0;\n"
           " } catch(const SemanticFailure& e){thiran_abi_error(failure,failure_cap,e.id);return 2;}"
           " catch(const std::exception& e){thiran_abi_error(failure,failure_cap,e.what());return 3;}"
           " catch(...){thiran_abi_error(failure,failure_cap,\"unknown native failure\");return 4;}\n"
           "}\n}\n";
    return out.str();
}

struct AbiArg {
    std::uint32_t kind = 0, dtype = 0, rank = 0, reserved = 0;
    const std::uint64_t* shape = nullptr;
    const void* data = nullptr;
    std::uint64_t count = 0;
};
using EntryFunction = int (*)(const AbiArg*, std::uint64_t, void*, std::uint64_t, char*, std::uint64_t);

struct CpuModule {
    std::unique_ptr<TemporaryDirectory> directory;
    void* handle = nullptr;
    EntryFunction entry = nullptr;
    ~CpuModule() { if (handle) dlclose(handle); }
};

std::variant<std::unique_ptr<CpuModule>, ArtifactError>
loadCpuModule(const std::vector<std::byte>& payload) {
    auto module = std::make_unique<CpuModule>();
    module->directory = std::make_unique<TemporaryDirectory>("thiran-native-module-");
    if (!module->directory->valid())
        return error(ArtifactErrorCategory::Load, "ARTIFACT-TEMP", "cannot create native module directory");
    const auto path = module->directory->path() / "payload.so";
    try { writeFile(path, payload); }
    catch (const std::exception& failure) {
        return error(ArtifactErrorCategory::Load, "ARTIFACT-PAYLOAD-WRITE", failure.what());
    }
    dlerror();
    module->handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module->handle) {
        const char* detail = dlerror();
        return error(ArtifactErrorCategory::Load, "ARTIFACT-DLOPEN", detail ? detail : "dlopen failed");
    }
    dlerror();
    void* symbol = dlsym(module->handle, "thiran_v0_entry");
    if (const char* failure = dlerror())
        return error(ArtifactErrorCategory::Load, "ARTIFACT-DLSYM", failure);
    static_assert(sizeof(module->entry) == sizeof(symbol));
    std::memcpy(&module->entry, &symbol, sizeof(symbol));
    return module;
}

std::optional<ArtifactError> validateValue(const ArtifactType& type, const ArtifactValue& value,
                                           std::size_t index) {
    const auto mismatch = [&](std::string detail) {
        return error(ArtifactErrorCategory::Execution, "ARTIFACT-INPUT-ABI",
                     "argument " + std::to_string(index) + ": " + detail);
    };
    if (type.kind == ValueKind::Scalar) {
        if (type.dtype == storage::DType::I64 && !std::holds_alternative<std::int64_t>(value))
            return mismatch("expected scalar i64");
        if (type.dtype == storage::DType::F32 && !std::holds_alternative<float>(value))
            return mismatch("expected scalar f32");
        return {};
    }
    const auto* tensor = std::get_if<storage::Tensor>(&value);
    if (!tensor) return mismatch("expected tensor");
    const auto& descriptor = tensor->descriptor();
    if (descriptor.dtype != type.dtype) return mismatch("tensor dtype mismatch");
    if (descriptor.shape.size() != type.rank) return mismatch("tensor rank mismatch");
    for (std::size_t axis = 0; axis < type.extents.size(); ++axis)
        if (type.extents[axis] && descriptor.shape[axis] != *type.extents[axis])
            return mismatch("tensor shape mismatch");
    if (descriptor.view || descriptor.elementOffset != 0 || !tensor->isContiguousRowMajor())
        return mismatch("tensor layout is not materialized contiguous row-major");
    return {};
}

ArtifactExecutionResult executeCpu(CpuModule& module, const EntryPoint& entry,
                                   const std::vector<ArtifactValue>& inputs) {
    ArtifactExecutionResult result;
    if (inputs.size() != entry.parameters.size()) {
        result.error = error(ArtifactErrorCategory::Execution, "ARTIFACT-ARGUMENT-COUNT",
                             "entry argument count mismatch");
        return result;
    }
    std::vector<AbiArg> arguments(inputs.size());
    std::vector<std::vector<std::int64_t>> i64Values(inputs.size());
    std::vector<std::vector<float>> f32Values(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (auto invalid = validateValue(entry.parameters[index], inputs[index], index)) {
            result.error = std::move(invalid); return result;
        }
        auto& argument = arguments[index];
        const auto& type = entry.parameters[index];
        argument.kind = static_cast<std::uint32_t>(type.kind);
        argument.dtype = dtypeCode(type.dtype);
        argument.rank = type.rank;
        if (const auto* tensor = std::get_if<storage::Tensor>(&inputs[index])) {
            argument.shape = tensor->descriptor().shape.data();
            if (type.dtype == storage::DType::I64) {
                i64Values[index] = tensor->logicalI64Values();
                argument.data = i64Values[index].data(); argument.count = i64Values[index].size();
            } else {
                f32Values[index] = tensor->logicalF32Values();
                argument.data = f32Values[index].data(); argument.count = f32Values[index].size();
            }
        } else if (const auto* integer = std::get_if<std::int64_t>(&inputs[index])) {
            argument.data = integer; argument.count = 1;
        } else {
            argument.data = &std::get<float>(inputs[index]); argument.count = 1;
        }
    }
    std::uint64_t outputCount = 1;
    std::vector<std::uint64_t> outputShape;
    if (entry.result.kind == ValueKind::Tensor) {
        for (const auto& extent : entry.result.extents) {
            if (!extent) {
                result.error = error(ArtifactErrorCategory::Execution, "ARTIFACT-RESULT-SHAPE",
                                     "CPU result specialization is not concrete");
                return result;
            }
            outputShape.push_back(*extent);
        }
        try { outputCount = storage::checkedElementCount(outputShape); }
        catch (const std::exception& failure) {
            result.error = error(ArtifactErrorCategory::Execution, "ARTIFACT-RESULT-SIZE", failure.what());
            return result;
        }
    }
    std::vector<std::int64_t> integerOutput;
    std::vector<float> floatOutput;
    void* output = nullptr;
    if (entry.result.dtype == storage::DType::I64) {
        integerOutput.resize(outputCount); output = integerOutput.data();
    } else {
        floatOutput.resize(outputCount); output = floatOutput.data();
    }
    std::array<char, 256> failure{};
    const int status = module.entry(arguments.data(), arguments.size(), output, outputCount,
                                    failure.data(), failure.size());
    if (status != 0) {
        const std::string code = status == 2 ? failure.data() : "ARTIFACT-NATIVE-EXECUTION";
        result.error = error(ArtifactErrorCategory::Execution, code,
                             failure[0] ? failure.data() : "native entry failed");
        return result;
    }
    if (entry.result.kind == ValueKind::Scalar) {
        if (entry.result.dtype == storage::DType::I64) result.value = integerOutput.front();
        else result.value = floatOutput.front();
    } else if (entry.result.dtype == storage::DType::I64) {
        result.value = storage::Tensor::materializeI64(outputShape, integerOutput);
    } else {
        result.value = storage::Tensor::materializeF32(outputShape, floatOutput);
    }
    return result;
}

ArtifactManifest makeManifest(const backend::TensorRegion& region, NativeBackend backendKind,
                              PayloadKind payloadKind, const EntryPoint& entry,
                              backend::PhysicalPlanOptions options,
                              const backend::PhysicalPlan& plan,
                              const std::vector<std::byte>& payload) {
    const auto regionBytes = serializeRegion(region);
    ArtifactManifest manifest;
    manifest.backend = backendKind;
    manifest.payloadKind = payloadKind;
    manifest.target = nativeTarget(backendKind);
    manifest.runtimeRequirement = backendKind == NativeBackend::Cpu ?
        "native-runtime-abi=1;system-cxx-runtime" :
        "native-runtime-abi=1;libcuda.so.1;driver-ptx-jit;min-cc=5.0";
    manifest.entry = entry;
    manifest.planning = options;
    manifest.planDigest = digestString(plan.dump());
    manifest.regionSize = regionBytes.size();
    manifest.regionDigest = digestBytes(regionBytes.data(), regionBytes.size());
    manifest.payloadSize = payload.size();
    manifest.payloadDigest = digestBytes(payload.data(), payload.size());
    return manifest;
}

} // namespace

ArtifactType artifactType(const semantic::Type& type, const semantic::ShapeFact& shape) {
    ArtifactType result;
    if (type.kind == semantic::TypeKind::Tensor) {
        result.kind = ValueKind::Tensor;
        result.rank = type.rank;
        if (type.elements.size() == 1) result.dtype = storage::fromSemantic(type.elements.front().kind);
        for (std::uint32_t axis = 0; axis < type.rank; ++axis) {
            if (axis < shape.extents.size() && shape.extents[axis] && *shape.extents[axis] >= 0)
                result.extents.emplace_back(static_cast<std::uint64_t>(*shape.extents[axis]));
            else result.extents.emplace_back(std::nullopt);
        }
    } else {
        result.kind = ValueKind::Scalar;
        result.dtype = storage::fromSemantic(type.kind);
    }
    return result;
}

EntryPoint entryPoint(const backend::TensorRegion& region) {
    EntryPoint entry;
    entry.name = region.name;
    for (auto input : region.inputs) {
        const auto& inputNode = node(region, input);
        entry.parameters.push_back(artifactType(inputNode.type, inputNode.shape));
    }
    const auto& output = node(region, region.output);
    entry.result = artifactType(output.type, output.shape);
    return entry;
}

EntryPoint specializeEntry(const backend::TensorRegion& region,
                           const std::vector<ArtifactValue>& values) {
    auto entry = entryPoint(region);
    if (values.size() != entry.parameters.size())
        throw std::invalid_argument("JIT-INPUT-ARITY: specialization argument count mismatch");
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (const auto* tensor = std::get_if<storage::Tensor>(&values[index])) {
            auto& type = entry.parameters[index];
            type.kind = ValueKind::Tensor;
            type.dtype = tensor->descriptor().dtype;
            type.rank = static_cast<std::uint32_t>(tensor->descriptor().shape.size());
            type.extents.clear();
            for (auto extent : tensor->descriptor().shape) type.extents.emplace_back(extent);
        } else if (std::holds_alternative<std::int64_t>(values[index])) {
            entry.parameters[index] = {ValueKind::Scalar, storage::DType::I64, 0, {}};
        } else entry.parameters[index] = {ValueKind::Scalar, storage::DType::F32, 0, {}};
        if (const auto invalid = validateValue(entry.parameters[index], values[index], index))
            throw std::invalid_argument(invalid->message);
    }
    if (entry.result.kind == ValueKind::Tensor &&
        std::any_of(entry.result.extents.begin(), entry.result.extents.end(),
                    [](const auto& extent) { return !extent.has_value(); })) {
        for (const auto& parameter : entry.parameters) {
            if (parameter.kind == ValueKind::Tensor && parameter.dtype == entry.result.dtype &&
                parameter.rank == entry.result.rank) {
                entry.result.extents = parameter.extents;
                break;
            }
        }
    }
    return entry;
}

std::string nativeTarget(NativeBackend backendKind) {
#if defined(__linux__) && defined(__x86_64__)
    const std::string host = "linux-x86_64";
#elif defined(__linux__) && defined(__aarch64__)
    const std::string host = "linux-aarch64";
#else
    const std::string host = "unsupported-host";
#endif
    return backendKind == NativeBackend::Cpu ? host : "ptx60-sm50-min-cc50";
}

std::string digestBytes(const std::byte* bytes, std::size_t size) {
    std::uint64_t value = 1469598103934665603ULL;
    for (std::size_t index = 0; index < size; ++index) {
        value ^= std::to_integer<std::uint8_t>(bytes[index]);
        value *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}
std::string digestString(std::string_view value) {
    return digestBytes(reinterpret_cast<const std::byte*>(value.data()), value.size());
}

std::optional<ArtifactError> writeArtifact(const NativeArtifact& artifact,
                                           const std::filesystem::path& path) {
    try {
        const auto parent = std::filesystem::absolute(path).parent_path();
        if (!std::filesystem::is_directory(parent))
            return error(ArtifactErrorCategory::Compilation, "ARTIFACT-OUTPUT", "output parent does not exist");
        writeFile(path, serializeArtifact(artifact));
        return {};
    } catch (const std::exception& failure) {
        return error(ArtifactErrorCategory::Compilation, "ARTIFACT-WRITE", failure.what());
    }
}

ArtifactBuildResult buildCpuAot(const backend::TensorRegion& region,
                                const NativeToolchain& toolchain,
                                const ArtifactBuildOptions& options) {
    ArtifactBuildResult result;
    try {
        const auto verification = backend::verifyRegion(region);
        if (!verification.ok()) {
            result.error = error(ArtifactErrorCategory::Compilation, "ARTIFACT-INVALID-REGION", verification.errors.front());
            return result;
        }
        const auto entry = options.specializedEntry.value_or(entryPoint(region));
        if (auto invalid = validateSignature(region, entry, true)) {
            invalid->category = ArtifactErrorCategory::Compilation;
            result.error = std::move(invalid); return result;
        }
        auto planned = backend::buildPhysicalPlan(region, backend::PhysicalDevice::Host, options.planning);
        if (!planned.ok()) {
            result.error = error(ArtifactErrorCategory::Compilation, "ARTIFACT-PLAN",
                                 planned.errors.empty() ? "CPU physical planning failed" : planned.errors.front());
            return result;
        }
        if (toolchain.compilerExecutable.empty()) {
            result.error = error(ArtifactErrorCategory::Compilation, "AOT-NATIVE-COMPILER",
                                 "host native compiler is not configured");
            return result;
        }
        TemporaryDirectory temporary("thiran-aot-build-");
        if (!temporary.valid()) {
            result.error = error(ArtifactErrorCategory::Compilation, "ARTIFACT-TEMP", "cannot create build directory");
            return result;
        }
        const auto sourcePath = temporary.path() / "generated.cpp";
        const auto payloadPath = temporary.path() / "payload.so";
        const auto source = backend::emitCpp20(region, false, cpuWrapper(region, entry), options.planning);
        {
            std::ofstream output(sourcePath, std::ios::binary);
            output.write(source.data(), static_cast<std::streamsize>(source.size()));
            if (!output) throw std::runtime_error("cannot write generated native source");
        }
        tooling::ProcessRequest request{toolchain.compilerExecutable, toolchain.compilerArguments};
        request.arguments.insert(request.arguments.end(),
                                 {"-std=c++20", "-fPIC", "-shared", "-Wl,--strip-debug"});
        for (const auto& include : toolchain.includePaths) request.arguments.push_back("-I" + include.string());
        request.arguments.push_back(sourcePath.string());
        for (const auto& library : toolchain.staticLibraries) request.arguments.push_back(library.string());
        request.arguments.push_back("-o"); request.arguments.push_back(payloadPath.string());
        result.compilerProcess = tooling::runProcess(request);
        if (!result.compilerProcess->launched || result.compilerProcess->exitStatus != 0) {
            result.error = error(ArtifactErrorCategory::Compilation, "AOT-NATIVE-COMPILER",
                result.compilerProcess->launcherError.empty() ? result.compilerProcess->standardError :
                                                                 result.compilerProcess->launcherError);
            return result;
        }
        auto payload = readFile(payloadPath, maxPayloadBytes);
        NativeArtifact artifact;
        artifact.region = region;
        artifact.payload = std::move(payload);
        artifact.manifest = makeManifest(region, NativeBackend::Cpu, PayloadKind::ElfSharedObject,
                                         entry, options.planning, *planned.plan, artifact.payload);
        if (auto failure = validateArtifact(artifact)) { result.error = std::move(failure); return result; }
        if (auto failure = writeArtifact(artifact, options.output)) {
            result.error = std::move(failure); return result;
        }
        result.manifest = artifact.manifest;
        result.success = true;
    } catch (const std::exception& failure) {
        result.error = error(ArtifactErrorCategory::Compilation, "ARTIFACT-CPU-BUILD", failure.what());
    }
    return result;
}

ArtifactBuildResult buildGpuAot(const backend::TensorRegion& region,
                                const ArtifactBuildOptions& options) {
    ArtifactBuildResult result;
    try {
        if (!backend::nativeGpuBackendBuilt()) {
            result.error = error(ArtifactErrorCategory::Compilation, "GPU-BACKEND-NOT-BUILT",
                                 "native GPU artifact support is disabled");
            return result;
        }
        const auto verification = backend::verifyRegion(region);
        if (!verification.ok()) {
            result.error = error(ArtifactErrorCategory::Compilation, "ARTIFACT-INVALID-REGION", verification.errors.front());
            return result;
        }
        const auto entry = options.specializedEntry.value_or(entryPoint(region));
        if (auto invalid = validateSignature(region, entry, false)) {
            invalid->category = ArtifactErrorCategory::Compilation;
            result.error = std::move(invalid); return result;
        }
        auto planned = backend::buildPhysicalPlan(region, backend::PhysicalDevice::Gpu, options.planning);
        if (!planned.ok()) {
            result.error = error(ArtifactErrorCategory::Compilation, "ARTIFACT-PLAN",
                                 planned.errors.empty() ? "GPU physical planning failed" : planned.errors.front());
            return result;
        }
        const auto ptx = backend::emitNativeGpuPtx(region, options.planning);
        std::vector<std::byte> payload(ptx.size());
        if (!ptx.empty()) std::memcpy(payload.data(), ptx.data(), ptx.size());
        NativeArtifact artifact;
        artifact.region = region;
        artifact.payload = std::move(payload);
        artifact.manifest = makeManifest(region, NativeBackend::Gpu, PayloadKind::Ptx,
                                         entry, options.planning, *planned.plan, artifact.payload);
        if (auto failure = validateArtifact(artifact)) { result.error = std::move(failure); return result; }
        if (auto failure = writeArtifact(artifact, options.output)) {
            result.error = std::move(failure); return result;
        }
        result.manifest = artifact.manifest;
        result.success = true;
    } catch (const std::exception& failure) {
        result.error = error(ArtifactErrorCategory::Compilation, "ARTIFACT-GPU-BUILD", failure.what());
    }
    return result;
}

ArtifactLoadResult loadArtifact(const std::filesystem::path& path) {
    ArtifactLoadResult result;
    try {
        const auto bytes = readFile(path, maxPayloadBytes + maxRegionBytes + (4U << 20));
        Reader reader(bytes);
        for (char expected : magic)
            if (reader.u8() != static_cast<std::uint8_t>(expected))
                throw std::runtime_error("ARTIFACT-MAGIC: missing THIRAN16 header");
        NativeArtifact artifact;
        auto& manifest = artifact.manifest;
        manifest.formatVersion = reader.u32();
        manifest.compilerAbiVersion = reader.u32();
        manifest.runtimeAbiVersion = reader.u32();
        manifest.backend = static_cast<NativeBackend>(reader.u8());
        manifest.payloadKind = static_cast<PayloadKind>(reader.u8());
        manifest.planning.enableReuse = reader.u8() != 0;
        manifest.planning.enableFusion = reader.u8() != 0;
        manifest.target = reader.text();
        manifest.runtimeRequirement = reader.text();
        manifest.entry.name = reader.text();
        const auto parameterCount = reader.u32();
        if (parameterCount > 64) throw std::runtime_error("ARTIFACT-SIGNATURE: too many parameters");
        for (std::uint32_t index = 0; index < parameterCount; ++index)
            manifest.entry.parameters.push_back(readArtifactType(reader));
        manifest.entry.result = readArtifactType(reader);
        manifest.planDigest = reader.text();
        manifest.regionSize = reader.u64();
        manifest.regionDigest = reader.text();
        manifest.payloadSize = reader.u64();
        manifest.payloadDigest = reader.text();
        const auto regionBytes = reader.raw(manifest.regionSize, maxRegionBytes);
        artifact.payload = reader.raw(manifest.payloadSize, maxPayloadBytes);
        if (!reader.done()) throw std::runtime_error("ARTIFACT-TRAILING-DATA: bytes follow declared payload");
        artifact.region = deserializeRegion(regionBytes);
        if (auto invalid = validateArtifact(artifact)) { result.error = std::move(invalid); return result; }
        result.artifact = std::move(artifact);
    } catch (const std::exception& failure) {
        std::string code = "ARTIFACT-MALFORMED";
        const std::string message = failure.what();
        if (message.starts_with("ARTIFACT-")) code = message.substr(0, message.find(':'));
        result.error = error(ArtifactErrorCategory::Load, std::move(code), message);
    }
    return result;
}

std::string inspectArtifact(const NativeArtifact& artifact) {
    const auto& manifest = artifact.manifest;
    std::ostringstream out;
    out << "format_version=" << manifest.formatVersion << '\n'
        << "compiler_artifact_abi=" << manifest.compilerAbiVersion << '\n'
        << "runtime_abi=" << manifest.runtimeAbiVersion << '\n'
        << "backend=" << (manifest.backend == NativeBackend::Cpu ? "cpu" : "gpu") << '\n'
        << "target=" << manifest.target << '\n'
        << "entry=" << manifest.entry.name << '\n'
        << "parameters=" << manifest.entry.parameters.size() << '\n';
    for (std::size_t index = 0; index < manifest.entry.parameters.size(); ++index)
        out << "parameter[" << index << "]=" << typeText(manifest.entry.parameters[index]) << '\n';
    out << "result=" << typeText(manifest.entry.result) << '\n'
        << "planning.reuse=" << (manifest.planning.enableReuse ? "true" : "false") << '\n'
        << "planning.fusion=" << (manifest.planning.enableFusion ? "true" : "false") << '\n'
        << "plan_digest=" << manifest.planDigest << '\n'
        << "payload_kind=" << (manifest.payloadKind == PayloadKind::ElfSharedObject ? "elf-shared-object" : "ptx") << '\n'
        << "payload_size=" << manifest.payloadSize << '\n'
        << "payload_digest=" << manifest.payloadDigest << '\n'
        << "region_size=" << manifest.regionSize << '\n'
        << "region_digest=" << manifest.regionDigest << '\n'
        << "runtime_requirements=" << manifest.runtimeRequirement << '\n';
    return out.str();
}

ArtifactExecutionResult executeArtifact(const NativeArtifact& artifact,
                                        const std::vector<ArtifactValue>& inputs, int device) {
    ArtifactExecutionResult result;
    if (auto invalid = validateArtifact(artifact)) { result.error = std::move(invalid); return result; }
    if (inputs.size() != artifact.manifest.entry.parameters.size()) {
        result.error = error(ArtifactErrorCategory::Execution, "ARTIFACT-ARGUMENT-COUNT",
                             "entry argument count mismatch");
        return result;
    }
    for (std::size_t index = 0; index < inputs.size(); ++index)
        if (auto invalid = validateValue(artifact.manifest.entry.parameters[index], inputs[index], index)) {
            result.error = std::move(invalid); return result;
        }
    if (artifact.manifest.backend == NativeBackend::Cpu) {
        auto loaded = loadCpuModule(artifact.payload);
        if (auto* failure = std::get_if<ArtifactError>(&loaded)) {
            result.error = std::move(*failure); return result;
        }
        return executeCpu(*std::get<std::unique_ptr<CpuModule>>(loaded), artifact.manifest.entry, inputs);
    }
    auto planned = backend::buildPhysicalPlan(artifact.region, backend::PhysicalDevice::Gpu,
                                              artifact.manifest.planning);
    if (!planned.ok()) {
        result.error = error(ArtifactErrorCategory::Load, "ARTIFACT-PLAN", "GPU plan reconstruction failed");
        return result;
    }
    std::string ptx(reinterpret_cast<const char*>(artifact.payload.data()), artifact.payload.size());
    auto executed = backend::executeNativeGpuPayload(
        artifact.region, *planned.plan, std::move(ptx), inputs, device);
    result.gpuEvidence = executed.evidence;
    if (!executed.ok()) {
        result.error = error(ArtifactErrorCategory::Execution,
            executed.error ? executed.error->code : "GPU-EXECUTION",
            executed.error ? executed.error->message : "GPU artifact execution failed");
    } else result.value = std::move(executed.value);
    return result;
}

ArtifactExecutionResult loadAndExecuteArtifact(const std::filesystem::path& path,
                                               const std::vector<ArtifactValue>& inputs,
                                               int device) {
    auto loaded = loadArtifact(path);
    if (!loaded.ok()) return {{}, loaded.error, {}};
    return executeArtifact(*loaded.artifact, inputs, device);
}

std::string formatArtifactValue(const ArtifactValue& value) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    if (const auto* integer = std::get_if<std::int64_t>(&value))
        out << "{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"i64\",\"value\":" << *integer << '}';
    else if (const auto* floating = std::get_if<float>(&value))
        out << "{\"status\":\"ok\",\"kind\":\"scalar\",\"dtype\":\"f32\",\"value\":" << *floating << '}';
    else {
        const auto& tensor = std::get<storage::Tensor>(value);
        const bool integerTensor = tensor.descriptor().dtype == storage::DType::I64;
        out << "{\"status\":\"ok\",\"kind\":\"tensor\",\"dtype\":\""
            << (integerTensor ? "i64" : "f32") << "\",\"shape\":[";
        for (std::size_t axis = 0; axis < tensor.descriptor().shape.size(); ++axis) {
            if (axis) out << ',';
            out << tensor.descriptor().shape[axis];
        }
        out << "],\"values\":[";
        if (integerTensor) {
            const auto values = tensor.logicalI64Values();
            for (std::size_t index = 0; index < values.size(); ++index) {
                if (index) out << ',';
                out << values[index];
            }
        } else {
            const auto values = tensor.logicalF32Values();
            for (std::size_t index = 0; index < values.size(); ++index) {
                if (index) out << ',';
                out << values[index];
            }
        }
        out << "]}";
    }
    return out.str();
}

struct JitExecutable::Impl {
    NativeBackend backend = NativeBackend::Cpu;
    EntryPoint entry;
    std::string key;
    std::unique_ptr<CpuModule> cpu;
    backend::TensorRegion region;
    std::optional<backend::PhysicalPlan> plan;
    std::string ptx;
};

JitExecutable::JitExecutable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
JitExecutable::~JitExecutable() = default;
NativeBackend JitExecutable::backend() const noexcept { return impl_->backend; }
const EntryPoint& JitExecutable::entry() const noexcept { return impl_->entry; }
const std::string& JitExecutable::cacheKey() const noexcept { return impl_->key; }
ArtifactExecutionResult JitExecutable::execute(const std::vector<ArtifactValue>& inputs,
                                               int device) const {
    if (inputs.size() != impl_->entry.parameters.size())
        return {{}, error(ArtifactErrorCategory::Execution, "ARTIFACT-ARGUMENT-COUNT",
                          "entry argument count mismatch"), {}};
    for (std::size_t index = 0; index < inputs.size(); ++index)
        if (auto invalid = validateValue(impl_->entry.parameters[index], inputs[index], index))
            return {{}, std::move(invalid), {}};
    if (impl_->backend == NativeBackend::Cpu) return executeCpu(*impl_->cpu, impl_->entry, inputs);
    auto executed = backend::executeNativeGpuPayload(impl_->region, *impl_->plan, impl_->ptx, inputs, device);
    ArtifactExecutionResult result;
    result.gpuEvidence = executed.evidence;
    if (executed.ok()) result.value = std::move(executed.value);
    else result.error = error(ArtifactErrorCategory::Execution,
        executed.error ? executed.error->code : "GPU-JIT-EXECUTION",
        executed.error ? executed.error->message : "GPU JIT execution failed");
    return result;
}

JitCompiler::JitCompiler(NativeToolchain toolchain) : toolchain_(std::move(toolchain)) {}

std::string JitCompiler::cacheIdentity(const backend::TensorRegion& region,
                                       NativeBackend backendKind, const EntryPoint& entry,
                                       backend::PhysicalPlanOptions options) {
    const auto regionBytes = serializeRegion(region);
    Writer writer;
    writer.u32(compilerArtifactAbiVersion);
    writer.u32(nativeRuntimeAbiVersion);
    writer.u8(static_cast<std::uint8_t>(backendKind));
    writer.text(nativeTarget(backendKind));
    writer.u8(options.enableReuse); writer.u8(options.enableFusion);
    writer.text(entry.name);
    writer.u32(static_cast<std::uint32_t>(entry.parameters.size()));
    for (const auto& parameter : entry.parameters) writeArtifactType(writer, parameter);
    writeArtifactType(writer, entry.result);
    writer.raw(regionBytes.data(), regionBytes.size());
    return digestBytes(writer.bytes().data(), writer.bytes().size());
}

JitCompileResult JitCompiler::compileCpu(const backend::TensorRegion& region,
                                         const std::vector<ArtifactValue>& values,
                                         backend::PhysicalPlanOptions options) {
    JitCompileResult result;
    EntryPoint entry;
    try { entry = values.empty() ? entryPoint(region) : specializeEntry(region, values); }
    catch (const std::exception& failure) {
        result.error = error(ArtifactErrorCategory::Compilation, "JIT-INVALID-INPUT", failure.what());
        return result;
    }
    const auto key = cacheIdentity(region, NativeBackend::Cpu, entry, options);
    if (const auto found = cache_.find(key); found != cache_.end()) {
        ++statistics_.cacheHits; result.cacheHit = true; result.executable = found->second; return result;
    }
    ++statistics_.cacheMisses;
    TemporaryDirectory temporary("thiran-cpu-jit-");
    if (!temporary.valid()) {
        result.error = error(ArtifactErrorCategory::Compilation, "JIT-TEMP", "cannot create JIT directory");
        return result;
    }
    ArtifactBuildOptions build{temporary.path() / "jit.tha", options, entry};
    auto built = buildCpuAot(region, toolchain_, build);
    if (!built.success) {
        result.error = built.error;
        if (result.error && result.error->code == "AOT-NATIVE-COMPILER")
            result.error->code = "JIT-NATIVE-COMPILER";
        if (result.error && result.error->code == "ARTIFACT-INVALID-REGION")
            result.error->code = "JIT-INVALID-INPUT";
        return result;
    }
    auto loaded = loadArtifact(build.output);
    if (!loaded.ok()) { result.error = loaded.error; return result; }
    auto cpu = loadCpuModule(loaded.artifact->payload);
    if (auto* failure = std::get_if<ArtifactError>(&cpu)) {
        result.error = std::move(*failure); return result;
    }
    auto impl = std::make_unique<JitExecutable::Impl>();
    impl->backend = NativeBackend::Cpu;
    impl->entry = std::move(entry);
    impl->key = key;
    impl->cpu = std::move(std::get<std::unique_ptr<CpuModule>>(cpu));
    result.executable = std::shared_ptr<JitExecutable>(new JitExecutable(std::move(impl)));
    cache_.emplace(key, result.executable);
    ++statistics_.compilations;
    return result;
}

JitCompileResult JitCompiler::compileGpu(const backend::TensorRegion& region,
                                         const std::vector<ArtifactValue>& values,
                                         backend::PhysicalPlanOptions options) {
    JitCompileResult result;
    if (values.empty() && !region.inputs.empty()) {
        result.error = error(ArtifactErrorCategory::Compilation, "JIT-INVALID-INPUT",
                             "GPU JIT requires a concrete runtime specialization");
        return result;
    }
    EntryPoint entry;
    try { entry = values.empty() ? entryPoint(region) : specializeEntry(region, values); }
    catch (const std::exception& failure) {
        result.error = error(ArtifactErrorCategory::Compilation, "JIT-INVALID-INPUT", failure.what());
        return result;
    }
    const auto key = cacheIdentity(region, NativeBackend::Gpu, entry, options);
    if (const auto found = cache_.find(key); found != cache_.end()) {
        ++statistics_.cacheHits; result.cacheHit = true; result.executable = found->second; return result;
    }
    ++statistics_.cacheMisses;
    const auto verification = backend::verifyRegion(region);
    if (!verification.ok()) {
        result.error = error(ArtifactErrorCategory::Compilation, "JIT-INVALID-INPUT", verification.errors.front());
        return result;
    }
    if (auto invalid = validateSignature(region, entry, false)) {
        invalid->category = ArtifactErrorCategory::Compilation; result.error = std::move(invalid); return result;
    }
    auto plan = backend::buildPhysicalPlan(region, backend::PhysicalDevice::Gpu, options);
    if (!plan.ok()) {
        result.error = error(ArtifactErrorCategory::Compilation, "JIT-GPU-PLAN",
                             plan.errors.empty() ? "GPU JIT planning failed" : plan.errors.front());
        return result;
    }
    auto ptx = backend::emitNativeGpuPtx(region, options);
    if (ptx.empty()) {
        result.error = error(ArtifactErrorCategory::Compilation, "JIT-GPU-PTX",
                             "GPU PTX generation produced no payload");
        return result;
    }
    auto impl = std::make_unique<JitExecutable::Impl>();
    impl->backend = NativeBackend::Gpu;
    impl->entry = std::move(entry);
    impl->key = key;
    impl->region = region;
    impl->plan = std::move(plan.plan);
    impl->ptx = std::move(ptx);
    result.executable = std::shared_ptr<JitExecutable>(new JitExecutable(std::move(impl)));
    cache_.emplace(key, result.executable);
    ++statistics_.compilations;
    return result;
}

} // namespace thiran::v0::artifact
