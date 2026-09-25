#pragma once

#include "backend/v0/NativeGpu.hpp"
#include "tooling/v0/Process.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace thiran::v0::artifact {

inline constexpr std::uint32_t artifactFormatVersion = 0;
inline constexpr std::uint32_t compilerArtifactAbiVersion = 1;
inline constexpr std::uint32_t nativeRuntimeAbiVersion = 1;

enum class NativeBackend : std::uint8_t { Cpu = 1, Gpu = 2 };
enum class PayloadKind : std::uint8_t { ElfSharedObject = 1, Ptx = 2 };
enum class ValueKind : std::uint8_t { Scalar = 1, Tensor = 2 };

struct ArtifactType {
    ValueKind kind = ValueKind::Scalar;
    storage::DType dtype = storage::DType::Invalid;
    std::uint32_t rank = 0;
    std::vector<std::optional<std::uint64_t>> extents;
    bool operator==(const ArtifactType&) const = default;
};

struct EntryPoint {
    std::string name;
    std::vector<ArtifactType> parameters;
    ArtifactType result;
    bool operator==(const EntryPoint&) const = default;
};

struct ArtifactManifest {
    std::uint32_t formatVersion = artifactFormatVersion;
    std::uint32_t compilerAbiVersion = compilerArtifactAbiVersion;
    std::uint32_t runtimeAbiVersion = nativeRuntimeAbiVersion;
    NativeBackend backend = NativeBackend::Cpu;
    PayloadKind payloadKind = PayloadKind::ElfSharedObject;
    std::string target;
    std::string runtimeRequirement;
    EntryPoint entry;
    backend::PhysicalPlanOptions planning;
    std::string planDigest;
    std::uint64_t regionSize = 0;
    std::string regionDigest;
    std::uint64_t payloadSize = 0;
    std::string payloadDigest;
    bool operator==(const ArtifactManifest&) const = default;
};

struct NativeArtifact {
    ArtifactManifest manifest;
    backend::TensorRegion region;
    std::vector<std::byte> payload;
};

enum class ArtifactErrorCategory { Compilation, Load, Execution };
struct ArtifactError {
    ArtifactErrorCategory category = ArtifactErrorCategory::Load;
    std::string code;
    std::string message;
};

struct ArtifactLoadResult {
    std::optional<NativeArtifact> artifact;
    std::optional<ArtifactError> error;
    bool ok() const noexcept { return artifact.has_value() && !error.has_value(); }
};

struct ArtifactBytesResult {
    std::optional<std::vector<std::byte>> bytes;
    std::optional<ArtifactError> error;
    bool ok() const noexcept { return bytes.has_value() && !error.has_value(); }
};

using ArtifactValue = backend::GpuValue;
struct ArtifactExecutionResult {
    std::optional<ArtifactValue> value;
    std::optional<ArtifactError> error;
    std::optional<backend::GpuExecutionEvidence> gpuEvidence;
    bool ok() const noexcept { return value.has_value() && !error.has_value(); }
};

struct NativeToolchain {
    std::string compilerExecutable;
    std::vector<std::string> compilerArguments;
    std::vector<std::filesystem::path> includePaths;
    std::vector<std::filesystem::path> staticLibraries;
};

struct ArtifactBuildOptions {
    std::filesystem::path output;
    backend::PhysicalPlanOptions planning;
    std::optional<EntryPoint> specializedEntry;
};

struct ArtifactBuildResult {
    bool success = false;
    std::optional<ArtifactError> error;
    std::optional<ArtifactManifest> manifest;
    std::optional<tooling::ProcessResult> compilerProcess;
};

ArtifactType artifactType(const semantic::Type&, const semantic::ShapeFact&);
EntryPoint entryPoint(const backend::TensorRegion&);
EntryPoint specializeEntry(const backend::TensorRegion&, const std::vector<ArtifactValue>&);
std::string nativeTarget(NativeBackend);
std::string digestBytes(const std::byte*, std::size_t);
std::string digestString(std::string_view);

ArtifactBuildResult buildCpuAot(const backend::TensorRegion&, const NativeToolchain&,
                                const ArtifactBuildOptions&);
ArtifactBuildResult buildGpuAot(const backend::TensorRegion&, const ArtifactBuildOptions&);
ArtifactLoadResult loadArtifact(const std::filesystem::path&);
ArtifactLoadResult loadArtifactBytes(const std::vector<std::byte>&);
ArtifactBytesResult encodeArtifact(const NativeArtifact&);
std::string inspectArtifact(const NativeArtifact&);
std::optional<ArtifactError> writeArtifact(const NativeArtifact&,
                                           const std::filesystem::path&);
ArtifactExecutionResult executeArtifact(const NativeArtifact&,
                                        const std::vector<ArtifactValue>& = {}, int device = 0);

struct ArtifactGpuSubmission {
    std::optional<backend::PendingGpuExecution> pending;
    std::optional<ArtifactError> error;
    std::optional<backend::GpuExecutionEvidence> gpuEvidence;
    bool ok() const noexcept { return pending.has_value() && !error.has_value(); }
};

ArtifactGpuSubmission submitArtifactGpu(const NativeArtifact&,
                                        const std::vector<ArtifactValue>& = {}, int device = 0);
ArtifactExecutionResult loadAndExecuteArtifact(const std::filesystem::path&,
                                               const std::vector<ArtifactValue>& = {}, int device = 0);
std::string formatArtifactValue(const ArtifactValue&);

struct JitStatistics {
    std::uint64_t compilations = 0;
    std::uint64_t cacheHits = 0;
    std::uint64_t cacheMisses = 0;
};

class JitExecutable {
public:
    JitExecutable(const JitExecutable&) = delete;
    JitExecutable& operator=(const JitExecutable&) = delete;
    ~JitExecutable();
    NativeBackend backend() const noexcept;
    const EntryPoint& entry() const noexcept;
    const std::string& cacheKey() const noexcept;
    ArtifactExecutionResult execute(const std::vector<ArtifactValue>& = {}, int device = 0) const;
private:
    struct Impl;
    explicit JitExecutable(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend class JitCompiler;
};

struct JitCompileResult {
    std::shared_ptr<JitExecutable> executable;
    std::optional<ArtifactError> error;
    bool cacheHit = false;
    bool ok() const noexcept { return executable && !error.has_value(); }
};

class JitCompiler {
public:
    explicit JitCompiler(NativeToolchain = {});
    JitCompileResult compileCpu(const backend::TensorRegion&,
                                const std::vector<ArtifactValue>& = {},
                                backend::PhysicalPlanOptions = {});
    JitCompileResult compileGpu(const backend::TensorRegion&,
                                const std::vector<ArtifactValue>& = {},
                                backend::PhysicalPlanOptions = {});
    JitStatistics statistics() const noexcept { return statistics_; }
    static std::string cacheIdentity(const backend::TensorRegion&, NativeBackend,
                                     const EntryPoint&, backend::PhysicalPlanOptions);
private:
    NativeToolchain toolchain_;
    JitStatistics statistics_;
    std::map<std::string, std::shared_ptr<JitExecutable>> cache_;
};

} // namespace thiran::v0::artifact
