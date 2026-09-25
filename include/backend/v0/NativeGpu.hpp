#pragma once

#include "backend/v0/TensorRegion.hpp"
#include "storage/v0/Storage.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace thiran::v0::backend {

NativeResult extractStrictGpu(const semantic::Module&,
                              const analysis::OwnershipAnalysisResult&,
                              const std::string& functionName,
                              bool standalone);

enum class GpuErrorCategory {
    BackendUnavailable,
    InvalidDevice,
    BackendUnsupported,
    RuntimeFailure,
    SemanticFailure
};

struct GpuError {
    GpuErrorCategory category = GpuErrorCategory::RuntimeFailure;
    std::string code;
    std::string message;
};

struct GpuDeviceInfo {
    bool backendBuilt = false;
    bool driverLoaded = false;
    bool deviceAvailable = false;
    int deviceCount = 0;
    int selectedDevice = -1;
    int driverVersion = 0;
    int computeMajor = 0;
    int computeMinor = 0;
    std::string name;
    std::optional<GpuError> error;
};

using GpuValue = std::variant<std::int64_t, float, storage::Tensor>;

struct GpuExecutionEvidence {
    GpuDeviceInfo device;
    std::uint64_t allocationCount = 0;
    std::uint64_t hostToDeviceCopies = 0;
    std::uint64_t deviceToHostCopies = 0;
    std::uint64_t deviceToDeviceCopies = 0;
    std::uint64_t kernelLaunches = 0;
    std::uint64_t synchronizations = 0;
};

struct GpuExecutionResult {
    std::optional<GpuValue> value;
    std::optional<GpuError> error;
    GpuExecutionEvidence evidence;
    bool ok() const { return value.has_value() && !error.has_value(); }
};

bool nativeGpuBackendBuilt() noexcept;
// Deterministic backend artifact for compiler tests and diagnostics. Empty
// when the native GPU implementation was disabled at build time.
std::string emitNativeGpuPtx();
GpuDeviceInfo probeNativeGpu(int device = 0) noexcept;

// Synchronous native execution. Inputs cross an explicit host-to-device
// boundary, kernels are synchronized and checked, and the result crosses an
// explicit device-to-host boundary before this function returns. There is no
// evaluator, native-CPU, framework, or retry fallback.
GpuExecutionResult executeNativeGpu(const TensorRegion&,
                                    const std::vector<GpuValue>& inputs = {},
                                    int device = 0) noexcept;

}
