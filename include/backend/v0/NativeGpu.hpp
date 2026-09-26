#pragma once

#include "backend/v0/PhysicalPlan.hpp"
#include "runtime/v0/Async.hpp"
#include "storage/v0/Storage.hpp"

#include <cstdint>
#include <cstddef>
#include <memory>
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
    ValidationFailure,
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
    std::uint64_t allocationBytesRequested = 0;
    std::uint64_t hostToDeviceCopies = 0;
    std::uint64_t hostToDeviceBytes = 0;
    std::uint64_t deviceToHostCopies = 0;
    std::uint64_t deviceToHostBytes = 0;
    std::uint64_t deviceToDeviceCopies = 0;
    std::uint64_t deviceToDeviceBytes = 0;
    std::uint64_t moduleLoads = 0;
    std::uint64_t driverJitLoads = 0;
    std::uint64_t inputStorageUploads = 0;
    std::uint64_t resultDownloads = 0;
    std::uint64_t kernelLaunches = 0;
    std::uint64_t synchronizations = 0;
    std::uint64_t submittedOperations = 0;
    std::uint64_t pendingOperations = 0;
    std::uint64_t activeReadReservations = 0;
    std::uint64_t activeWriteReservations = 0;
    std::uint64_t streamSubmissions = 0;
    std::uint64_t eventRecords = 0;
    std::uint64_t observations = 0;
    std::uint64_t droppedDrains = 0;
    std::uint64_t releasedReservations = 0;
    std::uint64_t logicalTensorValues = 0;
    std::uint64_t logicalIntermediates = 0;
    std::uint64_t materializedIntermediates = 0;
    std::uint64_t physicalSlots = 0;
    std::uint64_t physicalTemporarySlots = 0;
    std::uint64_t reusedSlotAssignments = 0;
    std::uint64_t fusionGroups = 0;
    std::uint64_t fusedKernelGroups = 0;
    std::uint64_t plannerOwnedAllocations = 0;
    std::uint64_t retainedPlannerAllocations = 0;
};

struct GpuExecutionResult {
    std::optional<GpuValue> value;
    std::optional<GpuError> error;
    GpuExecutionEvidence evidence;
    bool ok() const { return value.has_value() && !error.has_value(); }
};

bool nativeGpuBackendBuilt() noexcept;
analysis::EffectSet nativeGpuAsyncEffects() noexcept;
// Deterministic backend artifact for compiler tests and diagnostics. Empty
// when the native GPU implementation was disabled at build time.
std::string emitNativeGpuPtx();
std::string emitNativeGpuPtx(const TensorRegion&, PhysicalPlanOptions = {});
GpuDeviceInfo probeNativeGpu(int device = 0) noexcept;

struct NativeGpuPendingState;
struct GpuAsyncSubmission;

// A pending value is intentionally not a GpuValue. Only observe() can publish
// the completed value or its checked deferred failure.
class PendingGpuExecution {
public:
    PendingGpuExecution() = default;
    PendingGpuExecution(const PendingGpuExecution&) = delete;
    PendingGpuExecution& operator=(const PendingGpuExecution&) = delete;
    PendingGpuExecution(PendingGpuExecution&&) noexcept = default;
    PendingGpuExecution& operator=(PendingGpuExecution&&) noexcept = default;
    bool valid() const noexcept;
    runtime::AsyncOperationState state() const noexcept;
    bool observed() const noexcept;
    std::optional<GpuValue> value() const;
    runtime::AsyncResource outputResource() const;
    GpuExecutionEvidence evidence() const noexcept;
    GpuExecutionResult observe() noexcept;
private:
    std::shared_ptr<NativeGpuPendingState> state_;
    runtime::AsyncOperation operation_;
    PendingGpuExecution(std::shared_ptr<NativeGpuPendingState>, runtime::AsyncOperation);
    friend struct GpuAsyncSubmission;
    friend GpuAsyncSubmission submitNativeGpuAsync(const TensorRegion&,
                                                    const std::vector<GpuValue>&,
                                                    int,
                                                    PhysicalPlanOptions) noexcept;
    friend GpuAsyncSubmission submitNativeGpuPayloadAsync(const TensorRegion&,
                                                           const PhysicalPlan&,
                                                           std::string,
                                                           const std::vector<GpuValue>&,
                                                           int) noexcept;
};

struct GpuAsyncSubmission {
    std::optional<PendingGpuExecution> pending;
    std::optional<GpuError> error;
    GpuExecutionEvidence evidence;
    bool ok() const noexcept { return pending.has_value() && !error.has_value(); }
};

// Backend-internal untyped PTX execution substrate for structured native
// runtimes (such as graphics) that are not TensorRegion DAGs. CUDA launch
// geometry and device pointers remain private to the backend.
struct NativeGpuKernelRequest {
    std::string ptx;
    std::string entry;
    std::uint64_t workItems = 0;
    std::vector<std::vector<std::byte>> inputBuffers;
    std::vector<std::size_t> outputByteCounts;
    std::vector<runtime::AsyncResource> retainedReadResources;
};

struct NativeGpuKernelPendingState;
struct NativeGpuKernelSubmission;

struct NativeGpuKernelResult {
    std::optional<std::vector<std::vector<std::byte>>> outputs;
    std::optional<GpuError> error;
    GpuExecutionEvidence evidence;
    bool ok() const noexcept { return outputs.has_value() && !error.has_value(); }
};

class PendingNativeGpuKernel {
public:
    PendingNativeGpuKernel() = default;
    PendingNativeGpuKernel(const PendingNativeGpuKernel&) = delete;
    PendingNativeGpuKernel& operator=(const PendingNativeGpuKernel&) = delete;
    PendingNativeGpuKernel(PendingNativeGpuKernel&&) noexcept = default;
    PendingNativeGpuKernel& operator=(PendingNativeGpuKernel&&) noexcept = default;
    bool valid() const noexcept;
    runtime::AsyncOperationState state() const noexcept;
    bool observed() const noexcept;
    runtime::AsyncResource outputResource() const;
    GpuExecutionEvidence evidence() const noexcept;
    NativeGpuKernelResult observe() noexcept;
private:
    std::shared_ptr<NativeGpuKernelPendingState> state_;
    runtime::AsyncOperation operation_;
    PendingNativeGpuKernel(std::shared_ptr<NativeGpuKernelPendingState>, runtime::AsyncOperation);
    friend struct NativeGpuKernelSubmission;
    friend NativeGpuKernelSubmission submitNativeGpuKernelAsync(NativeGpuKernelRequest,
                                                                 int) noexcept;
};

struct NativeGpuKernelSubmission {
    std::optional<PendingNativeGpuKernel> pending;
    std::optional<GpuError> error;
    GpuExecutionEvidence evidence;
    bool ok() const noexcept { return pending.has_value() && !error.has_value(); }
};

NativeGpuKernelSubmission submitNativeGpuKernelAsync(NativeGpuKernelRequest request,
                                                      int device = 0) noexcept;

GpuAsyncSubmission submitNativeGpuAsync(const TensorRegion&,
                                        const std::vector<GpuValue>& inputs = {},
                                        int device = 0,
                                        PhysicalPlanOptions options = {}) noexcept;

// Executes a previously planned region with a previously generated PTX
// payload. The plan is independently verified and the payload is loaded
// verbatim; this path never calls the Thiran PTX emitter. The pending state
// owns both through observation/drain.
GpuAsyncSubmission submitNativeGpuPayloadAsync(const TensorRegion&,
                                               const PhysicalPlan&,
                                               std::string ptx,
                                               const std::vector<GpuValue>& inputs = {},
                                               int device = 0) noexcept;

// Synchronous native execution. Inputs cross an explicit host-to-device
// boundary and the result crosses an explicit device-to-host boundary before
// this function returns. It is exactly async submit followed by observation;
// there is no evaluator, native-CPU, framework, or retry fallback.
GpuExecutionResult executeNativeGpu(const TensorRegion&,
                                    const std::vector<GpuValue>& inputs = {},
                                    int device = 0,
                                    PhysicalPlanOptions options = {}) noexcept;
GpuExecutionResult executeNativeGpuPayload(const TensorRegion&,
                                           const PhysicalPlan&,
                                           std::string ptx,
                                           const std::vector<GpuValue>& inputs = {},
                                           int device = 0) noexcept;

}
