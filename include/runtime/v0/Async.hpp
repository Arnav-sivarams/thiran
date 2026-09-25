#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace thiran::v0::storage { class StorageHandle; }

namespace thiran::v0::runtime {

using AsyncOperationId = std::uint64_t;
using AsyncResourceId = std::uint64_t;

enum class AsyncReservationKind { Read, Write };
enum class AsyncOperationState { Submitted, Pending, Completed, Failed, Observed };
enum class AsyncPollState { Pending, Completed, Failed };

struct AsyncError {
    std::string code;
    std::string message;
    bool operator==(const AsyncError&) const = default;
};

class AsyncContractError final : public std::runtime_error {
public:
    AsyncContractError(std::string code, std::string message);
    const std::string& code() const noexcept { return code_; }
private:
    std::string code_;
};

struct AsyncResourceState;

// A device-neutral physical-resource root. Copies identify the same resource;
// a storage-backed instance also retains the storage object while reserved.
class AsyncResource {
public:
    AsyncResource() = default;
    static AsyncResource create();
    bool valid() const noexcept;
    AsyncResourceId id() const noexcept;
    std::uint64_t activeReads() const noexcept;
    std::uint64_t activeWrites() const noexcept;
    void requireReadable() const;
    void requireMutable() const;
    void requireMovable() const;
    void requireDestroyable() const;
    bool operator==(const AsyncResource& other) const noexcept;
    // Opaque implementation identity used by the runtime reservation engine.
    const std::shared_ptr<AsyncResourceState>& stateHandle() const noexcept { return state_; }
private:
    std::shared_ptr<AsyncResourceState> state_;
    std::shared_ptr<void> retention_;
    explicit AsyncResource(std::shared_ptr<AsyncResourceState> state,
                           std::shared_ptr<void> retention = {});
    friend class thiran::v0::storage::StorageHandle;
};

struct AsyncResourceAccess {
    AsyncResource resource;
    AsyncReservationKind kind = AsyncReservationKind::Read;
};

struct AsyncPollResult {
    AsyncPollState state = AsyncPollState::Pending;
    std::optional<AsyncError> error;
};

// submit is called only after reservations have been acquired and declared
// dependencies have been narrowly observed. If it returns an error, abort is
// called before reservations are released. wait must not return until the
// backend no longer references any reserved resource.
struct AsyncBackendCallbacks {
    std::function<std::optional<AsyncError>()> submit;
    std::function<AsyncPollResult()> poll;
    std::function<std::optional<AsyncError>()> wait;
    std::function<std::optional<AsyncError>()> abort;
};

struct AsyncOperationEvidence {
    std::uint64_t submitted = 0;
    std::uint64_t pending = 0;
    std::uint64_t completed = 0;
    std::uint64_t failed = 0;
    std::uint64_t observations = 0;
    std::uint64_t droppedDrains = 0;
    std::uint64_t activeReadReservations = 0;
    std::uint64_t activeWriteReservations = 0;
    std::uint64_t releasedReservations = 0;
};

struct AsyncRuntimeEvidence {
    std::uint64_t submitted = 0;
    std::uint64_t observations = 0;
    std::uint64_t droppedDrains = 0;
    std::uint64_t releasedReservations = 0;
    std::uint64_t unobservedFailures = 0;
};

struct AsyncOperationResult {
    bool success = false;
    std::optional<AsyncError> error;
};

struct AsyncOperationImpl;
struct AsyncSubmission;

class AsyncDependency {
public:
    AsyncDependency() = default;
    bool valid() const noexcept;
    AsyncOperationId operationId() const noexcept;
private:
    explicit AsyncDependency(std::shared_ptr<AsyncOperationImpl> operation);
    std::shared_ptr<AsyncOperationImpl> operation_;
    friend class AsyncOperation;
    friend AsyncSubmission submitAsyncOperation(std::vector<AsyncResourceAccess>,
                                                std::vector<AsyncDependency>,
                                                AsyncBackendCallbacks) noexcept;
};

class AsyncOperation {
public:
    AsyncOperation() = default;
    ~AsyncOperation();
    AsyncOperation(const AsyncOperation&) = delete;
    AsyncOperation& operator=(const AsyncOperation&) = delete;
    AsyncOperation(AsyncOperation&&) noexcept;
    AsyncOperation& operator=(AsyncOperation&&) noexcept;

    bool valid() const noexcept;
    AsyncOperationId id() const noexcept;
    AsyncOperationState state() const noexcept;
    bool observed() const noexcept;
    AsyncDependency dependency() const;
    AsyncOperationEvidence evidence() const noexcept;
    AsyncOperationResult observe() noexcept;

private:
    explicit AsyncOperation(std::shared_ptr<AsyncOperationImpl> operation);
    std::shared_ptr<AsyncOperationImpl> operation_;
    void drainOnDrop() noexcept;
    friend struct AsyncSubmission;
    friend AsyncSubmission submitAsyncOperation(std::vector<AsyncResourceAccess>,
                                                std::vector<AsyncDependency>,
                                                AsyncBackendCallbacks) noexcept;
};

struct AsyncSubmission {
    std::optional<AsyncOperation> operation;
    std::optional<AsyncError> error;
    bool ok() const noexcept { return operation.has_value() && !error.has_value(); }
};

AsyncSubmission submitAsyncOperation(std::vector<AsyncResourceAccess> accesses,
                                     std::vector<AsyncDependency> dependencies,
                                     AsyncBackendCallbacks callbacks) noexcept;

// Dropped work is synchronously drained. A deferred failure discovered by that
// drain is retained here until the enclosing runtime explicitly consumes it.
std::vector<AsyncError> takeUnobservedAsyncErrors();
AsyncRuntimeEvidence asyncRuntimeEvidence() noexcept;

}
