#include "runtime/v0/Async.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <utility>

namespace thiran::v0::runtime {
namespace {

struct Reservation {
    AsyncOperationId operation = 0;
    AsyncReservationKind kind = AsyncReservationKind::Read;
};

std::mutex resourceMutex;
std::atomic<std::uint64_t> nextResource{1};
std::atomic<std::uint64_t> nextOperation{1};
std::atomic<std::uint64_t> submittedCount{0};
std::atomic<std::uint64_t> observationCount{0};
std::atomic<std::uint64_t> droppedDrainCount{0};
std::atomic<std::uint64_t> releasedReservationCount{0};
std::atomic<std::uint64_t> unobservedFailureCount{0};
std::mutex unobservedMutex;
std::vector<AsyncError> unobservedErrors;

AsyncError contractError(std::string code, std::string message) {
    return {std::move(code), std::move(message)};
}

bool conflicts(AsyncReservationKind a, AsyncReservationKind b) {
    return a == AsyncReservationKind::Write || b == AsyncReservationKind::Write;
}

void recordUnobserved(const AsyncError& error) {
    std::lock_guard lock(unobservedMutex);
    unobservedErrors.push_back(error);
    ++unobservedFailureCount;
}

std::optional<AsyncError> callError(const std::function<std::optional<AsyncError>()>& callback,
                                    const char* missingCode) noexcept {
    if (!callback) return contractError(missingCode, "async backend callback is missing");
    try { return callback(); }
    catch (const AsyncContractError& error) { return contractError(error.code(), error.what()); }
    catch (const std::exception& error) { return contractError("ASYNC-BACKEND-EXCEPTION", error.what()); }
    catch (...) { return contractError("ASYNC-BACKEND-EXCEPTION", "unknown async backend exception"); }
}

}

struct AsyncResourceState {
    AsyncResourceId id = 0;
    std::vector<Reservation> reservations;
};

struct AsyncOperationImpl {
    AsyncOperationId id = 0;
    mutable std::mutex mutex;
    std::condition_variable condition;
    AsyncOperationState state = AsyncOperationState::Submitted;
    bool observing = false;
    bool observed = false;
    bool released = false;
    AsyncOperationResult outcome;
    AsyncBackendCallbacks callbacks;
    std::vector<AsyncResourceAccess> accesses;
    std::vector<std::shared_ptr<AsyncOperationImpl>> dependencies;
    std::set<AsyncOperationId> dependencyClosure;
    AsyncOperationEvidence evidence;
};

namespace {

void collectDependencyIds(const std::shared_ptr<AsyncOperationImpl>& operation,
                          std::set<AsyncOperationId>& ids) {
    if (!operation || !ids.insert(operation->id).second) return;
    for (const auto& dependency : operation->dependencies) collectDependencyIds(dependency, ids);
}

void releaseReservations(const std::shared_ptr<AsyncOperationImpl>& operation) noexcept {
    if (!operation) return;
    std::lock_guard resources(resourceMutex);
    {
        std::lock_guard state(operation->mutex);
        if (operation->released) return;
        operation->released = true;
    }
    std::uint64_t released = 0;
    for (const auto& access : operation->accesses) {
        if (!access.resource.stateHandle()) continue;
        auto& entries = access.resource.stateHandle()->reservations;
        const auto old = entries.size();
        entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const Reservation& reservation) {
            return reservation.operation == operation->id;
        }), entries.end());
        released += old - entries.size();
    }
    operation->accesses.clear();
    {
        std::lock_guard state(operation->mutex);
        operation->evidence.releasedReservations += released;
        operation->evidence.activeReadReservations = 0;
        operation->evidence.activeWriteReservations = 0;
    }
    releasedReservationCount += released;
}

AsyncOperationResult observeImpl(const std::shared_ptr<AsyncOperationImpl>& operation) noexcept {
    if (!operation) return {false, contractError("ASYNC-INVALID-HANDLE", "async handle is invalid")};
    std::optional<AsyncError> knownFailure;
    {
        std::unique_lock lock(operation->mutex);
        while (operation->observing) operation->condition.wait(lock);
        if (operation->observed) return operation->outcome;
        if (operation->state == AsyncOperationState::Failed) knownFailure = operation->outcome.error;
        operation->observing = true;
    }

    std::optional<AsyncError> failure;
    for (const auto& dependency : operation->dependencies) {
        const auto result = observeImpl(dependency);
        if (!result.success) {
            failure = result.error.value_or(contractError("ASYNC-DEPENDENCY-FAILED", "async dependency failed"));
            break;
        }
    }
    const auto waitFailure = callError(operation->callbacks.wait, "ASYNC-WAIT-MISSING");
    if (!failure) failure = knownFailure ? knownFailure : waitFailure;

    releaseReservations(operation);
    AsyncOperationResult result{!failure.has_value(), failure};
    {
        std::lock_guard lock(operation->mutex);
        operation->outcome = result;
        operation->observed = true;
        operation->observing = false;
        operation->state = AsyncOperationState::Observed;
        operation->evidence.pending = 0;
        operation->evidence.completed = result.success ? 1 : 0;
        operation->evidence.failed = result.success ? 0 : 1;
        ++operation->evidence.observations;
        operation->dependencies.clear();
        operation->dependencyClosure.clear();
    }
    ++observationCount;
    operation->condition.notify_all();
    return result;
}

void acquireReservations(const std::shared_ptr<AsyncOperationImpl>& operation) {
    if (operation->accesses.empty())
        throw AsyncContractError("ASYNC-EMPTY-RESERVATION", "async operation has no resource reservation");

    std::map<AsyncResourceId, AsyncResourceAccess> normalized;
    for (const auto& access : operation->accesses) {
        if (!access.resource.valid())
            throw AsyncContractError("ASYNC-INVALID-RESOURCE", "async reservation references an invalid resource");
        auto [it, inserted] = normalized.emplace(access.resource.id(), access);
        if (!inserted && access.kind == AsyncReservationKind::Write)
            it->second.kind = AsyncReservationKind::Write;
    }
    operation->accesses.clear();
    for (auto& [id, access] : normalized) {
        (void)id;
        operation->accesses.push_back(std::move(access));
    }

    std::lock_guard lock(resourceMutex);
    for (const auto& access : operation->accesses) {
        for (const auto& reservation : access.resource.stateHandle()->reservations) {
            if (conflicts(access.kind, reservation.kind) &&
                !operation->dependencyClosure.contains(reservation.operation))
                throw AsyncContractError("ASYNC-RESERVATION-CONFLICT",
                    "async reservation conflicts with operation " + std::to_string(reservation.operation));
        }
    }
    for (const auto& access : operation->accesses) {
        access.resource.stateHandle()->reservations.push_back({operation->id, access.kind});
        if (access.kind == AsyncReservationKind::Read) ++operation->evidence.activeReadReservations;
        else ++operation->evidence.activeWriteReservations;
    }
}

}

AsyncContractError::AsyncContractError(std::string code, std::string message)
    : std::runtime_error(std::move(message)), code_(std::move(code)) {}

AsyncResource::AsyncResource(std::shared_ptr<AsyncResourceState> state, std::shared_ptr<void> retention)
    : state_(std::move(state)), retention_(std::move(retention)) {}

AsyncResource AsyncResource::create() {
    const auto id = nextResource.fetch_add(1);
    if (id == 0) throw AsyncContractError("ASYNC-RESOURCE-ID-OVERFLOW", "async resource identity exhausted");
    auto state = std::make_shared<AsyncResourceState>();
    state->id = id;
    return AsyncResource(std::move(state));
}

bool AsyncResource::valid() const noexcept { return state_ && state_->id != 0; }
AsyncResourceId AsyncResource::id() const noexcept { return state_ ? state_->id : 0; }

std::uint64_t AsyncResource::activeReads() const noexcept {
    if (!state_) return 0;
    std::lock_guard lock(resourceMutex);
    return static_cast<std::uint64_t>(std::count_if(state_->reservations.begin(), state_->reservations.end(),
        [](const Reservation& reservation) { return reservation.kind == AsyncReservationKind::Read; }));
}

std::uint64_t AsyncResource::activeWrites() const noexcept {
    if (!state_) return 0;
    std::lock_guard lock(resourceMutex);
    return static_cast<std::uint64_t>(std::count_if(state_->reservations.begin(), state_->reservations.end(),
        [](const Reservation& reservation) { return reservation.kind == AsyncReservationKind::Write; }));
}

void AsyncResource::requireReadable() const {
    if (!state_) throw AsyncContractError("ASYNC-INVALID-RESOURCE", "read references an invalid async resource");
    std::lock_guard lock(resourceMutex);
    if (std::any_of(state_->reservations.begin(), state_->reservations.end(),
                    [](const Reservation& reservation) { return reservation.kind == AsyncReservationKind::Write; }))
        throw AsyncContractError("ASYNC-WRITE-PENDING", "completed-value read conflicts with a pending async writer");
}

void AsyncResource::requireMutable() const {
    if (!state_) throw AsyncContractError("ASYNC-INVALID-RESOURCE", "mutation references an invalid async resource");
    std::lock_guard lock(resourceMutex);
    if (!state_->reservations.empty())
        throw AsyncContractError("ASYNC-MUTATION-PENDING", "mutation conflicts with an async reservation");
}

void AsyncResource::requireMovable() const {
    if (!state_) throw AsyncContractError("ASYNC-INVALID-RESOURCE", "move references an invalid async resource");
    std::lock_guard lock(resourceMutex);
    if (!state_->reservations.empty())
        throw AsyncContractError("ASYNC-MOVE-PENDING", "move conflicts with an async reservation");
}

void AsyncResource::requireDestroyable() const {
    if (!state_) throw AsyncContractError("ASYNC-INVALID-RESOURCE", "destroy references an invalid async resource");
    std::lock_guard lock(resourceMutex);
    if (!state_->reservations.empty())
        throw AsyncContractError("ASYNC-DESTROY-PENDING", "destruction conflicts with an async reservation");
}

bool AsyncResource::operator==(const AsyncResource& other) const noexcept { return state_ == other.state_; }

AsyncDependency::AsyncDependency(std::shared_ptr<AsyncOperationImpl> operation) : operation_(std::move(operation)) {}
bool AsyncDependency::valid() const noexcept { return static_cast<bool>(operation_); }
AsyncOperationId AsyncDependency::operationId() const noexcept { return operation_ ? operation_->id : 0; }

AsyncOperation::AsyncOperation(std::shared_ptr<AsyncOperationImpl> operation) : operation_(std::move(operation)) {}
AsyncOperation::~AsyncOperation() { drainOnDrop(); }
AsyncOperation::AsyncOperation(AsyncOperation&& other) noexcept : operation_(std::move(other.operation_)) {}
AsyncOperation& AsyncOperation::operator=(AsyncOperation&& other) noexcept {
    if (this != &other) { drainOnDrop(); operation_ = std::move(other.operation_); }
    return *this;
}
bool AsyncOperation::valid() const noexcept { return static_cast<bool>(operation_); }
AsyncOperationId AsyncOperation::id() const noexcept { return operation_ ? operation_->id : 0; }

AsyncOperationState AsyncOperation::state() const noexcept {
    if (!operation_) return AsyncOperationState::Failed;
    {
        std::lock_guard lock(operation_->mutex);
        if (operation_->observed || operation_->observing || operation_->state != AsyncOperationState::Pending)
            return operation_->state;
    }
    if (!operation_->callbacks.poll) return AsyncOperationState::Pending;
    AsyncPollResult poll;
    try { poll = operation_->callbacks.poll(); }
    catch (const std::exception& error) {
        poll = {AsyncPollState::Failed, contractError("ASYNC-POLL-EXCEPTION", error.what())};
    } catch (...) {
        poll = {AsyncPollState::Failed, contractError("ASYNC-POLL-EXCEPTION", "unknown async poll exception")};
    }
    std::lock_guard lock(operation_->mutex);
    if (operation_->state != AsyncOperationState::Pending) return operation_->state;
    if (poll.state == AsyncPollState::Completed) {
        operation_->state = AsyncOperationState::Completed;
        operation_->evidence.pending = 0;
        operation_->evidence.completed = 1;
    } else if (poll.state == AsyncPollState::Failed) {
        operation_->state = AsyncOperationState::Failed;
        operation_->outcome = {false, poll.error.value_or(contractError("ASYNC-POLL-FAILED", "async poll failed"))};
        operation_->evidence.pending = 0;
        operation_->evidence.failed = 1;
    }
    return operation_->state;
}

bool AsyncOperation::observed() const noexcept {
    if (!operation_) return false;
    std::lock_guard lock(operation_->mutex);
    return operation_->observed;
}

AsyncDependency AsyncOperation::dependency() const { return AsyncDependency(operation_); }

AsyncOperationEvidence AsyncOperation::evidence() const noexcept {
    if (!operation_) return {};
    std::lock_guard lock(operation_->mutex);
    return operation_->evidence;
}

AsyncOperationResult AsyncOperation::observe() noexcept { return observeImpl(operation_); }

void AsyncOperation::drainOnDrop() noexcept {
    if (!operation_) return;
    bool needsDrain = false;
    {
        std::lock_guard lock(operation_->mutex);
        needsDrain = !operation_->observed;
        if (needsDrain) ++operation_->evidence.droppedDrains;
    }
    if (needsDrain) {
        ++droppedDrainCount;
        const auto result = observeImpl(operation_);
        if (!result.success && result.error) recordUnobserved(*result.error);
    }
    operation_.reset();
}

AsyncSubmission submitAsyncOperation(std::vector<AsyncResourceAccess> accesses,
                                     std::vector<AsyncDependency> dependencies,
                                     AsyncBackendCallbacks callbacks) noexcept {
    try {
    auto operation = std::make_shared<AsyncOperationImpl>();
    operation->id = nextOperation.fetch_add(1);
    operation->callbacks = std::move(callbacks);
    operation->accesses = std::move(accesses);
    operation->evidence.submitted = 1;
    if (operation->id == 0)
        return {{}, contractError("ASYNC-OPERATION-ID-OVERFLOW", "async operation identity exhausted")};
    for (const auto& dependency : dependencies) {
        if (!dependency.operation_)
            return {{}, contractError("ASYNC-INVALID-DEPENDENCY", "async dependency handle is invalid")};
        if (dependency.operation_->id == operation->id)
            return {{}, contractError("ASYNC-DEPENDENCY-CYCLE", "async operation depends on itself")};
        operation->dependencies.push_back(dependency.operation_);
        collectDependencyIds(dependency.operation_, operation->dependencyClosure);
    }

    for (const auto& dependency : operation->dependencies) {
        const auto dependencyResult = observeImpl(dependency);
        if (!dependencyResult.success) {
            if (dependencyResult.error)
                return {{}, contractError("ASYNC-DEPENDENCY-FAILED",
                    "async dependency failed: " + dependencyResult.error->code + ": " +
                    dependencyResult.error->message)};
            return {{}, contractError("ASYNC-DEPENDENCY-FAILED", "async dependency failed")};
        }
    }

    try { acquireReservations(operation); }
    catch (const AsyncContractError& error) { return {{}, contractError(error.code(), error.what())}; }
    catch (const std::exception& error) { return {{}, contractError("ASYNC-RESERVATION-FAILURE", error.what())}; }

    const auto submissionError = callError(operation->callbacks.submit, "ASYNC-SUBMIT-MISSING");
    if (submissionError) {
        const auto abortError = callError(operation->callbacks.abort, "ASYNC-ABORT-MISSING");
        releaseReservations(operation);
        if (abortError)
            return {{}, contractError(submissionError->code,
                submissionError->message + "; safe abort also reported: " + abortError->message)};
        return {{}, submissionError};
    }

    {
        std::lock_guard lock(operation->mutex);
        operation->state = AsyncOperationState::Pending;
        operation->evidence.pending = 1;
    }
    ++submittedCount;
    return {std::optional<AsyncOperation>(AsyncOperation(std::move(operation))), {}};
    } catch (const AsyncContractError& error) {
        return {{}, contractError(error.code(), error.what())};
    } catch (const std::exception& error) {
        return {{}, contractError("ASYNC-SUBMISSION-FAILURE", error.what())};
    } catch (...) {
        return {{}, contractError("ASYNC-SUBMISSION-FAILURE", "unknown async submission failure")};
    }
}

std::vector<AsyncError> takeUnobservedAsyncErrors() {
    std::lock_guard lock(unobservedMutex);
    auto result = std::move(unobservedErrors);
    unobservedErrors.clear();
    return result;
}

AsyncRuntimeEvidence asyncRuntimeEvidence() noexcept {
    return {submittedCount.load(), observationCount.load(), droppedDrainCount.load(),
            releasedReservationCount.load(), unobservedFailureCount.load()};
}

}
