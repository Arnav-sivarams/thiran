#include "runtime/v0/Async.hpp"
#include "storage/v0/Storage.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace thiran::v0;
namespace runtime = thiran::v0::runtime;
namespace storage = thiran::v0::storage;

namespace {
int checks = 0;

void require(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

template<class F>
void contract(F&& action, const std::string& code) {
    try { action(); }
    catch (const runtime::AsyncContractError& error) {
        require(error.code() == code, "expected " + code + ", got " + error.code());
        return;
    }
    throw std::runtime_error("expected async contract error " + code);
}

struct Gate {
    bool complete = false;
    bool deferredFailure = false;
    bool immediateFailure = false;
    int submits = 0;
    int polls = 0;
    int waits = 0;
    int aborts = 0;
};

runtime::AsyncBackendCallbacks callbacks(const std::shared_ptr<Gate>& gate) {
    return {
        [gate]() -> std::optional<runtime::AsyncError> {
            ++gate->submits;
            if (gate->immediateFailure) return runtime::AsyncError{"TEST-SUBMIT", "induced submission failure"};
            return {};
        },
        [gate] {
            ++gate->polls;
            if (!gate->complete) return runtime::AsyncPollResult{};
            if (gate->deferredFailure)
                return runtime::AsyncPollResult{runtime::AsyncPollState::Failed,
                                                runtime::AsyncError{"TEST-DEFERRED", "induced deferred failure"}};
            return runtime::AsyncPollResult{runtime::AsyncPollState::Completed, {}};
        },
        [gate]() -> std::optional<runtime::AsyncError> {
            ++gate->waits;
            gate->complete = true;
            if (gate->deferredFailure) return runtime::AsyncError{"TEST-DEFERRED", "induced deferred failure"};
            return {};
        },
        [gate]() -> std::optional<runtime::AsyncError> { ++gate->aborts; gate->complete = true; return {}; }
    };
}

runtime::AsyncOperation submit(std::vector<runtime::AsyncResourceAccess> accesses,
                               const std::shared_ptr<Gate>& gate,
                               std::vector<runtime::AsyncDependency> dependencies = {}) {
    auto submitted = runtime::submitAsyncOperation(std::move(accesses), std::move(dependencies), callbacks(gate));
    require(submitted.ok(), submitted.error ? submitted.error->code : "async submission failed");
    return std::move(*submitted.operation);
}
}

int main() {
    try {
        (void)runtime::takeUnobservedAsyncErrors();

        auto tensor = storage::Tensor::materializeI64({2, 2}, {1, 2, 3, 4});
        auto alias = tensor;
        auto view = tensor.select({std::uint64_t{1}});
        auto copied = tensor.deepCopy();
        require(tensor.asyncResource() == alias.asyncResource() &&
                tensor.asyncResource() == view.asyncResource(),
                "aliases/views did not share the physical async root");
        require(!(tensor.asyncResource() == copied.asyncResource()),
                "explicit copy did not get an independent async root");

        auto readGate = std::make_shared<Gate>();
        auto read = submit({{view.asyncResource(), runtime::AsyncReservationKind::Read}}, readGate);
        require(read.state() == runtime::AsyncOperationState::Pending && readGate->submits == 1,
                "read operation did not enter pending state");
        require(tensor.asyncResource().activeReads() == 1 && tensor.asyncResource().activeWrites() == 0,
                "pending read reservation was not visible");
        require(alias.loadI64({0, 0}) == 1 && view.loadI64({0}) == 3,
                "immutable aliases were blocked by a read reservation");
        contract([&] { storage::MutableTensorRef mutation(alias); }, "ASYNC-MUTATION-PENDING");
        contract([&] { auto moved = std::move(alias); (void)moved; }, "ASYNC-MOVE-PENDING");
        contract([&] { tensor.asyncResource().requireDestroyable(); }, "ASYNC-DESTROY-PENDING");
        auto replacement = storage::Tensor::materializeI64({2, 2}, {8, 8, 8, 8});
        contract([&] { alias = replacement; }, "ASYNC-DESTROY-PENDING");
        storage::MutableTensorRef independentMutation(copied);
        independentMutation.storeI64({0, 0}, 99);
        require(copied.loadI64({0, 0}) == 99 && tensor.loadI64({0, 0}) == 1,
                "independent copied storage was falsely coupled");
        const auto readResult = read.observe();
        require(readResult.success && readGate->waits == 1 && tensor.asyncResource().activeReads() == 0,
                "successful observation did not release the read reservation");
        require(read.observed() && read.observe().success && readGate->waits == 1,
                "double observation was not deterministic");
        storage::MutableTensorRef mutationAfter(alias);
        mutationAfter.storeI64({0, 0}, 7);
        require(tensor.loadI64({0, 0}) == 7, "mutation remained blocked after observation");

        auto aliasWriteGate = std::make_shared<Gate>();
        auto aliasWrite = submit({{alias.asyncResource(), runtime::AsyncReservationKind::Write}}, aliasWriteGate);
        contract([&] { (void)view.loadI64({0}); }, "ASYNC-WRITE-PENDING");
        require(aliasWrite.observe().success && view.loadI64({0}) == 3,
                "view did not become readable after root write observation");

        auto output = runtime::AsyncResource::create();
        auto writeGate = std::make_shared<Gate>();
        auto write = submit({{output, runtime::AsyncReservationKind::Write}}, writeGate);
        require(output.activeWrites() == 1 && write.evidence().activeWriteReservations == 1,
                "pending write reservation was not visible");
        contract([&] { output.requireReadable(); }, "ASYNC-WRITE-PENDING");
        contract([&] { output.requireMutable(); }, "ASYNC-MUTATION-PENDING");
        auto conflictingRead = runtime::submitAsyncOperation(
            {{output, runtime::AsyncReservationKind::Read}}, {}, callbacks(std::make_shared<Gate>()));
        require(!conflictingRead.ok() && conflictingRead.error &&
                conflictingRead.error->code == "ASYNC-RESERVATION-CONFLICT",
                "write/read conflict was not rejected");
        auto conflictingWrite = runtime::submitAsyncOperation(
            {{output, runtime::AsyncReservationKind::Write}}, {}, callbacks(std::make_shared<Gate>()));
        require(!conflictingWrite.ok() && conflictingWrite.error &&
                conflictingWrite.error->code == "ASYNC-RESERVATION-CONFLICT",
                "write/write conflict was not rejected");
        writeGate->complete = true;
        require(write.state() == runtime::AsyncOperationState::Completed && output.activeWrites() == 1,
                "backend completion prematurely released an unobserved reservation");
        require(write.observe().success && output.activeWrites() == 0,
                "write observation did not publish/release the output");
        output.requireReadable();

        auto shared = runtime::AsyncResource::create();
        auto readOne = submit({{shared, runtime::AsyncReservationKind::Read}}, std::make_shared<Gate>());
        auto readTwo = submit({{shared, runtime::AsyncReservationKind::Read}}, std::make_shared<Gate>());
        require(shared.activeReads() == 2, "two immutable async reads did not coexist");
        require(readOne.observe().success && shared.activeReads() == 1,
                "observing one read released another operation's reservation");
        require(readTwo.observe().success && shared.activeReads() == 0,
                "second read reservation was not released");

        auto independentA = runtime::AsyncResource::create();
        auto independentB = runtime::AsyncResource::create();
        auto operationA = submit({{independentA, runtime::AsyncReservationKind::Read}}, std::make_shared<Gate>());
        auto operationB = submit({{independentB, runtime::AsyncReservationKind::Write}}, std::make_shared<Gate>());
        require(operationA.observe().success && independentA.activeReads() == 0 && independentB.activeWrites() == 1,
                "independent operation observation released an unrelated reservation");
        require(operationB.observe().success, "independent operation B failed");

        auto chainResource = runtime::AsyncResource::create();
        auto producerGate = std::make_shared<Gate>();
        auto consumerGate = std::make_shared<Gate>();
        auto producer = submit({{chainResource, runtime::AsyncReservationKind::Write}}, producerGate);
        auto consumer = submit({{chainResource, runtime::AsyncReservationKind::Read}}, consumerGate,
                               {producer.dependency()});
        require(producer.observed() && chainResource.activeWrites() == 0 &&
                chainResource.activeReads() == 1 && producerGate->waits == 1,
                "explicit producer dependency was not narrowly observed before consumer submission");
        require(consumer.observe().success && consumerGate->waits == 1 &&
                chainResource.activeReads() == 0 && chainResource.activeWrites() == 0,
                "producer/consumer observation order was incorrect");

        auto readThenWriteResource = runtime::AsyncResource::create();
        auto orderedRead = submit({{readThenWriteResource, runtime::AsyncReservationKind::Read}},
                                  std::make_shared<Gate>());
        auto orderedWrite = submit({{readThenWriteResource, runtime::AsyncReservationKind::Write}},
                                   std::make_shared<Gate>(), {orderedRead.dependency()});
        require(orderedRead.observed() && readThenWriteResource.activeReads() == 0 &&
                readThenWriteResource.activeWrites() == 1 && orderedWrite.observe().success,
                "read/write dependency was not ordered");

        auto writeThenWriteResource = runtime::AsyncResource::create();
        auto orderedWriteOne = submit({{writeThenWriteResource, runtime::AsyncReservationKind::Write}},
                                      std::make_shared<Gate>());
        auto orderedWriteTwo = submit({{writeThenWriteResource, runtime::AsyncReservationKind::Write}},
                                      std::make_shared<Gate>(), {orderedWriteOne.dependency()});
        require(orderedWriteOne.observed() && writeThenWriteResource.activeWrites() == 1 &&
                orderedWriteTwo.observe().success,
                "write/write dependency was not ordered");

        auto failedResource = runtime::AsyncResource::create();
        auto failedGate = std::make_shared<Gate>();
        failedGate->deferredFailure = true;
        auto failed = submit({{failedResource, runtime::AsyncReservationKind::Write}}, failedGate);
        const auto failure = failed.observe();
        require(!failure.success && failure.error && failure.error->code == "TEST-DEFERRED" &&
                failedResource.activeWrites() == 0,
                "deferred failure did not surface and release reservations");
        require(!failed.observe().success && failedGate->waits == 1,
                "failed double observation did not return the cached failure");

        auto dependencyFailureResource = runtime::AsyncResource::create();
        auto dependencyFailureGate = std::make_shared<Gate>();
        dependencyFailureGate->deferredFailure = true;
        auto failingProducer = submit({{dependencyFailureResource, runtime::AsyncReservationKind::Write}},
                                      dependencyFailureGate);
        auto skippedConsumerGate = std::make_shared<Gate>();
        auto skippedConsumer = runtime::submitAsyncOperation(
            {{dependencyFailureResource, runtime::AsyncReservationKind::Read}},
            {failingProducer.dependency()}, callbacks(skippedConsumerGate));
        require(!skippedConsumer.ok() && skippedConsumer.error &&
                skippedConsumer.error->code == "ASYNC-DEPENDENCY-FAILED" &&
                failingProducer.observed() && skippedConsumerGate->submits == 0 &&
                dependencyFailureResource.activeReads() == 0 &&
                dependencyFailureResource.activeWrites() == 0,
                "failed dependency started its consumer or leaked reservations");

        auto immediateResource = runtime::AsyncResource::create();
        auto immediateGate = std::make_shared<Gate>();
        immediateGate->immediateFailure = true;
        auto immediate = runtime::submitAsyncOperation(
            {{immediateResource, runtime::AsyncReservationKind::Read}}, {}, callbacks(immediateGate));
        require(!immediate.ok() && immediate.error && immediate.error->code == "TEST-SUBMIT" &&
                immediateGate->aborts == 1 && immediateResource.activeReads() == 0,
                "immediate failure leaked a reservation or skipped safe abort");

        auto malformed = runtime::submitAsyncOperation(
            {{runtime::AsyncResource{}, runtime::AsyncReservationKind::Read}}, {}, callbacks(std::make_shared<Gate>()));
        require(!malformed.ok() && malformed.error && malformed.error->code == "ASYNC-INVALID-RESOURCE",
                "malformed reservation was not rejected");
        auto empty = runtime::submitAsyncOperation({}, {}, callbacks(std::make_shared<Gate>()));
        require(!empty.ok() && empty.error && empty.error->code == "ASYNC-EMPTY-RESERVATION",
                "empty reservation set was not rejected");
        runtime::AsyncOperation invalidHandle;
        const auto invalidObservation = invalidHandle.observe();
        require(!invalidObservation.success && invalidObservation.error &&
                invalidObservation.error->code == "ASYNC-INVALID-HANDLE" && !invalidHandle.observed(),
                "invalid async handle observation was not deterministic");
        auto invalidDependencyResource = runtime::AsyncResource::create();
        auto invalidDependency = runtime::submitAsyncOperation(
            {{invalidDependencyResource, runtime::AsyncReservationKind::Read}},
            {runtime::AsyncDependency{}}, callbacks(std::make_shared<Gate>()));
        require(!invalidDependency.ok() && invalidDependency.error &&
                invalidDependency.error->code == "ASYNC-INVALID-DEPENDENCY" &&
                invalidDependencyResource.activeReads() == 0,
                "invalid dependency acquired a reservation");

        auto droppedResource = runtime::AsyncResource::create();
        auto droppedGate = std::make_shared<Gate>();
        {
            auto dropped = submit({{droppedResource, runtime::AsyncReservationKind::Read}}, droppedGate);
            require(droppedResource.activeReads() == 1, "drop fixture never became pending");
        }
        require(droppedGate->waits == 1 && droppedResource.activeReads() == 0,
                "unobserved successful work was not drained on drop");

        runtime::AsyncOperation retainedStorageOperation;
        runtime::AsyncResource retainedStorageRoot;
        auto retainedStorageGate = std::make_shared<Gate>();
        {
            auto temporary = storage::Tensor::materializeI64({1}, {42});
            retainedStorageRoot = temporary.asyncResource();
            retainedStorageOperation = submit(
                {{retainedStorageRoot, runtime::AsyncReservationKind::Read}}, retainedStorageGate);
        }
        require(retainedStorageRoot.activeReads() == 1 && retainedStorageOperation.observe().success,
                "dropping the last tensor wrapper invalidated its pending storage reservation");

        auto droppedFailureResource = runtime::AsyncResource::create();
        auto droppedFailureGate = std::make_shared<Gate>();
        droppedFailureGate->deferredFailure = true;
        {
            auto dropped = submit({{droppedFailureResource, runtime::AsyncReservationKind::Write}}, droppedFailureGate);
            (void)dropped;
        }
        const auto unobserved = runtime::takeUnobservedAsyncErrors();
        require(unobserved.size() == 1 && unobserved[0].code == "TEST-DEFERRED" &&
                droppedFailureResource.activeWrites() == 0,
                "dropped deferred failure disappeared or retained reservations");

        auto sameInput = runtime::AsyncResource::create();
        auto duplicateRead = submit({{sameInput, runtime::AsyncReservationKind::Read},
                                     {sameInput, runtime::AsyncReservationKind::Read}}, std::make_shared<Gate>());
        require(sameInput.activeReads() == 1 && duplicateRead.evidence().activeReadReservations == 1,
                "same input supplied twice created duplicate physical reservations");
        require(duplicateRead.observe().success, "duplicate-read operation failed");

        for (std::size_t index = 0; index < 500; ++index) {
            auto stressResource = runtime::AsyncResource::create();
            auto stressGate = std::make_shared<Gate>();
            {
                auto stress = submit({{stressResource, index % 3 == 0 ?
                    runtime::AsyncReservationKind::Write : runtime::AsyncReservationKind::Read}},
                    stressGate);
                if (index % 2 == 0)
                    require(stress.observe().success, "host async stress observation failed");
            }
            require(stressResource.activeReads() == 0 && stressResource.activeWrites() == 0,
                    "host async stress leaked a reservation");
        }

        auto evidence = runtime::asyncRuntimeEvidence();
        require(evidence.submitted >= 519 && evidence.observations >= 519 &&
                evidence.droppedDrains >= 252 && evidence.unobservedFailures >= 1,
                "device-neutral async evidence counters are incomplete");

        std::cout << "V0AsyncLifetimeTests PASS " << checks << " checks\n"
                  << "submitted=" << evidence.submitted
                  << " observations=" << evidence.observations
                  << " dropped_drains=" << evidence.droppedDrains
                  << " released_reservations=" << evidence.releasedReservations
                  << " unobserved_failures=" << evidence.unobservedFailures << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V0AsyncLifetimeTests FAIL: " << error.what() << '\n';
        return 1;
    }
}
