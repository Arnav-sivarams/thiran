#pragma once

#include "analysis/v0/Ownership.hpp"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace thiran::v0::storage {

struct StorageObjectId { std::uint64_t value = 0; bool operator==(const StorageObjectId&) const = default; };
enum class DType { Bool, U8, I32, I64, U32, U64, F32, F64, Invalid };
std::size_t elementWidth(DType);
std::string dtypeName(DType);
DType fromSemantic(semantic::TypeKind);

struct StorageObject;
class StorageHandle {
public:
    StorageHandle() = default;
    bool valid() const;
    StorageObjectId id() const;
    std::size_t byteLength() const;
private:
    explicit StorageHandle(std::shared_ptr<StorageObject> object);
    std::shared_ptr<StorageObject> object_;
    friend class Tensor;
    friend class MutableTensorRef;
    friend class HostBuffer;
    friend class MutableHostBufferRef;
    friend class ResourceStorageBridge;
    friend StorageHandle allocate(std::size_t);
};

// This is a verifiable runtime descriptor, deliberately not a public memory ABI.
struct TensorDescriptor {
    DType dtype = DType::Invalid;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> strides;
    std::uint64_t elementOffset = 0;
    StorageHandle storage;
    bool view = false;
    bool claimedContiguous = false;
};

std::uint64_t checkedAdd(std::uint64_t, std::uint64_t);
std::uint64_t checkedMultiply(std::uint64_t, std::uint64_t);
std::uint64_t checkedElementCount(const std::vector<std::uint64_t>&);
std::vector<std::uint64_t> checkedRowMajorStrides(const std::vector<std::uint64_t>&);
std::size_t checkedByteCount(std::uint64_t, DType);
std::uint64_t checkedMaximumOffset(const TensorDescriptor&);
bool isContiguousRowMajor(const TensorDescriptor&);
void verifyDescriptor(const TensorDescriptor&);
void verifySemanticShape(const semantic::Type&, const semantic::ShapeFact&, const TensorDescriptor&);

struct Slice { std::optional<std::uint64_t> start, end; std::int64_t step = 1; };
using Selector = std::variant<std::uint64_t, Slice>;

class Tensor {
public:
    explicit Tensor(TensorDescriptor descriptor);
    static Tensor materializeI64(std::vector<std::uint64_t> shape, const std::vector<std::int64_t>& values);
    static Tensor empty(DType dtype, std::vector<std::uint64_t> shape);
    const TensorDescriptor& descriptor() const { return descriptor_; }
    StorageObjectId storageId() const { return descriptor_.storage.id(); }
    bool isView() const { return descriptor_.view; }
    bool isContiguousRowMajor() const { return storage::isContiguousRowMajor(descriptor_); }
    std::int64_t loadI64(const std::vector<std::uint64_t>& indices) const;
    Tensor select(const std::vector<Selector>& selectors) const;
    Tensor transpose() const;
    Tensor reshapeView(const std::vector<std::uint64_t>& shape) const;
    Tensor reshapeCopy(const std::vector<std::uint64_t>& shape) const;
    Tensor deepCopy() const;
    std::vector<std::int64_t> logicalI64Values() const;
    std::string observe() const;
    std::string debugLayout() const;
private:
    TensorDescriptor descriptor_;
    std::uint64_t checkedLogicalOffset(const std::vector<std::uint64_t>& indices) const;
    friend class MutableTensorRef;
};

// Only lowering with a prior TH-006 exclusivity proof may obtain this primitive.
// It is intentionally move-only; handle counts do not grant mutation permission.
class MutableTensorRef {
public:
    explicit MutableTensorRef(Tensor& tensor) : tensor_(tensor) {}
    MutableTensorRef(const MutableTensorRef&) = delete;
    MutableTensorRef& operator=(const MutableTensorRef&) = delete;
    MutableTensorRef(MutableTensorRef&&) = default;
    void storeI64(const std::vector<std::uint64_t>& indices, std::int64_t value);
private:
    Tensor tensor_; // Retains bytes even if the C++ wrapper used to obtain it expires.
};

// Buffer is a semantically distinct one-dimensional collection, not a Tensor
// descriptor with rank one. It shares only the retained byte-storage substrate.
class HostBuffer {
public:
    static HostBuffer emptyU8();
    static HostBuffer materializeU8(const std::vector<std::uint8_t>& values);
    static HostBuffer allocateMetadata(DType dtype, std::uint64_t elementCount);
    DType dtype() const { return dtype_; }
    std::uint64_t elementCount() const { return elementCount_; }
    StorageObjectId storageId() const { return storage_.id(); }
    std::uint8_t loadU8(std::uint64_t index) const;
    HostBuffer deepCopy() const;
    std::string debug() const;
private:
    HostBuffer(DType dtype, std::uint64_t elementCount, StorageHandle storage);
    DType dtype_ = DType::Invalid;
    std::uint64_t elementCount_ = 0;
    StorageHandle storage_;
    friend class MutableHostBufferRef;
};

// Requires a prior TH-006 exclusivity proof. This low-level wrapper is not a
// safe-source mutable byte escape hatch and is deliberately move-only.
class MutableHostBufferRef {
public:
    explicit MutableHostBufferRef(HostBuffer& buffer) : buffer_(buffer) {}
    MutableHostBufferRef(const MutableHostBufferRef&) = delete;
    MutableHostBufferRef& operator=(const MutableHostBufferRef&) = delete;
    MutableHostBufferRef(MutableHostBufferRef&&) = default;
    MutableHostBufferRef& operator=(MutableHostBufferRef&&) = default;
    void storeU8(std::uint64_t index, std::uint8_t value);
private:
    HostBuffer buffer_;
};

std::vector<std::uint64_t> broadcastShape(const std::vector<std::uint64_t>&,
                                           const std::vector<std::uint64_t>&);
std::vector<std::uint64_t> matmulShape(const std::vector<std::uint64_t>&,
                                       const std::vector<std::uint64_t>&);
std::vector<std::uint64_t> sumShape(const std::vector<std::uint64_t>&, std::uint64_t axis);

// Conservative first baseline, not a ResourceId-to-allocation language invariant.
class ResourceStorageBridge {
public:
    explicit ResourceStorageBridge(const analysis::OwnershipAnalysisResult& facts);
    void bindFresh(analysis::ResourceId resource, const Tensor& tensor);
    Tensor resolve(const analysis::ResourceValue& value) const;
    Tensor bindCopy(const analysis::ResourceValue& copyFact, const analysis::ResourceValue& source);
    std::optional<StorageObjectId> mappedId(analysis::ResourceId resource) const;
private:
    std::map<analysis::ResourceId, analysis::ProvenanceKind> createdResources_;
    std::map<analysis::ResourceId, Tensor> mapped_;
};
}
