#include "storage/v0/Storage.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace thiran::v0::storage {
namespace {
[[noreturn]] void fail(const char* id) { throw std::runtime_error(id); }
std::uint64_t nextStorageId = 1;
std::uint64_t storageElements(const TensorDescriptor& d) {
    const auto width = elementWidth(d.dtype);
    if (!d.storage.valid() || d.storage.byteLength() % width != 0) fail("TH007-DESCRIPTOR");
    return static_cast<std::uint64_t>(d.storage.byteLength() / width);
}
void visitIndices(const std::vector<std::uint64_t>& shape,
                  const std::function<void(const std::vector<std::uint64_t>&)>& visit) {
    if (checkedElementCount(shape) == 0) return;
    std::vector<std::uint64_t> index(shape.size(), 0);
    for (;;) {
        visit(index);
        if (shape.empty()) break;
        std::size_t axis = shape.size();
        while (axis != 0) {
            --axis;
            ++index[axis]; // index < extent on entry, so this increment cannot overflow.
            if (index[axis] < shape[axis]) break;
            index[axis] = 0;
        }
        if (axis == 0 && index[0] == 0) break;
    }
}
}

struct StorageObject { StorageObjectId id; std::vector<std::byte> bytes; };
StorageHandle::StorageHandle(std::shared_ptr<StorageObject> object) : object_(std::move(object)) {}
bool StorageHandle::valid() const { return static_cast<bool>(object_); }
StorageObjectId StorageHandle::id() const { if (!object_) fail("TH007-DESCRIPTOR"); return object_->id; }
std::size_t StorageHandle::byteLength() const { if (!object_) fail("TH007-DESCRIPTOR"); return object_->bytes.size(); }
StorageHandle allocate(std::size_t bytes) {
    if (nextStorageId == std::numeric_limits<std::uint64_t>::max()) fail("TH007-STORAGE-ID-OVERFLOW");
    auto object = std::make_shared<StorageObject>();
    object->id = {nextStorageId++};
    object->bytes.resize(bytes);
    return StorageHandle(std::move(object));
}
std::size_t elementWidth(DType dtype) {
    switch (dtype) {
    case DType::Bool: return 1;
    case DType::I32: case DType::U32: case DType::F32: return 4;
    case DType::I64: case DType::U64: case DType::F64: return 8;
    default: fail("TH007-DTYPE");
    }
}
std::string dtypeName(DType dtype) {
    switch (dtype) {
    case DType::Bool: return "bool"; case DType::I32: return "i32";
    case DType::I64: return "i64"; case DType::U32: return "u32";
    case DType::U64: return "u64"; case DType::F32: return "f32";
    case DType::F64: return "f64"; default: fail("TH007-DTYPE");
    }
}
DType fromSemantic(semantic::TypeKind kind) {
    switch (kind) {
    case semantic::TypeKind::Bool: return DType::Bool;
    case semantic::TypeKind::I32: return DType::I32;
    case semantic::TypeKind::I64: return DType::I64;
    case semantic::TypeKind::U32: return DType::U32;
    case semantic::TypeKind::U64: return DType::U64;
    case semantic::TypeKind::F32: return DType::F32;
    case semantic::TypeKind::F64: return DType::F64;
    default: fail("TH007-DTYPE");
    }
}
std::uint64_t checkedAdd(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) fail("TH007-SIZE-OVERFLOW");
    return a + b;
}
std::uint64_t checkedMultiply(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) fail("TH007-SIZE-OVERFLOW");
    return a * b;
}
std::uint64_t checkedElementCount(const std::vector<std::uint64_t>& shape) {
    std::uint64_t count = 1;
    for (auto extent : shape) count = checkedMultiply(count, extent);
    return count;
}
std::vector<std::uint64_t> checkedRowMajorStrides(const std::vector<std::uint64_t>& shape) {
    std::vector<std::uint64_t> strides(shape.size(), 1);
    std::uint64_t stride = 1;
    for (std::size_t i = shape.size(); i != 0;) {
        --i;
        strides[i] = stride;
        stride = checkedMultiply(stride, std::max<std::uint64_t>(shape[i], 1));
    }
    return strides;
}
std::size_t checkedByteCount(std::uint64_t count, DType dtype) {
    const auto bytes = checkedMultiply(count, elementWidth(dtype));
    if (bytes > std::numeric_limits<std::size_t>::max()) fail("TH007-SIZE-OVERFLOW");
    return static_cast<std::size_t>(bytes);
}
std::uint64_t checkedMaximumOffset(const TensorDescriptor& d) {
    if (d.shape.size() != d.strides.size()) fail("TH007-DESCRIPTOR");
    auto maximum = d.elementOffset;
    if (checkedElementCount(d.shape) == 0) return maximum;
    for (std::size_t i = 0; i < d.shape.size(); ++i)
        maximum = checkedAdd(maximum, checkedMultiply(d.shape[i] - 1, d.strides[i]));
    return maximum;
}
bool isContiguousRowMajor(const TensorDescriptor& d) {
    if (d.shape.size() != d.strides.size()) return false;
    // No logical element is reachable when an axis is empty. Any positive layout is vacuously contiguous.
    if (checkedElementCount(d.shape) == 0) return true;
    std::uint64_t expected = 1;
    for (std::size_t i = d.shape.size(); i != 0;) {
        --i;
        if (d.shape[i] > 1 && d.strides[i] != expected) return false;
        expected = checkedMultiply(expected, d.shape[i]);
    }
    return true;
}
void verifyDescriptor(const TensorDescriptor& d) {
    const auto width = elementWidth(d.dtype);
    if (!d.storage.valid() || d.shape.size() != d.strides.size()) fail("TH007-DESCRIPTOR");
    for (auto stride : d.strides) if (stride == 0) fail("TH007-DESCRIPTOR");
    const auto count = checkedElementCount(d.shape);
    for (auto extent : d.shape)
        if (extent > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            fail("TH-SPEC-SHAPE");
    const auto elements = storageElements(d);
    if (d.claimedContiguous && !isContiguousRowMajor(d)) fail("TH007-DESCRIPTOR");
    if (count == 0) {
        // Empty offsets may denote a one-past cursor; they never authorize a dereference.
        if (d.elementOffset > elements) fail("TH007-DESCRIPTOR");
    } else if (checkedMaximumOffset(d) >= elements) fail("TH007-DESCRIPTOR");
    if (!d.view) {
        if (d.elementOffset != 0 || !isContiguousRowMajor(d) ||
            d.storage.byteLength() != checkedByteCount(count, d.dtype)) fail("TH007-DESCRIPTOR");
    }
    (void)width;
}
void verifySemanticShape(const semantic::Type& type, const semantic::ShapeFact& fact,
                         const TensorDescriptor& d) {
    verifyDescriptor(d);
    if (type.kind != semantic::TypeKind::Tensor || type.elements.size() != 1 ||
        type.rank != d.shape.size() || fact.extents.size() != type.rank ||
        fromSemantic(type.elements[0].kind) != d.dtype) fail("TH-SPEC-SHAPE");
    for (std::size_t i = 0; i < d.shape.size(); ++i) {
        if (d.shape[i] > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
            (fact.extents[i] && (*fact.extents[i] < 0 ||
              static_cast<std::uint64_t>(*fact.extents[i]) != d.shape[i]))) fail("TH-SPEC-SHAPE");
    }
}
Tensor::Tensor(TensorDescriptor descriptor) : descriptor_(std::move(descriptor)) { verifyDescriptor(descriptor_); }
Tensor Tensor::materializeI64(std::vector<std::uint64_t> shape, const std::vector<std::int64_t>& values) {
    auto count = checkedElementCount(shape);
    if (count != values.size()) fail("TH-SPEC-SHAPE");
    TensorDescriptor d{DType::I64, shape, checkedRowMajorStrides(shape), 0,
                       allocate(checkedByteCount(count, DType::I64)), false, true};
    if (!values.empty()) std::memcpy(d.storage.object_->bytes.data(), values.data(), d.storage.byteLength());
    return Tensor(std::move(d));
}
Tensor Tensor::empty(DType dtype, std::vector<std::uint64_t> shape) {
    if (checkedElementCount(shape) != 0) fail("TH-SPEC-SHAPE");
    TensorDescriptor d{dtype, shape, checkedRowMajorStrides(shape), 0, allocate(0), false, true};
    return Tensor(std::move(d));
}
std::uint64_t Tensor::checkedLogicalOffset(const std::vector<std::uint64_t>& indices) const {
    verifyDescriptor(descriptor_);
    if (indices.size() != descriptor_.shape.size()) fail("TH-SPEC-BOUNDS");
    auto offset = descriptor_.elementOffset;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= descriptor_.shape[i]) fail("TH-SPEC-BOUNDS");
        offset = checkedAdd(offset, checkedMultiply(indices[i], descriptor_.strides[i]));
    }
    if (offset >= storageElements(descriptor_)) fail("TH-SPEC-BOUNDS");
    return offset;
}
std::int64_t Tensor::loadI64(const std::vector<std::uint64_t>& indices) const {
    if (descriptor_.dtype != DType::I64) fail("TH007-DTYPE-EXECUTION-DEFERRED");
    const auto offset = checkedLogicalOffset(indices);
    std::int64_t value;
    std::memcpy(&value, descriptor_.storage.object_->bytes.data() + checkedByteCount(offset, DType::I64), sizeof(value));
    return value;
}
void MutableTensorRef::storeI64(const std::vector<std::uint64_t>& indices, std::int64_t value) {
    if (tensor_.descriptor_.dtype != DType::I64) fail("TH007-DTYPE-EXECUTION-DEFERRED");
    const auto offset = tensor_.checkedLogicalOffset(indices);
    std::memcpy(tensor_.descriptor_.storage.object_->bytes.data() + checkedByteCount(offset, DType::I64),
                &value, sizeof(value));
}
Tensor Tensor::select(const std::vector<Selector>& selectors) const {
    verifyDescriptor(descriptor_);
    if (selectors.size() > descriptor_.shape.size()) fail("TH-SPEC-BOUNDS");
    auto d = descriptor_;
    d.view = true;
    d.claimedContiguous = false;
    d.shape.clear(); d.strides.clear();
    for (std::size_t i = 0; i < descriptor_.shape.size(); ++i) {
        const auto extent = descriptor_.shape[i], stride = descriptor_.strides[i];
        if (i < selectors.size() && std::holds_alternative<std::uint64_t>(selectors[i])) {
            const auto index = std::get<std::uint64_t>(selectors[i]);
            if (index >= extent) fail("TH-SPEC-BOUNDS");
            d.elementOffset = checkedAdd(d.elementOffset, checkedMultiply(index, stride));
        } else {
            Slice slice = i < selectors.size() ? std::get<Slice>(selectors[i]) : Slice{};
            if (slice.step <= 0) fail("TH-SPEC-SLICE");
            const auto start = slice.start.value_or(0), end = slice.end.value_or(extent);
            if (start > end || end > extent) fail("TH-SPEC-SLICE");
            const auto step = static_cast<std::uint64_t>(slice.step);
            const auto length = end - start;
            const auto outputExtent = length == 0 ? 0 : 1 + (length - 1) / step;
            d.elementOffset = checkedAdd(d.elementOffset, checkedMultiply(start, stride));
            d.shape.push_back(outputExtent);
            d.strides.push_back(checkedMultiply(stride, step));
        }
    }
    // Empty descriptors have no accessible address: canonicalize a computed cursor
    // that may pass the allocation after selecting a different nonempty axis.
    if (checkedElementCount(d.shape) == 0 && d.elementOffset > storageElements(d))
        d.elementOffset = storageElements(d);
    return Tensor(std::move(d));
}
Tensor Tensor::transpose() const {
    if (descriptor_.shape.size() != 2) fail("TH-SPEC-SHAPE");
    auto d = descriptor_;
    std::swap(d.shape[0], d.shape[1]);
    std::swap(d.strides[0], d.strides[1]);
    d.view = true; d.claimedContiguous = false;
    return Tensor(std::move(d));
}
Tensor Tensor::reshapeView(const std::vector<std::uint64_t>& shape) const {
    if (checkedElementCount(shape) != checkedElementCount(descriptor_.shape)) fail("TH-SPEC-SHAPE");
    if (!isContiguousRowMajor()) fail("TH-SPEC-SHAPE");
    auto d = descriptor_;
    d.shape = shape; d.strides = checkedRowMajorStrides(shape);
    d.view = true; d.claimedContiguous = true;
    return Tensor(std::move(d));
}
Tensor Tensor::reshapeCopy(const std::vector<std::uint64_t>& shape) const {
    if (checkedElementCount(shape) != checkedElementCount(descriptor_.shape)) fail("TH-SPEC-SHAPE");
    return Tensor::materializeI64(shape, logicalI64Values());
}
std::vector<std::int64_t> Tensor::logicalI64Values() const {
    if (descriptor_.dtype != DType::I64) fail("TH007-DTYPE-EXECUTION-DEFERRED");
    std::vector<std::int64_t> values;
    values.reserve(static_cast<std::size_t>(checkedElementCount(descriptor_.shape)));
    visitIndices(descriptor_.shape, [&](const auto& index) { values.push_back(loadI64(index)); });
    return values;
}
Tensor Tensor::deepCopy() const { return Tensor::materializeI64(descriptor_.shape, logicalI64Values()); }
std::string Tensor::observe() const {
    std::ostringstream out;
    out << (descriptor_.view ? "view" : "materialized") << " " << dtypeName(descriptor_.dtype) << " shape=[";
    for (std::size_t i = 0; i < descriptor_.shape.size(); ++i) { if (i) out << ','; out << descriptor_.shape[i]; }
    out << "]";
    if (descriptor_.dtype == DType::I64) {
        out << " values=[";
        auto values = logicalI64Values();
        for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
        out << ']';
    }
    return out.str();
}
std::string Tensor::debugLayout() const {
    std::ostringstream out;
    out << observe() << " storage=" << storageId().value << " offset=" << descriptor_.elementOffset << " strides=[";
    for (std::size_t i = 0; i < descriptor_.strides.size(); ++i) { if (i) out << ','; out << descriptor_.strides[i]; }
    out << "] contiguous=" << (isContiguousRowMajor() ? "true" : "false");
    return out.str();
}
std::vector<std::uint64_t> broadcastShape(const std::vector<std::uint64_t>& a,
                                           const std::vector<std::uint64_t>& b) {
    const auto rank = std::max(a.size(), b.size());
    std::vector<std::uint64_t> output(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        const auto x = i < a.size() ? a[a.size() - 1 - i] : 1;
        const auto y = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (x != y && x != 1 && y != 1) fail("TH-SPEC-BROADCAST");
        output[rank - 1 - i] = x == 1 ? y : x;
    }
    return output;
}
std::vector<std::uint64_t> matmulShape(const std::vector<std::uint64_t>& a,
                                       const std::vector<std::uint64_t>& b) {
    if (a.size() != 2 || b.size() != 2 || a[1] != b[0]) fail("TH-SPEC-SHAPE");
    return {a[0], b[1]};
}
std::vector<std::uint64_t> sumShape(const std::vector<std::uint64_t>& a, std::uint64_t axis) {
    if (axis >= a.size()) fail("TH-SPEC-AXIS");
    auto output = a;
    output.erase(output.begin() + static_cast<std::ptrdiff_t>(axis));
    return output;
}
ResourceStorageBridge::ResourceStorageBridge(const analysis::OwnershipAnalysisResult& facts)
    : createdResources_(facts.createdResources) {
    if (!facts.ok()) fail("TH007-BRIDGE-INVALID-FACTS");
}
void ResourceStorageBridge::bindFresh(analysis::ResourceId resource, const Tensor& tensor) {
    const auto it = createdResources_.find(resource);
    if (it == createdResources_.end() || it->second != analysis::ProvenanceKind::Fresh ||
        mapped_.contains(resource)) fail("TH007-BRIDGE-RESOURCE");
    for (const auto& [other, existing] : mapped_) {
        (void)other;
        if (existing.storageId() == tensor.storageId()) fail("TH007-BRIDGE-STORAGE-COLLISION");
    }
    mapped_.emplace(resource, tensor);
}
Tensor ResourceStorageBridge::resolve(const analysis::ResourceValue& value) const {
    if (value.kind == analysis::ProvenanceKind::NoResource || value.resources.size() != 1)
        fail("TH007-BRIDGE-RESOURCE");
    const auto it = mapped_.find(*value.resources.begin());
    if (it == mapped_.end()) fail("TH007-BRIDGE-RESOURCE");
    return it->second;
}
Tensor ResourceStorageBridge::bindCopy(const analysis::ResourceValue& copyFact,
                                       const analysis::ResourceValue& source) {
    if (copyFact.kind != analysis::ProvenanceKind::IndependentCopyOf ||
        copyFact.resources.size() != 1 || source.resources.size() != 1 ||
        copyFact.copySources != source.resources ||
        copyFact.resources == source.resources) fail("TH007-BRIDGE-RESOURCE");
    const auto copyResource = *copyFact.resources.begin();
    const auto it = createdResources_.find(copyResource);
    if (it == createdResources_.end() || it->second != analysis::ProvenanceKind::IndependentCopyOf ||
        mapped_.contains(copyResource)) fail("TH007-BRIDGE-RESOURCE");
    auto copy = resolve(source).deepCopy();
    mapped_.emplace(copyResource, copy);
    return copy;
}
std::optional<StorageObjectId> ResourceStorageBridge::mappedId(analysis::ResourceId resource) const {
    const auto it = mapped_.find(resource);
    if (it == mapped_.end()) return std::nullopt;
    return it->second.storageId();
}
}
