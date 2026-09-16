#include "storage/v0/Storage.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <limits>
#include <type_traits>

using namespace thiran::v0::storage;
template<class F> void error(F&& fn, const char* id) {
    try { fn(); } catch (const std::runtime_error& e) { assert(std::string(e.what()) == id); return; }
    assert(false);
}
int main() {
    static_assert(!std::is_same_v<HostBuffer, Tensor>); // BUF15
    static_assert(!std::is_copy_constructible_v<MutableHostBufferRef>); // BUF12
    auto zero = HostBuffer::emptyU8(); // BUF01
    assert(zero.dtype() == DType::U8 && zero.elementCount() == 0);
    error([&] { zero.loadU8(0); }, "TH009-BUFFER-BOUNDS"); // BUF05
    auto bytes = HostBuffer::materializeU8({0, 7, 255}); // BUF02
    assert(bytes.elementCount() == 3); // BUF03
    assert(bytes.loadU8(0) == 0 && bytes.loadU8(1) == 7 && bytes.loadU8(2) == 255); // BUF04
    error([&] { bytes.loadU8(3); }, "TH009-BUFFER-BOUNDS");
    auto alias = bytes; // BUF06
    auto copy = bytes.deepCopy(); // BUF07
    assert(alias.storageId() == bytes.storageId() && copy.storageId() != bytes.storageId());
    MutableHostBufferRef write(bytes);
    write.storeU8(1, 42);
    assert(alias.loadU8(1) == 42); // BUF08, BUF10
    assert(copy.loadU8(1) == 7); // BUF09
    auto retained = [] { auto root = HostBuffer::materializeU8({9}); return root; }();
    assert(retained.loadU8(0) == 9); // BUF11
    error([&] { HostBuffer::allocateMetadata(DType::I64, std::numeric_limits<std::uint64_t>::max()); },
          "TH009-BUFFER-SIZE-OVERFLOW"); // BUF13
    const auto first = bytes.debug(), second = bytes.debug();
    assert(first == second && first.find("0x") == std::string::npos); // BUF14
    auto tensor = Tensor::materializeI64({3}, {0, 42, 255});
    assert(tensor.storageId() != bytes.storageId() && tensor.descriptor().shape.size() == 1); // BUF15
    std::cout << "V0BufferTests PASS BUF01-BUF16\n";
}
