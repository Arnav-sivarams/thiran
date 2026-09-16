#include "storage/v0/Storage.hpp"
// Keep mechanical test assertions active in Release configurations.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

using namespace thiran::v0;
using namespace thiran::v0::storage;

template<class F> void error(F&& fn, const char* id) {
    try { fn(); } catch (const std::runtime_error& e) {
        if (std::string(e.what()) != id) std::cerr << "expected " << id << " got " << e.what() << '\n';
        assert(std::string(e.what()) == id); return;
    }
    assert(false);
}
template<class F> void rejected(F&& fn) {
    try { fn(); } catch (const std::runtime_error&) { return; }
    assert(false);
}

int main(int argc, char** argv) {
    static_assert(!std::is_same_v<analysis::ResourceId, StorageObjectId>);
    static_assert(!std::is_copy_constructible_v<MutableTensorRef>);
    assert(elementWidth(DType::Bool) == 1 && elementWidth(DType::I32) == 4 &&
           elementWidth(DType::I64) == 8 && elementWidth(DType::F64) == 8);

    auto v = Tensor::materializeI64({3}, {4,5,6}); // ST01, ST03
    assert((v.descriptor().strides == std::vector<std::uint64_t>{1}));
    assert(v.loadI64({2}) == 6);
    auto a = Tensor::materializeI64({2,3}, {1,2,3,4,5,6}); // ST02
    assert((a.descriptor().strides == std::vector<std::uint64_t>{3,1}));
    assert((checkedRowMajorStrides({2,3,4}) == std::vector<std::uint64_t>{12,4,1}));
    assert(a.isContiguousRowMajor() && a.descriptor().elementOffset == 0);
    assert(a.loadI64({1,2}) == 6); // ST08

    auto alias = a; // ST04: a C++ handle alias has no tensor-data allocation.
    assert(alias.storageId() == a.storageId());
    auto independent = a.deepCopy(); // ST05, ST06
    assert(independent.storageId() != a.storageId());
    assert(independent.logicalI64Values() == a.logicalI64Values());
    auto row = a.select({std::uint64_t{1}}); // ST07
    assert(row.isView() && row.storageId() == a.storageId());
    assert((row.descriptor().shape == std::vector<std::uint64_t>{3}));
    assert((row.descriptor().strides == std::vector<std::uint64_t>{1}));
    assert(row.descriptor().elementOffset == 3 && row.loadI64({2}) == 6);
    assert(row.isContiguousRowMajor()); // ST28: contiguous despite nonzero offset.

    auto slice = a.select({Slice{0,2,1}, Slice{1,3,1}}); // ST09
    assert(slice.isView() && slice.storageId() == a.storageId());
    assert((slice.descriptor().shape == std::vector<std::uint64_t>{2,2}));
    assert((slice.logicalI64Values() == std::vector<std::int64_t>{2,3,5,6}));
    auto stepped = a.select({Slice{0,2,1}, Slice{0,3,2}}); // ST10
    assert((stepped.descriptor().strides == std::vector<std::uint64_t>{3,2}));
    assert((stepped.logicalI64Values() == std::vector<std::int64_t>{1,3,4,6}));
    auto mixed = a.select({std::uint64_t{1}, Slice{1,3,1}}); // ST11
    assert(mixed.descriptor().elementOffset == 4 &&
           (mixed.descriptor().shape == std::vector<std::uint64_t>{2}) &&
           (mixed.descriptor().strides == std::vector<std::uint64_t>{1}));
    assert((mixed.logicalI64Values() == std::vector<std::int64_t>{5,6}));

    auto transposed = a.transpose(); // ST12, ST13
    assert(transposed.isView() && transposed.storageId() == a.storageId());
    assert((transposed.descriptor().shape == std::vector<std::uint64_t>{3,2}));
    assert((transposed.descriptor().strides == std::vector<std::uint64_t>{1,3}));
    assert(!transposed.isContiguousRowMajor());
    assert((transposed.logicalI64Values() == std::vector<std::int64_t>{1,4,2,5,3,6}));
    auto chained = transposed.select({std::uint64_t{1}, Slice{}}); // ST14
    assert(chained.storageId() == a.storageId() && chained.descriptor().elementOffset == 1);
    assert((chained.descriptor().strides == std::vector<std::uint64_t>{3}));
    assert((chained.logicalI64Values() == std::vector<std::int64_t>{2,5}));
    auto chained2 = a.select({Slice{0,2,2}, Slice{}}).transpose();
    assert(chained2.storageId() == a.storageId());
    assert((chained2.descriptor().strides == std::vector<std::uint64_t>{1,6}));
    auto tcopy = transposed.deepCopy(); // ST15
    assert(tcopy.storageId() != a.storageId() && tcopy.isContiguousRowMajor());
    assert((tcopy.descriptor().strides == std::vector<std::uint64_t>{2,1}));
    assert(tcopy.logicalI64Values() == transposed.logicalI64Values());

    MutableTensorRef legalWrite(a); // ST16-ST19: test-only, prior exclusivity proof assumed.
    legalWrite.storeI64({1,0}, 99);
    assert(alias.loadI64({1,0}) == 99 && row.loadI64({0}) == 99);
    assert(transposed.loadI64({0,1}) == 99);
    assert(independent.loadI64({1,0}) == 4 && tcopy.loadI64({0,1}) == 4);
    assert(alias.storageId() == a.storageId()); // No COW.

    auto z1 = Tensor::empty(DType::I64, {0}); // ST20, ST21
    auto z2 = Tensor::empty(DType::I64, {0,3});
    auto z3 = Tensor::empty(DType::I64, {4,0});
    assert(z1.descriptor().storage.byteLength() == 0 && z2.isContiguousRowMajor());
    assert(z3.select({std::uint64_t{1}}).logicalI64Values().empty());
    assert(z2.transpose().logicalI64Values().empty());
    error([&] { z1.loadI64({0}); }, "TH-SPEC-BOUNDS");
    error([&] { a.loadI64({2,0}); }, "TH-SPEC-BOUNDS"); // ST25
    error([&] { a.loadI64({0}); }, "TH-SPEC-BOUNDS");
    error([&] { a.select({Slice{0,2,0}}); }, "TH-SPEC-SLICE"); // ST26
    error([&] { a.select({Slice{0,2,-1}}); }, "TH-SPEC-SLICE");
    error([&] { a.select({Slice{0,3,1}}); }, "TH-SPEC-SLICE");

    const auto max = std::numeric_limits<std::uint64_t>::max();
    error([&] { checkedElementCount({max,2}); }, "TH007-SIZE-OVERFLOW"); // ST22
    error([&] { checkedByteCount(max, DType::I64); }, "TH007-SIZE-OVERFLOW"); // ST23
    error([&] { checkedRowMajorStrides({2,max,2}); }, "TH007-SIZE-OVERFLOW"); // ST24
    error([&] { checkedAdd(max,1); }, "TH007-SIZE-OVERFLOW");

    auto d = a.descriptor(); // ST27, SV01-SV06, SV09: manual malformed descriptors.
    d.strides.pop_back(); rejected([&] { verifyDescriptor(d); }); // SV01
    d = a.descriptor(); d.strides[1] = 0; rejected([&] { verifyDescriptor(d); }); // SV02
    d = a.descriptor(); d.elementOffset = 7; rejected([&] { verifyDescriptor(d); }); // SV03
    d = a.descriptor(); d.view = true; d.strides[0] = 6; rejected([&] { verifyDescriptor(d); }); // SV04
    d = a.descriptor(); d.view = true; d.claimedContiguous = false;
    d.elementOffset = max; d.strides[0] = max;
    error([&] { verifyDescriptor(d); }, "TH007-SIZE-OVERFLOW"); // SV05
    d = a.descriptor(); d.dtype = DType::Invalid; rejected([&] { verifyDescriptor(d); }); // SV06
    d = a.descriptor(); d.view = true; d.storage = {}; rejected([&] { verifyDescriptor(d); }); // SV09
    d = a.descriptor(); d.view = true; d.claimedContiguous = true; d.strides = {1,3};
    rejected([&] { verifyDescriptor(d); });

    auto semanticType = semantic::tensor(semantic::scalar(semantic::TypeKind::I64), 2);
    semantic::ShapeFact known{{2,3}}, unknown{{std::nullopt,std::nullopt}};
    verifySemanticShape(semanticType, known, a.descriptor()); // ST shape bridge
    verifySemanticShape(semanticType, unknown, a.descriptor());
    auto wrongRank = semantic::tensor(semantic::scalar(semantic::TypeKind::I64), 1);
    error([&] { verifySemanticShape(wrongRank, {{3}}, a.descriptor()); }, "TH-SPEC-SHAPE"); // SV07
    error([&] { verifySemanticShape(semanticType, {{2,4}}, a.descriptor()); }, "TH-SPEC-SHAPE"); // SV08

    assert((broadcastShape({2,1}, {1,3}) == std::vector<std::uint64_t>{2,3})); // ST29
    assert((broadcastShape({0,1}, {1,3}) == std::vector<std::uint64_t>{0,3}));
    assert((broadcastShape({}, {2,3}) == std::vector<std::uint64_t>{2,3}));
    assert((broadcastShape({0}, {1}) == std::vector<std::uint64_t>{0}));
    error([&] { broadcastShape({2}, {3}); }, "TH-SPEC-BROADCAST"); // ST30
    assert((matmulShape({2,3}, {3,4}) == std::vector<std::uint64_t>{2,4})); // ST31
    error([&] { matmulShape({2,3}, {4,2}); }, "TH-SPEC-SHAPE"); // ST32
    assert((sumShape({2,3},0) == std::vector<std::uint64_t>{3})); // ST33
    assert((sumShape({2,3},1) == std::vector<std::uint64_t>{2}));
    assert(sumShape({3},0).empty() && (sumShape({4,0},1) == std::vector<std::uint64_t>{4}));
    error([&] { sumShape({2,3},2); }, "TH-SPEC-AXIS");

    auto reshaped = a.reshapeView({3,2});
    assert(reshaped.isView() && reshaped.storageId() == a.storageId());
    error([&] { transposed.reshapeView({6}); }, "TH-SPEC-SHAPE");
    auto reshapeCopy = transposed.reshapeCopy({6});
    assert(reshapeCopy.storageId() != a.storageId() &&
           reshapeCopy.logicalI64Values() == transposed.logicalI64Values());

    analysis::OwnershipAnalysisResult facts;
    facts.createdResources.emplace(101, analysis::ProvenanceKind::Fresh);
    facts.createdResources.emplace(102, analysis::ProvenanceKind::IndependentCopyOf);
    facts.createdResources.emplace(103, analysis::ProvenanceKind::Fresh);
    ResourceStorageBridge bridge(facts); // ST34, ST35
    bridge.bindFresh(101, a);
    analysis::ResourceValue aliasFact{analysis::ProvenanceKind::AliasOf}; aliasFact.resources.insert(101);
    analysis::ResourceValue viewFact{analysis::ProvenanceKind::ReadViewOf}; viewFact.resources.insert(101);
    assert(bridge.resolve(aliasFact).storageId() == a.storageId());
    assert(bridge.resolve(viewFact).storageId() == row.storageId());
    analysis::ResourceValue copyFact{analysis::ProvenanceKind::IndependentCopyOf};
    copyFact.resources.insert(102); copyFact.copySources.insert(101);
    auto bridgeCopy = bridge.bindCopy(copyFact, aliasFact);
    assert(bridgeCopy.storageId() != a.storageId());
    auto forgedCopy = copyFact; forgedCopy.copySources = {103};
    rejected([&] { bridge.bindCopy(forgedCopy, aliasFact); });
    bridge.bindFresh(103, v);
    assert(bridge.mappedId(103) != bridge.mappedId(101));
    analysis::ResourceValue scalarFact;
    rejected([&] { bridge.resolve(scalarFact); }); // No tensor storage for scalar.
    analysis::ResourceValue stale{analysis::ProvenanceKind::AliasOf}; stale.resources.insert(999);
    rejected([&] { bridge.resolve(stale); }); // SV10
    rejected([&] { bridge.bindFresh(999, v); });
    rejected([&] { bridge.bindFresh(101, v); });
    auto bridgeAfterFacts = [] {
        analysis::OwnershipAnalysisResult local;
        local.createdResources.emplace(201, analysis::ProvenanceKind::Fresh);
        return ResourceStorageBridge(local);
    }();
    bridgeAfterFacts.bindFresh(201, Tensor::materializeI64({1}, {8}));
    assert(bridgeAfterFacts.mappedId(201).has_value());

    auto mutableAfterWrapper = [] {
        auto local = Tensor::materializeI64({1}, {8});
        return MutableTensorRef(local);
    }();
    mutableAfterWrapper.storeI64({0}, 42); // Runtime wrapper retained bytes; source proof remains separate.

    Tensor retained = [&] { auto root = Tensor::materializeI64({2,2}, {7,8,9,10});
                            return root.select({std::uint64_t{1}}); }();
    assert(retained.loadI64({0}) == 9); // ST36: physical retention; source escape remains illegal.
    const auto first = a.debugLayout();
    const auto second = a.debugLayout();
    assert(first == second && first.find("0x") == std::string::npos);
    if (argc == 2 && std::string(argv[1]) == "--dump") std::cout << first << '\n';
    std::cout << "V0StorageTests PASS ST01-ST36 SV01-SV10\n";
}
