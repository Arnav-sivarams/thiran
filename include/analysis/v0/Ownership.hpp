#pragma once

#include "semantic/v0/Analyzer.hpp"
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace thiran::v0::analysis {
using ResourceId = std::uint64_t;
using ViewId = std::uint64_t;
using semantic::BindingId;
using semantic::ValueId;
using semantic::FunctionId;

enum class ProvenanceKind { NoResource, Fresh, AliasOf, ReadViewOf, IndependentCopyOf, PossibleAlias };
struct ResourceValue {
    ProvenanceKind kind = ProvenanceKind::NoResource;
    std::set<ResourceId> resources;
    std::set<ResourceId> copySources;
    std::set<BindingId> viewRoots;
    BindingId sourceBinding = 0;
    ViewId view = 0;
    std::vector<ResourceValue> elements;
    bool operator==(const ResourceValue&) const = default;
};
struct BindingFact {
    enum class Availability { DefinitelyAvailable, DefinitelyMoved, MaybeUnavailable } availability = Availability::DefinitelyAvailable;
    ResourceValue value;
    bool mutableBinding = false;
    SourceSpan declaration;
    SourceSpan moveSite;
    bool operator==(const BindingFact& other) const {
        return availability==other.availability && value==other.value && mutableBinding==other.mutableBinding &&
            declaration.begin.offset==other.declaration.begin.offset && moveSite.begin.offset==other.moveSite.begin.offset;
    }
};
struct ViewFact { ViewId id = 0; std::set<ResourceId> resources; std::set<BindingId> roots; SourceSpan span; };
struct BorrowFact { FunctionId function = 0; ValueId call = 0; BindingId binding = 0;
    std::set<ResourceId> resources; std::set<BindingId> liveConflicts; SourceSpan span; };
struct MoveFact { FunctionId function = 0; ValueId instruction = 0; BindingId binding = 0;
    BindingFact::Availability availabilityAfter = BindingFact::Availability::DefinitelyMoved; SourceSpan span; };
enum class EffectKind : std::uint32_t { MayTrap = 1, Mutates = 2, RNG = 4, IO = 8, Transfer = 16, Async = 32 };
using EffectSet = std::uint32_t; // zero is Pure
struct FunctionEffects { EffectSet kinds = 0; std::set<std::size_t> mutatedParameters;
    bool pureTensorCandidate = true; bool operator==(const FunctionEffects&) const = default; };
struct OwnershipAnalysisResult {
    std::vector<semantic::SemanticDiagnostic> diagnostics;
    std::map<FunctionId,std::map<BindingId,BindingFact>> bindingAvailability;
    std::map<FunctionId,std::map<ValueId,ResourceValue>> valueProvenance;
    std::map<ResourceId,ProvenanceKind> createdResources;
    std::map<ViewId,ViewFact> views;
    std::vector<BorrowFact> mutableBorrows;
    std::vector<MoveFact> moves;
    std::map<FunctionId,FunctionEffects> functionEffects;
    std::map<FunctionId,std::map<ValueId,std::set<BindingId>>> liveAfter;
    bool ok() const { return diagnostics.empty(); }
    std::string dump() const;
};
OwnershipAnalysisResult analyze(const semantic::Module& module);
std::vector<std::string> auditFacts(const semantic::Module&, const OwnershipAnalysisResult&);
}
