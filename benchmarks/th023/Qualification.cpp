#include "CppBaseline.hpp"

#include "artifact/v0/NativeArtifacts.hpp"
#include "backend/v0/NativeCpu.hpp"
#include "extension/v0/Extension.hpp"
#include "model/v0/ReferenceModel.hpp"
#include "persistence/v0/Persistence.hpp"
#include "tooling/v0/BuildConfig.hpp"
#include "tooling/v0/Driver.hpp"
#include "training/v0/Checkpoint.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <variant>
#include <vector>

namespace fs = std::filesystem;
namespace artifact = thiran::v0::artifact;
namespace backend = thiran::v0::backend;
namespace extension = thiran::v0::extension;
namespace model = thiran::v0::model;
namespace persistence = thiran::v0::persistence;
namespace semantic = thiran::v0::semantic;
namespace storage = thiran::v0::storage;
namespace tooling = thiran::v0::tooling;
namespace training = thiran::v0::training;
using namespace thiran::bench::th023;

namespace {

constexpr std::string_view kCommit = "7722a58f260ca9993ab335ffc9dbbc22512371a0";
constexpr std::size_t kSamples = 30;
constexpr std::size_t kCpuWarmups = 5;
constexpr std::size_t kGpuWarmups = 10;

const std::string kElementwise = R"(
fn bench(x: Tensor<f32,1>, y: Tensor<f32,1>) -> Tensor<f32,1> {
  let a = x + y
  let b = -a
  let c = b .* y
  let d = c - x
  let e = d .* y
  return e
})";

const std::string kExtension = R"(
fn bench(x: Tensor<f32,1>, y: Tensor<f32,1>) -> Tensor<f32,1> {
  let a = -x
  let b = research_square_linear(a)
  let c = b + y
  return c
})";

const std::string kExtensionBuiltIn = R"(
fn bench(x: Tensor<f32,1>, y: Tensor<f32,1>) -> Tensor<f32,1> {
  let a = -x
  let s = a .* a
  let b = s + a
  let c = b + y
  return c
})";

const std::string kMaterialized = R"(
fn bench(x: Tensor<f32,1>, y: Tensor<f32,1>) -> Tensor<f32,1> {
  let a = x + y
  let b = -a
  let c = a .* y
  let d = b + c
  let e = d - x
  return e
})";

struct TemporaryDirectory {
    fs::path path;
    explicit TemporaryDirectory(std::string prefix) {
        std::string pattern = "/tmp/" + prefix + "-XXXXXX";
        if (char* result = ::mkdtemp(pattern.data())) path = result;
        else throw std::runtime_error("mkdtemp failed");
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

struct Statistics {
    double median = 0;
    double mean = 0;
    double stddev = 0;
    double p10 = 0;
    double p90 = 0;
    double minimum = 0;
    double maximum = 0;
};

struct Record {
    std::string benchmarkName;
    std::string implementation;
    std::string backendName;
    std::uint64_t size = 0;
    std::size_t warmups = 0;
    std::size_t batch = 1;
    std::string timingScope;
    std::vector<double> samplesNs;
    std::string inputDigest;
    std::string outputDigest;
    std::map<std::string, std::uint64_t> counters;
    std::map<std::string, std::string> labels;
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void require(bool condition, const std::string& message) { if (!condition) fail(message); }

std::string json(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (character < 0x20) out << "\\u" << std::hex << std::setw(4)
                                      << std::setfill('0') << static_cast<unsigned>(character)
                                      << std::dec << std::setfill(' ');
            else out << static_cast<char>(character);
        }
    }
    return out.str() + '"';
}

Statistics statistics(const std::vector<double>& samples) {
    require(!samples.empty(), "statistics require samples");
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
        const double position = p * static_cast<double>(sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(std::floor(position));
        const auto upper = static_cast<std::size_t>(std::ceil(position));
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    };
    Statistics result;
    result.median = percentile(0.5);
    result.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                  static_cast<double>(samples.size());
    double squared = 0;
    for (const double sample : samples) squared += (sample - result.mean) * (sample - result.mean);
    result.stddev = std::sqrt(squared / static_cast<double>(samples.size()));
    result.p10 = percentile(0.1);
    result.p90 = percentile(0.9);
    result.minimum = sorted.front();
    result.maximum = sorted.back();
    return result;
}

std::string digest(const std::vector<float>& values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const float value : values) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned shift = 0; shift != 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(bits >> shift);
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream out;
    out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

std::string inputDigest(const std::vector<float>& x, const std::vector<float>& y) {
    std::vector<float> combined;
    combined.reserve(x.size() + y.size());
    combined.insert(combined.end(), x.begin(), x.end());
    combined.insert(combined.end(), y.begin(), y.end());
    return digest(combined);
}

std::pair<std::vector<float>, std::vector<float>> inputs(std::size_t count) {
    std::vector<float> x(count), y(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto xi = static_cast<std::int64_t>((i * 17) % 1009) - 504;
        const auto yi = static_cast<std::int64_t>((i * 29 + 11) % 1013) - 506;
        x[i] = static_cast<float>(xi) / 37.0f;
        y[i] = static_cast<float>(yi) / 41.0f;
    }
    return {std::move(x), std::move(y)};
}

std::vector<float> elementwiseReference(const std::vector<float>& x,
                                        const std::vector<float>& y) {
    return cppFused(x, y);
}

std::vector<float> extensionReference(const std::vector<float>& x,
                                      const std::vector<float>& y) {
    std::vector<float> result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float a = -x[i];
        const float squared = a * a;
        const float b = squared + a;
        result[i] = b + y[i];
    }
    return result;
}

std::vector<float> materializedReference(const std::vector<float>& x,
                                         const std::vector<float>& y) {
    std::vector<float> result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float a = x[i] + y[i];
        const float b = -a;
        const float c = a * y[i];
        const float d = b + c;
        result[i] = d - x[i];
    }
    return result;
}

void exact(const std::vector<float>& actual, const std::vector<float>& expected,
           std::string_view context) {
    require(actual.size() == expected.size(), std::string(context) + " size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]))
            fail(std::string(context) + " mismatch at element " + std::to_string(i));
    }
}

tooling::HostToolchainConfig driverToolchain() {
    const fs::path build = TH023_BUILD;
    return {TH023_CXX, tooling::configuredHostCompilerArguments(), {TH023_INCLUDE},
            {build / "libthiran_v0_storage.a", build / "libthiran_v0_async.a",
             build / "libthiran_v0_analysis.a", build / "libthiran_v0_semantic.a",
             build / "libthiran_v0_frontend.a", build / "libthiran_v0_extension.a"}};
}

artifact::NativeToolchain artifactToolchain() {
    const auto config = driverToolchain();
    return {config.compilerExecutable, config.compilerArguments,
            config.includePaths, config.v0StaticLibraries};
}

backend::TensorRegion region(const std::string& source, backend::NativeTarget target,
                             extension::ExtensionRegistry* registry = nullptr) {
    tooling::CompilerDriver driver(driverToolchain());
    auto checked = driver.checkSource({"<th023>", source}, registry);
    if (!checked.success) {
        std::string detail;
        for (const auto& item : checked.diagnostics) detail += item.format() + "; ";
        fail("checking failed: " + detail);
    }
    auto lowered = backend::extractStrictRegion(checked.checked->module, checked.checked->ownership,
                                                "bench", false, target);
    require(lowered.ok(), "lowering failed: " + lowered.diagnostic + lowered.coverage);
    return std::move(*lowered.region);
}

std::vector<float> values(const artifact::ArtifactExecutionResult& result) {
    require(result.ok(), result.error ? result.error->code + ": " + result.error->message :
                                      "artifact execution failed");
    return std::get<storage::Tensor>(*result.value).logicalF32Values();
}

void addPlan(Record& record, const backend::TensorRegion& selected,
             backend::PhysicalDevice device, backend::PhysicalPlanOptions options) {
    auto planned = backend::buildPhysicalPlan(selected, device, options);
    require(planned.ok(), "physical plan failed");
    const auto& plan = *planned.plan;
    record.counters["fusion_groups"] = plan.fusionGroups.size();
    record.counters["loops_or_kernels"] = plan.fusionGroups.size();
    record.counters["fused_groups"] = static_cast<std::uint64_t>(std::count_if(
        plan.fusionGroups.begin(), plan.fusionGroups.end(), [](const auto& group) { return group.fused; }));
    record.counters["physical_slots"] = plan.slots.size();
    record.counters["physical_temporary_slots"] = static_cast<std::uint64_t>(std::count_if(
        plan.slots.begin(), plan.slots.end(), [](const auto& slot) { return slot.reusable && !slot.external; }));
    std::uint64_t logical = 0, materialized = 0, bytes = 0;
    for (const auto& node : selected.nodes) {
        if (node.type.kind == semantic::TypeKind::Tensor && node.op != backend::RegionOp::Input &&
            node.op != backend::RegionOp::Alias && node.id != selected.output) ++logical;
    }
    for (const auto& value : plan.values) {
        if (value.value == value.root && value.classification == backend::PhysicalValueClass::Temporary &&
            value.materialized) {
            ++materialized;
            bytes += value.requirement.byteCount.value_or(0);
        }
    }
    record.counters["logical_intermediates"] = logical;
    record.counters["materialized_intermediates"] = materialized;
    record.counters["logical_temporary_bytes"] = bytes;
}

void addGpu(Record& record, const backend::GpuExecutionEvidence& evidence) {
    record.counters["allocation_count"] = evidence.allocationCount;
    record.counters["allocation_bytes_requested"] = evidence.allocationBytesRequested;
    record.counters["h2d_count"] = evidence.hostToDeviceCopies;
    record.counters["h2d_bytes"] = evidence.hostToDeviceBytes;
    record.counters["d2h_count"] = evidence.deviceToHostCopies;
    record.counters["d2h_bytes"] = evidence.deviceToHostBytes;
    record.counters["d2d_count"] = evidence.deviceToDeviceCopies;
    record.counters["d2d_bytes"] = evidence.deviceToDeviceBytes;
    record.counters["kernel_count"] = evidence.kernelLaunches;
    record.counters["synchronizations"] = evidence.synchronizations;
    record.counters["module_loads"] = evidence.moduleLoads;
    record.counters["driver_jit_count"] = evidence.driverJitLoads;
    record.counters["planner_allocations"] = evidence.plannerOwnedAllocations;
    record.labels["gpu_name"] = evidence.device.name;
    record.labels["compute_capability"] = std::to_string(evidence.device.computeMajor) + "." +
                                            std::to_string(evidence.device.computeMinor);
    record.labels["cuda_driver_api_version"] = std::to_string(evidence.device.driverVersion);
}

template <class Function>
std::pair<double, std::vector<float>> timed(Function&& function, std::size_t batch = 1) {
    std::vector<float> output;
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < batch; ++i) output = function();
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double, std::nano>(end - begin).count();
    return {elapsed / static_cast<double>(batch), std::move(output)};
}

std::shared_ptr<artifact::JitExecutable> compile(artifact::JitCompiler& compiler,
                                                 const backend::TensorRegion& selected,
                                                 const storage::Tensor& x,
                                                 const storage::Tensor& y,
                                                 backend::PhysicalPlanOptions options,
                                                 bool gpu) {
    auto compiled = gpu ? compiler.compileGpu(selected, {x, y}, options) :
                          compiler.compileCpu(selected, {x, y}, options);
    require(compiled.ok(), compiled.error ? compiled.error->code + ": " + compiled.error->message :
                                           "JIT compilation failed");
    return std::move(compiled.executable);
}

std::vector<float> execute(const std::shared_ptr<artifact::JitExecutable>& executable,
                           const storage::Tensor& x, const storage::Tensor& y,
                           std::optional<backend::GpuExecutionEvidence>& evidence) {
    auto result = executable->execute({x, y});
    if (result.gpuEvidence) evidence = result.gpuEvidence;
    return values(result);
}

std::size_t cppBatch(std::size_t count) {
    if (count == (1ULL << 10)) return 8192;
    if (count == (1ULL << 20)) return 8;
    return 1;
}

std::string sizeLabel(std::size_t count) {
    if (count == (1ULL << 10)) return "small";
    if (count == (1ULL << 20)) return "medium";
    return "large";
}

void cpuElementwise(std::vector<Record>& records, std::size_t count) {
    auto [xValues, yValues] = inputs(count);
    const auto expected = elementwiseReference(xValues, yValues);
    const auto x = storage::Tensor::materializeF32({count}, xValues);
    const auto y = storage::Tensor::materializeF32({count}, yValues);
    const auto selected = region(kElementwise, backend::NativeTarget::Cpu);
    artifact::JitCompiler compiler(artifactToolchain());
    auto fused = compile(compiler, selected, x, y, {true, true}, false);
    auto unfused = compile(compiler, selected, x, y, {false, false}, false);
    std::optional<backend::GpuExecutionEvidence> ignored;
    exact(execute(fused, x, y, ignored), expected, "Thiran CPU fused correctness");
    exact(execute(unfused, x, y, ignored), expected, "Thiran CPU unfused correctness");
    exact(cppFused(xValues, yValues), expected, "C++ fused correctness");
    exact(cppMaterialized(xValues, yValues), expected, "C++ materialized correctness");

    std::vector<Record> local(4);
    local[0] = {"elementwise_chain", "thiran_fused", "cpu", count, kCpuWarmups, 1,
                "warm_in_process_host_wall", {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    local[1] = {"elementwise_chain", "cpp_fused", "cpu", count, kCpuWarmups, cppBatch(count),
                "warm_in_process_host_wall", {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    local[2] = {"elementwise_chain", "thiran_unfused", "cpu", count, kCpuWarmups, 1,
                "warm_in_process_host_wall", {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    local[3] = {"elementwise_chain", "cpp_materialized", "cpu", count, kCpuWarmups, cppBatch(count),
                "warm_in_process_host_wall", {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    addPlan(local[0], selected, backend::PhysicalDevice::Host, {true, true});
    addPlan(local[2], selected, backend::PhysicalDevice::Host, {false, false});
    local[0].labels["size_label"] = local[1].labels["size_label"] =
        local[2].labels["size_label"] = local[3].labels["size_label"] = sizeLabel(count);
    local[0].labels["compiler_flags"] = local[2].labels["compiler_flags"] =
        "-std=c++20 -fPIC -shared -Wl,--strip-debug";
    local[1].labels["compiler_flags"] = local[3].labels["compiler_flags"] =
        "-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -std=c++20";

    const auto run = [&](std::size_t which) {
        std::pair<double, std::vector<float>> observation;
        if (which == 0) observation = timed([&] { return execute(fused, x, y, ignored); });
        else if (which == 1) observation = timed([&] { return cppFused(xValues, yValues); }, cppBatch(count));
        else if (which == 2) observation = timed([&] { return execute(unfused, x, y, ignored); });
        else observation = timed([&] { return cppMaterialized(xValues, yValues); }, cppBatch(count));
        exact(observation.second, expected, "timed CPU output");
        require(digest(observation.second) == local[which].outputDigest, "timed CPU digest mismatch");
        local[which].samplesNs.push_back(observation.first);
    };
    for (std::size_t warmup = 0; warmup < kCpuWarmups; ++warmup)
        for (std::size_t which = 0; which < 4; ++which) {
            std::pair<double, std::vector<float>> discarded;
            if (which == 0) discarded = timed([&] { return execute(fused, x, y, ignored); });
            else if (which == 1) discarded = timed([&] { return cppFused(xValues, yValues); }, cppBatch(count));
            else if (which == 2) discarded = timed([&] { return execute(unfused, x, y, ignored); });
            else discarded = timed([&] { return cppMaterialized(xValues, yValues); }, cppBatch(count));
            require(!discarded.second.empty(), "CPU warmup output missing");
        }
    for (std::size_t round = 0; round < kSamples; ++round)
        for (std::size_t offset = 0; offset < 4; ++offset) run((round + offset) % 4);
    records.insert(records.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
}

void paired(std::vector<Record>& records, Record first, Record second,
            const std::vector<float>& expected,
            const std::function<std::pair<std::vector<float>, std::optional<backend::GpuExecutionEvidence>>()>& a,
            const std::function<std::pair<std::vector<float>, std::optional<backend::GpuExecutionEvidence>>()>& b,
            std::size_t warmups) {
    auto check = [&](auto&& function, std::string_view name) {
        auto result = function(); exact(result.first, expected, name); return result;
    };
    check(a, first.implementation); check(b, second.implementation);
    for (std::size_t i = 0; i < warmups; ++i) {
        if ((i & 1U) == 0) { check(a, "warmup A"); check(b, "warmup B"); }
        else { check(b, "warmup B"); check(a, "warmup A"); }
    }
    std::optional<backend::GpuExecutionEvidence> firstEvidence, secondEvidence;
    for (std::size_t block = 0; block < kSamples / 2; ++block) {
        for (const int which : {0, 1, 1, 0}) {
            const auto begin = std::chrono::steady_clock::now();
            auto result = which == 0 ? a() : b();
            const auto end = std::chrono::steady_clock::now();
            exact(result.first, expected, "paired timed output");
            const double ns = std::chrono::duration<double, std::nano>(end - begin).count();
            if (which == 0) { first.samplesNs.push_back(ns); firstEvidence = result.second; }
            else { second.samplesNs.push_back(ns); secondEvidence = result.second; }
        }
    }
    if (firstEvidence) addGpu(first, *firstEvidence);
    if (secondEvidence) addGpu(second, *secondEvidence);
    records.push_back(std::move(first)); records.push_back(std::move(second));
}

void gpuElementwise(std::vector<Record>& records, std::size_t count) {
    auto [xValues, yValues] = inputs(count);
    const auto expected = elementwiseReference(xValues, yValues);
    const auto x = storage::Tensor::materializeF32({count}, xValues);
    const auto y = storage::Tensor::materializeF32({count}, yValues);
    const auto selected = region(kElementwise, backend::NativeTarget::Gpu);
    artifact::JitCompiler compiler;
    auto fused = compile(compiler, selected, x, y, {true, true}, true);
    auto unfused = compile(compiler, selected, x, y, {false, false}, true);
    Record a{"elementwise_chain", "thiran_fused", "gpu", count, kGpuWarmups, 1,
             "transfer_inclusive_synchronized_host_wall", {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    Record b{"elementwise_chain", "thiran_unfused", "gpu", count, kGpuWarmups, 1,
             "transfer_inclusive_synchronized_host_wall", {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    addPlan(a, selected, backend::PhysicalDevice::Gpu, {true, true});
    addPlan(b, selected, backend::PhysicalDevice::Gpu, {false, false});
    a.labels["size_label"] = b.labels["size_label"] = sizeLabel(count);
    const auto run = [&](const auto& executable) {
        std::optional<backend::GpuExecutionEvidence> evidence;
        auto output = execute(executable, x, y, evidence);
        return std::make_pair(std::move(output), evidence);
    };
    paired(records, std::move(a), std::move(b), expected,
           [&] { return run(fused); }, [&] { return run(unfused); }, kGpuWarmups);
}

void extensionBench(std::vector<Record>& records, bool gpu) {
    constexpr std::size_t count = 1ULL << 20;
    auto [xValues, yValues] = inputs(count);
    const auto expected = extensionReference(xValues, yValues);
    const auto x = storage::Tensor::materializeF32({count}, xValues);
    const auto y = storage::Tensor::materializeF32({count}, yValues);
    extension::ExtensionRegistry registry;
    const auto loaded = registry.load(TH023_EXTENSION);
    require(loaded.ok(), "cannot load TH-021 benchmark extension: " + loaded.message);
    const auto target = gpu ? backend::NativeTarget::Gpu : backend::NativeTarget::Cpu;
    const auto extensionRegion = region(kExtension, target, &registry);
    const auto builtInRegion = region(kExtensionBuiltIn, target);
    const auto extensionCpp = backend::emitCpp20(extensionRegion, false);
    require(extensionCpp.find("evaluateRecipe") == std::string::npos &&
            extensionCpp.find("ExtensionRegistry") == std::string::npos &&
            extensionCpp.find("dlopen") == std::string::npos,
            "extension CPU hot path retained runtime dispatch");
    if (gpu) {
        const auto ptx = backend::emitNativeGpuPtx(extensionRegion);
        require(ptx.find("research_square_linear") == std::string::npos &&
                ptx.find("callback") == std::string::npos,
                "extension GPU hot path retained symbolic dispatch");
    }
    artifact::JitCompiler compiler(gpu ? artifact::NativeToolchain{} : artifactToolchain());
    auto extensionExecutable = compile(compiler, extensionRegion, x, y, {true, true}, gpu);
    auto builtInExecutable = compile(compiler, builtInRegion, x, y, {true, true}, gpu);
    const std::string backendName = gpu ? "gpu" : "cpu";
    const std::string scope = gpu ? "transfer_inclusive_synchronized_host_wall" :
                                    "warm_in_process_host_wall";
    Record a{"extension_chain", "generic_extension", backendName, count,
             gpu ? kGpuWarmups : kCpuWarmups, 1, scope, {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    Record b{"extension_chain", "builtin_equivalent", backendName, count,
             gpu ? kGpuWarmups : kCpuWarmups, 1, scope, {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    addPlan(a, extensionRegion, gpu ? backend::PhysicalDevice::Gpu : backend::PhysicalDevice::Host,
            {true, true});
    addPlan(b, builtInRegion, gpu ? backend::PhysicalDevice::Gpu : backend::PhysicalDevice::Host,
            {true, true});
    a.counters["runtime_registry_lookups"] = b.counters["runtime_registry_lookups"] = 0;
    a.counters["runtime_callbacks"] = b.counters["runtime_callbacks"] = 0;
    const auto run = [&](const auto& executable) {
        std::optional<backend::GpuExecutionEvidence> evidence;
        auto output = execute(executable, x, y, evidence);
        return std::make_pair(std::move(output), evidence);
    };
    paired(records, std::move(a), std::move(b), expected,
           [&] { return run(extensionExecutable); }, [&] { return run(builtInExecutable); },
           gpu ? kGpuWarmups : kCpuWarmups);
}

void materializationBench(std::vector<Record>& records, bool gpu) {
    constexpr std::size_t count = 1ULL << 20;
    auto [xValues, yValues] = inputs(count);
    const auto expected = materializedReference(xValues, yValues);
    const auto x = storage::Tensor::materializeF32({count}, xValues);
    const auto y = storage::Tensor::materializeF32({count}, yValues);
    const auto selected = region(kMaterialized, gpu ? backend::NativeTarget::Gpu : backend::NativeTarget::Cpu);
    artifact::JitCompiler compiler(gpu ? artifact::NativeToolchain{} : artifactToolchain());
    auto executable = compile(compiler, selected, x, y, {true, true}, gpu);
    Record record{"multiple_consumer_materialization", "thiran_planned", gpu ? "gpu" : "cpu", count,
                  gpu ? kGpuWarmups : kCpuWarmups, 1,
                  gpu ? "transfer_inclusive_synchronized_host_wall" : "warm_in_process_host_wall",
                  {}, inputDigest(xValues, yValues), digest(expected), {}, {}};
    addPlan(record, selected, gpu ? backend::PhysicalDevice::Gpu : backend::PhysicalDevice::Host,
            {true, true});
    std::optional<backend::GpuExecutionEvidence> lastEvidence;
    auto run = [&] {
        std::optional<backend::GpuExecutionEvidence> evidence;
        auto output = execute(executable, x, y, evidence);
        lastEvidence = evidence;
        return output;
    };
    exact(run(), expected, "materialization correctness");
    for (std::size_t i = 0; i < record.warmups; ++i) exact(run(), expected, "materialization warmup");
    for (std::size_t i = 0; i < kSamples; ++i) {
        auto observation = timed(run);
        exact(observation.second, expected, "materialization timed output");
        record.samplesNs.push_back(observation.first);
    }
    if (lastEvidence) addGpu(record, *lastEvidence);
    records.push_back(std::move(record));
}

const training::ParameterValue& parameter(const training::TrainingPlan& plan,
                                          const training::TrainingState& state,
                                          std::size_t sourceIndex) {
    const auto descriptor = std::find_if(plan.parameters.begin(), plan.parameters.end(),
        [&](const auto& item) { return item.sourceParameterIndex == sourceIndex; });
    require(descriptor != plan.parameters.end(), "model parameter descriptor missing");
    const auto value = std::find_if(state.parameters.begin(), state.parameters.end(),
        [&](const auto& item) { return item.id == descriptor->id; });
    require(value != state.parameters.end(), "model parameter value missing");
    return *value;
}

artifact::ArtifactValue artifactValue(const training::ParameterValue& value) {
    const auto& tensor = std::get<semantic::RuntimeTensor>(value.value.data);
    std::vector<std::uint64_t> shape;
    for (const auto extent : tensor.shape) shape.push_back(static_cast<std::uint64_t>(extent));
    return storage::Tensor::materializeF32(std::move(shape), tensor.f32Values);
}

float scalarResult(const artifact::ArtifactValue& value) {
    return std::get<storage::Tensor>(value).logicalF32Values().at(0);
}

model::ModelBundle buildModelBundle(const fs::path& bundlePath, bool includeGpu,
                                    float& expected) {
    TemporaryDirectory temporary("th023-model");
    auto reference = model::createReferenceAffinePlan();
    require(reference.ok(), "reference model plan failed: " + reference.error);
    auto initialized = model::initializeReferenceAffineState(*reference.plan);
    require(initialized.ok(), "reference model initialization failed");
    auto schema = training::createTrainingCheckpointSchema(
        std::string(model::referenceAffineModelIdentity), *reference.plan, *initialized.state);
    require(schema.ok(), "reference checkpoint schema failed");
    auto state = std::move(*initialized.state);
    for (std::size_t step = 0; step < 200; ++step) {
        auto next = training::trainingStep(*reference.plan, state, model::referenceAffineBatch());
        require(next.ok(), "reference model training failed");
        state = std::move(*next.nextState);
    }
    const auto checkpoint = temporary.path / "trained.thc";
    auto saved = training::saveTrainingCheckpoint(checkpoint, *schema.schema, *reference.plan, state);
    require(saved.ok(), "reference checkpoint save failed");
    state = {};
    auto restored = training::loadTrainingCheckpoint(checkpoint, *schema.schema, *reference.plan);
    require(restored.ok(), "reference checkpoint load failed");
    auto snapshot = model::createDeploymentSnapshot(
        *reference.plan, restored, model::referenceAffineSnapshotRequest());
    require(snapshot.ok(), "deployment snapshot failed");
    auto native = model::referenceAffineInferenceRegion(*reference.module, backend::NativeTarget::Cpu);
    require(native.ok(), "reference inference lowering failed");
    const auto publicInput = model::referenceAffinePublicInput(2.0f);
    const auto weight = artifactValue(parameter(*reference.plan, *restored.state, 2));
    const auto bias = artifactValue(parameter(*reference.plan, *restored.state, 3));
    const auto entry = artifact::specializeEntry(*native.region, {publicInput, weight, bias});
    auto cpuBuilt = artifact::buildCpuAot(*native.region, artifactToolchain(),
        {temporary.path / "model-cpu.tha", {true, true}, entry});
    require(cpuBuilt.success, cpuBuilt.error ? cpuBuilt.error->message : "model CPU artifact build failed");
    auto cpuArtifact = artifact::loadArtifact(temporary.path / "model-cpu.tha");
    require(cpuArtifact.ok(), "model CPU artifact load failed");
    std::vector<model::EmbeddedArtifact> artifacts;
    artifacts.push_back({std::string(model::referenceAffineModelIdentity), *cpuArtifact.artifact});
    if (includeGpu) {
        auto gpuBuilt = artifact::buildGpuAot(*native.region,
            {temporary.path / "model-gpu.tha", {true, true}, entry});
        require(gpuBuilt.success, gpuBuilt.error ? gpuBuilt.error->message : "model GPU artifact build failed");
        auto gpuArtifact = artifact::loadArtifact(temporary.path / "model-gpu.tha");
        require(gpuArtifact.ok(), "model GPU artifact load failed");
        artifacts.push_back({std::string(model::referenceAffineModelIdentity), *gpuArtifact.artifact});
    }
    auto bundled = model::createModelBundle(*snapshot.snapshot, std::move(artifacts));
    require(bundled.ok(), bundled.error ? bundled.error->message : "model bundle creation failed");
    auto written = model::writeModelBundle(*bundled.bundle, bundlePath);
    require(written.ok(), written.error ? written.error->message : "model bundle write failed");
    auto oracle = model::evaluateReferenceAffine(*reference.module, *reference.plan,
                                                  *restored.state, 2.0f);
    require(oracle.ok, "reference model evaluation failed");
    expected = std::get<semantic::RuntimeTensor>(oracle.value->data).f32Values.at(0);
    return std::move(*bundled.bundle);
}

void modelBench(std::vector<Record>& records, bool gpu, const fs::path& bundlePath) {
    float expected = 0;
    auto bundle = buildModelBundle(bundlePath, gpu, expected);
    const auto publicInput = model::referenceAffinePublicInput(2.0f);
    const auto selected = gpu ? model::ModelBackend::Gpu : model::ModelBackend::Cpu;
    Record record{"reference_model_deployment", "thiran_model_runtime", gpu ? "gpu" : "cpu", 1,
                  gpu ? kGpuWarmups : kCpuWarmups, 1,
                  gpu ? "deployment_transfer_inclusive_synchronized_host_wall" :
                        "deployment_warm_in_process_host_wall", {}, digest({2.0f}), {}, {}, {}};
    std::optional<backend::GpuExecutionEvidence> lastEvidence;
    auto run = [&] {
        auto result = model::executeModel(bundle, selected, {publicInput});
        require(result.ok(), result.error ? result.error->code + ": " + result.error->message :
                                           "model execution failed");
        if (result.gpuEvidence) lastEvidence = result.gpuEvidence;
        const float actual = scalarResult(*result.value);
        require(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(expected),
                "deployed model output mismatch");
        return std::vector<float>{actual};
    };
    record.outputDigest = digest(run());
    for (std::size_t i = 0; i < record.warmups; ++i) run();
    for (std::size_t i = 0; i < kSamples; ++i) {
        auto observation = timed(run);
        record.samplesNs.push_back(observation.first);
    }
    record.counters["model_bundle_bytes"] = fs::file_size(bundlePath);
    if (lastEvidence) addGpu(record, *lastEvidence);
    records.push_back(std::move(record));
}

void writeRecords(const fs::path& output, std::string_view backendName,
                  const std::vector<Record>& records) {
    std::ofstream out(output, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(out), "cannot create result JSON");
    out << "{\n  \"schema_version\":1,\n  \"commit\":" << json(kCommit)
        << ",\n  \"build_type\":\"Release\",\n  \"qualification_backend\":"
        << json(backendName) << ",\n  \"records\":[\n";
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        const auto stats = statistics(record.samplesNs);
        out << "    {\"benchmark_name\":" << json(record.benchmarkName)
            << ",\"implementation\":" << json(record.implementation)
            << ",\"backend\":" << json(record.backendName)
            << ",\"size\":" << record.size
            << ",\"build_type\":\"Release\",\"warmup_count\":" << record.warmups
            << ",\"sample_count\":" << record.samplesNs.size()
            << ",\"batch_count\":" << record.batch
            << ",\"timing_scope\":" << json(record.timingScope)
            << ",\"samples_ns\":[";
        out << std::setprecision(17);
        for (std::size_t sample = 0; sample < record.samplesNs.size(); ++sample) {
            if (sample) out << ',';
            out << record.samplesNs[sample];
        }
        out << "],\"median_ns\":" << stats.median << ",\"mean_ns\":" << stats.mean
            << ",\"stddev_ns\":" << stats.stddev << ",\"p10_ns\":" << stats.p10
            << ",\"p90_ns\":" << stats.p90 << ",\"min_ns\":" << stats.minimum
            << ",\"max_ns\":" << stats.maximum
            << ",\"output_digest\":" << json(record.outputDigest)
            << ",\"input_digest\":" << json(record.inputDigest)
            << ",\"commit\":" << json(kCommit) << ",\"counters\":{";
        std::size_t field = 0;
        for (const auto& [name, value] : record.counters) {
            if (field++) out << ',';
            out << json(name) << ':' << value;
        }
        out << "},\"labels\":{";
        field = 0;
        for (const auto& [name, value] : record.labels) {
            if (field++) out << ',';
            out << json(name) << ':' << json(value);
        }
        out << "}}" << (i + 1 == records.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    require(static_cast<bool>(out), "failed to write result JSON");
}

struct Options { bool gpu = false; fs::path output; fs::path modelBundle; };
Options options(int argc, char* argv[]) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--backend" && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "gpu") result.gpu = true;
            else if (value != "cpu") fail("--backend must be cpu or gpu");
        } else if (argument == "--output" && i + 1 < argc) result.output = argv[++i];
        else if (argument == "--model-bundle" && i + 1 < argc) result.modelBundle = argv[++i];
        else fail("unknown or incomplete argument: " + argument);
    }
    require(!result.output.empty(), "--output is required");
    require(!result.modelBundle.empty(), "--model-bundle is required");
    return result;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const auto selected = options(argc, argv);
        std::vector<Record> records;
        if (selected.gpu) {
            const auto probe = backend::probeNativeGpu();
            require(probe.deviceAvailable, probe.error ? probe.error->code + ": " + probe.error->message :
                                                        "physical GPU unavailable");
            for (const std::size_t size : {1ULL << 10, 1ULL << 20, 1ULL << 24})
                gpuElementwise(records, size);
            extensionBench(records, true);
            materializationBench(records, true);
            modelBench(records, true, selected.modelBundle);
        } else {
            for (const std::size_t size : {1ULL << 10, 1ULL << 20, 1ULL << 24})
                cpuElementwise(records, size);
            extensionBench(records, false);
            materializationBench(records, false);
            modelBench(records, false, selected.modelBundle);
        }
        writeRecords(selected.output, selected.gpu ? "gpu" : "cpu", records);
        std::cout << "TH-023 qualification " << (selected.gpu ? "gpu" : "cpu")
                  << " PASS records=" << records.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TH-023 qualification FAIL: " << error.what() << '\n';
        return 1;
    }
}
