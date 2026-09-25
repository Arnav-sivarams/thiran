#include "backend/v0/NativeGpu.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef THIRAN_ENABLE_NATIVE_GPU
#define THIRAN_ENABLE_NATIVE_GPU 0
#endif

namespace thiran::v0::backend {

NativeResult extractStrictGpu(const semantic::Module& module,
                              const analysis::OwnershipAnalysisResult& facts,
                              const std::string& name,
                              bool standalone) {
    return extractStrictRegion(module, facts, name, standalone, NativeTarget::Gpu);
}

bool nativeGpuBackendBuilt() noexcept { return THIRAN_ENABLE_NATIVE_GPU != 0; }
analysis::EffectSet nativeGpuAsyncEffects() noexcept {
    return static_cast<analysis::EffectSet>(analysis::EffectKind::Transfer) |
           static_cast<analysis::EffectSet>(analysis::EffectKind::Async);
}

namespace {

#if THIRAN_ENABLE_NATIVE_GPU

class Failure final : public std::exception {
public:
    explicit Failure(GpuError error) : error_(std::move(error)) {}
    const char* what() const noexcept override { return error_.message.c_str(); }
    const GpuError& error() const noexcept { return error_; }
private:
    GpuError error_;
};

[[noreturn]] void fail(GpuErrorCategory category, std::string code, std::string message) {
    throw Failure({category, std::move(code), std::move(message)});
}

using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUevent_st;
struct CUstream_st;
using CUcontext = CUctx_st*;
using CUmodule = CUmod_st*;
using CUfunction = CUfunc_st*;
using CUevent = CUevent_st*;
using CUstream = CUstream_st*;
using CUresult = int;

constexpr CUresult cudaSuccess = 0;
constexpr CUresult cudaErrorNotReady = 600;
constexpr unsigned int cudaStreamNonBlocking = 1;
constexpr unsigned int cudaEventDisableTiming = 2;

void preflight(const TensorRegion& region, const std::vector<GpuValue>& inputs) {
    const auto verification = verifyRegion(region);
    if (!verification.ok())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-REGION", verification.errors.front());
    if (inputs.size() != region.inputs.size())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-INPUT-ARITY", "GPU input arity mismatch");
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto id = region.inputs[index];
        const auto node = std::find_if(region.nodes.begin(), region.nodes.end(),
                                       [&](const auto& candidate) { return candidate.id == id; });
        if (node == region.nodes.end())
            fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-REGION", "missing GPU input node");
        if (const auto* tensor = std::get_if<storage::Tensor>(&inputs[index])) {
            if (node->type.kind != semantic::TypeKind::Tensor)
                fail(GpuErrorCategory::BackendUnsupported, "GPU-INPUT-TYPE",
                     "tensor supplied for scalar GPU input");
            const auto& descriptor = tensor->descriptor();
            try { storage::verifySemanticShape(node->type, node->shape, descriptor); }
            catch (const std::exception& error) {
                fail(GpuErrorCategory::BackendUnsupported, "GPU-INPUT-SHAPE", error.what());
            }
            if (descriptor.view || descriptor.elementOffset != 0 || !tensor->isContiguousRowMajor())
                fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-LAYOUT",
                     "native GPU requires materialized offset-zero contiguous row-major tensors");
        } else if (std::holds_alternative<std::int64_t>(inputs[index])) {
            if (node->type != semantic::scalar(semantic::TypeKind::I64))
                fail(GpuErrorCategory::BackendUnsupported, "GPU-INPUT-TYPE",
                     "i64 supplied for non-i64 GPU input");
        } else if (node->type != semantic::scalar(semantic::TypeKind::F32)) {
            fail(GpuErrorCategory::BackendUnsupported, "GPU-INPUT-TYPE",
                 "f32 supplied for non-f32 GPU input");
        }
    }
}

class Driver final {
public:
    Driver() {
        handle_ = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!handle_)
            fail(GpuErrorCategory::BackendUnavailable, "GPU-DRIVER-NOT-FOUND",
                 std::string("CUDA driver library unavailable: ") + dlerror());
        load(cuInit, "cuInit");
        load(cuDriverGetVersion, "cuDriverGetVersion");
        load(cuDeviceGetCount, "cuDeviceGetCount");
        load(cuDeviceGet, "cuDeviceGet");
        load(cuDeviceGetName, "cuDeviceGetName");
        load(cuDeviceComputeCapability, "cuDeviceComputeCapability");
        load(cuCtxCreate, "cuCtxCreate_v2");
        load(cuCtxDestroy, "cuCtxDestroy_v2");
        load(cuCtxSetCurrent, "cuCtxSetCurrent");
        load(cuMemAlloc, "cuMemAlloc_v2");
        load(cuMemFree, "cuMemFree_v2");
        load(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
        load(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
        load(cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
        load(cuModuleLoadDataEx, "cuModuleLoadDataEx");
        load(cuModuleUnload, "cuModuleUnload");
        load(cuModuleGetFunction, "cuModuleGetFunction");
        load(cuLaunchKernel, "cuLaunchKernel");
        load(cuStreamCreate, "cuStreamCreate");
        load(cuStreamDestroy, "cuStreamDestroy_v2");
        load(cuStreamSynchronize, "cuStreamSynchronize");
        load(cuEventCreate, "cuEventCreate");
        load(cuEventRecord, "cuEventRecord");
        load(cuEventQuery, "cuEventQuery");
        load(cuEventSynchronize, "cuEventSynchronize");
        load(cuEventDestroy, "cuEventDestroy_v2");
        load(cuGetErrorName, "cuGetErrorName");
        load(cuGetErrorString, "cuGetErrorString");
    }

    ~Driver() { if (handle_) dlclose(handle_); }
    Driver(const Driver&) = delete;
    Driver& operator=(const Driver&) = delete;

    std::string describe(CUresult result) const {
        const char* name = nullptr;
        const char* detail = nullptr;
        (void)cuGetErrorName(result, &name);
        (void)cuGetErrorString(result, &detail);
        std::string text = name ? name : "CUDA_ERROR_UNKNOWN";
        if (detail) text += ": " + std::string(detail);
        return text;
    }

    void check(CUresult result, std::string_view operation,
               GpuErrorCategory category = GpuErrorCategory::RuntimeFailure) const {
        if (result != cudaSuccess)
            fail(category, "GPU-CUDA-" + std::to_string(result),
                 std::string(operation) + " failed: " + describe(result));
    }

    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuDriverGetVersion)(int*) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceComputeCapability)(int*, int*, CUdevice) = nullptr;
    CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = nullptr;
    CUresult (*cuCtxDestroy)(CUcontext) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, std::size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, std::size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) = nullptr;
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, void*, void*) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, CUstream, void**, void**) = nullptr;
    CUresult (*cuStreamCreate)(CUstream*, unsigned int) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuEventCreate)(CUevent*, unsigned int) = nullptr;
    CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*cuEventQuery)(CUevent) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*cuEventDestroy)(CUevent) = nullptr;
    CUresult (*cuGetErrorName)(CUresult, const char**) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;

private:
    template<class Function>
    void load(Function& target, const char* name) {
        dlerror();
        void* symbol = dlsym(handle_, name);
        if (const char* error = dlerror())
            fail(GpuErrorCategory::BackendUnavailable, "GPU-DRIVER-SYMBOL",
                 std::string("CUDA driver symbol ") + name + " unavailable: " + error);
        static_assert(sizeof(target) == sizeof(symbol));
        std::memcpy(&target, &symbol, sizeof(target));
    }
    void* handle_ = nullptr;
};

GpuDeviceInfo initialize(Driver& driver, int selected) {
    GpuDeviceInfo info;
    info.backendBuilt = true;
    info.driverLoaded = true;
    driver.check(driver.cuInit(0), "cuInit", GpuErrorCategory::BackendUnavailable);
    driver.check(driver.cuDriverGetVersion(&info.driverVersion), "cuDriverGetVersion");
    driver.check(driver.cuDeviceGetCount(&info.deviceCount), "cuDeviceGetCount",
                 GpuErrorCategory::BackendUnavailable);
    if (info.deviceCount == 0)
        fail(GpuErrorCategory::BackendUnavailable, "GPU-NO-DEVICE", "no CUDA device is available");
    if (selected < 0 || selected >= info.deviceCount)
        fail(GpuErrorCategory::InvalidDevice, "GPU-INVALID-DEVICE",
             "requested GPU device " + std::to_string(selected) + " but device count is " +
             std::to_string(info.deviceCount));
    CUdevice device = 0;
    driver.check(driver.cuDeviceGet(&device, selected), "cuDeviceGet", GpuErrorCategory::InvalidDevice);
    char name[256]{};
    driver.check(driver.cuDeviceGetName(name, static_cast<int>(sizeof(name)), device), "cuDeviceGetName");
    driver.check(driver.cuDeviceComputeCapability(&info.computeMajor, &info.computeMinor, device),
                 "cuDeviceComputeCapability");
    if (info.computeMajor < 5)
        fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-CAPABILITY",
             "native GPU requires compute capability 5.0 or newer");
    info.deviceAvailable = true;
    info.selectedDevice = selected;
    info.name = name;
    return info;
}

class Context final {
public:
    Context(Driver& driver, int ordinal) : driver_(driver) {
        CUdevice device = 0;
        driver_.check(driver_.cuDeviceGet(&device, ordinal), "cuDeviceGet", GpuErrorCategory::InvalidDevice);
        driver_.check(driver_.cuCtxCreate(&context_, 0, device), "cuCtxCreate");
    }
    ~Context() {
        if (context_) {
            (void)driver_.cuCtxSetCurrent(context_);
            (void)driver_.cuCtxDestroy(context_);
        }
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void activate() { driver_.check(driver_.cuCtxSetCurrent(context_), "cuCtxSetCurrent"); }
    CUcontext get() const noexcept { return context_; }
private:
    Driver& driver_;
    CUcontext context_ = nullptr;
};

class Stream final {
public:
    Stream(Driver& driver, Context& context) : driver_(driver), context_(context.get()) {
        context.activate();
        driver_.check(driver_.cuStreamCreate(&stream_, cudaStreamNonBlocking), "cuStreamCreate");
    }
    ~Stream() {
        if (stream_) {
            (void)driver_.cuCtxSetCurrent(context_);
            (void)driver_.cuStreamDestroy(stream_);
        }
    }
    CUstream get() const noexcept { return stream_; }
private:
    Driver& driver_;
    CUcontext context_ = nullptr;
    CUstream stream_ = nullptr;
};

class Event final {
public:
    Event(Driver& driver, Context& context) : driver_(driver), context_(context.get()) {
        context.activate();
        driver_.check(driver_.cuEventCreate(&event_, cudaEventDisableTiming), "cuEventCreate");
    }
    ~Event() {
        if (event_) {
            (void)driver_.cuCtxSetCurrent(context_);
            (void)driver_.cuEventDestroy(event_);
        }
    }
    CUevent get() const noexcept { return event_; }
private:
    Driver& driver_;
    CUcontext context_ = nullptr;
    CUevent event_ = nullptr;
};

struct DeviceAllocation final {
    Driver* driver = nullptr;
    CUcontext context = nullptr;
    CUdeviceptr pointer = 0;
    std::size_t bytes = 0;
    ~DeviceAllocation() {
        if (pointer && driver) {
            (void)driver->cuCtxSetCurrent(context);
            (void)driver->cuMemFree(pointer);
        }
    }
};

using DeviceAllocationPtr = std::shared_ptr<DeviceAllocation>;

DeviceAllocationPtr allocate(Driver& driver, Context& context, std::size_t bytes,
                             GpuExecutionEvidence& evidence) {
    context.activate();
    auto allocation = std::make_shared<DeviceAllocation>();
    allocation->driver = &driver;
    allocation->context = context.get();
    allocation->bytes = bytes;
    if (bytes != 0) {
        driver.check(driver.cuMemAlloc(&allocation->pointer, bytes), "cuMemAlloc");
        ++evidence.allocationCount;
    }
    return allocation;
}

std::string binaryPtx(std::string_view name, bool f32, std::string_view operation,
                      std::string_view overflow) {
    const unsigned width = f32 ? 4U : 8U;
    std::ostringstream out;
    out << ".visible .entry " << name << "(\n"
        << " .param .u64 out, .param .u64 lhs, .param .u64 rhs,\n"
        << " .param .u64 count, .param .u64 error) {\n"
        << " .reg .pred %p<3>; .reg .b32 %r<6>; .reg .b64 %rd<20>;";
    if (f32) out << " .reg .f32 %f<4>;";
    out << "\n mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;\n"
        << " mad.lo.s32 %r4,%r1,%r2,%r3; cvt.u64.u32 %rd1,%r4;\n"
        << " ld.param.u64 %rd2,[count]; setp.ge.u64 %p1,%rd1,%rd2; @%p1 bra DONE;\n"
        << " mul.lo.u64 %rd3,%rd1," << width << ";\n"
        << " ld.param.u64 %rd4,[out]; ld.param.u64 %rd5,[lhs]; ld.param.u64 %rd6,[rhs];\n"
        << " add.u64 %rd7,%rd4,%rd3; add.u64 %rd8,%rd5,%rd3; add.u64 %rd9,%rd6,%rd3;\n";
    if (f32) {
        out << " ld.global.f32 %f1,[%rd8]; ld.global.f32 %f2,[%rd9];\n"
            << ' ' << operation << ".rn.f32 %f3,%f1,%f2; st.global.f32 [%rd7],%f3;\n";
    } else {
        out << " ld.global.s64 %rd10,[%rd8]; ld.global.s64 %rd11,[%rd9];\n"
            << ' ' << operation << ".s64 %rd12,%rd10,%rd11;\n"
            << overflow << '\n'
            << " st.global.u64 [%rd7],%rd12;\n";
    }
    out << "DONE: ret; }\n";
    return out.str();
}

std::string indexPtx(std::string_view name, bool f32) {
    const unsigned width = f32 ? 4U : 8U;
    std::ostringstream out;
    out << ".visible .entry " << name
        << "(.param .u64 out, .param .u64 input, .param .u64 index) {\n"
        << " .reg .pred %p<2>; .reg .b32 %r<3>; .reg .b64 %rd<8>;\n"
        << " mov.u32 %r1,%tid.x; setp.ne.u32 %p1,%r1,0; @%p1 bra DONE;\n"
        << " ld.param.u64 %rd1,[out]; ld.param.u64 %rd2,[input]; ld.param.u64 %rd3,[index];\n"
        << " mul.lo.u64 %rd4,%rd3," << width << "; add.u64 %rd5,%rd2,%rd4;\n";
    if (f32)
        out << " ld.global.u32 %r2,[%rd5]; st.global.u32 [%rd1],%r2;\n";
    else
        out << " ld.global.u64 %rd6,[%rd5]; st.global.u64 [%rd1],%rd6;\n";
    out << "DONE: ret; }\n";
    return out.str();
}

std::string unaryPtx(std::string_view name, bool f32) {
    const unsigned width = f32 ? 4U : 8U;
    std::ostringstream out;
    out << ".visible .entry " << name << "(\n"
        << " .param .u64 out, .param .u64 input, .param .u64 count, .param .u64 error) {\n"
        << " .reg .pred %p<3>; .reg .b32 %r<6>; .reg .b64 %rd<16>;";
    if (f32) out << " .reg .f32 %f<3>;";
    out << "\n mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;\n"
        << " mad.lo.s32 %r4,%r1,%r2,%r3; cvt.u64.u32 %rd1,%r4;\n"
        << " ld.param.u64 %rd2,[count]; setp.ge.u64 %p1,%rd1,%rd2; @%p1 bra DONE;\n"
        << " mul.lo.u64 %rd3,%rd1," << width << "; ld.param.u64 %rd4,[out];\n"
        << " ld.param.u64 %rd5,[input]; add.u64 %rd6,%rd4,%rd3; add.u64 %rd7,%rd5,%rd3;\n";
    if (f32) {
        out << " ld.global.f32 %f1,[%rd7]; neg.f32 %f2,%f1; st.global.f32 [%rd6],%f2;\n";
    } else {
        out << " ld.global.s64 %rd8,[%rd7]; neg.s64 %rd9,%rd8;\n"
            << " setp.eq.s64 %p2,%rd8,0x8000000000000000; @!%p2 bra NOERR;\n"
            << " ld.param.u64 %rd10,[error]; mov.u32 %r5,1; atom.global.exch.b32 %r5,[%rd10],%r5; NOERR:\n"
            << " st.global.u64 [%rd6],%rd9;\n";
    }
    out << "DONE: ret; }\n";
    return out.str();
}

std::string ptx() {
    const std::string addOverflow =
        " xor.b64 %rd13,%rd12,%rd10; xor.b64 %rd14,%rd12,%rd11; and.b64 %rd15,%rd13,%rd14;\n"
        " setp.lt.s64 %p2,%rd15,0; @!%p2 bra NOERR; ld.param.u64 %rd16,[error];\n"
        " mov.u32 %r5,1; atom.global.exch.b32 %r5,[%rd16],%r5; NOERR:";
    const std::string subOverflow =
        " xor.b64 %rd13,%rd10,%rd11; xor.b64 %rd14,%rd12,%rd10; and.b64 %rd15,%rd13,%rd14;\n"
        " setp.lt.s64 %p2,%rd15,0; @!%p2 bra NOERR; ld.param.u64 %rd16,[error];\n"
        " mov.u32 %r5,1; atom.global.exch.b32 %r5,[%rd16],%r5; NOERR:";
    const std::string mulOverflow =
        " mul.hi.s64 %rd13,%rd10,%rd11; shr.s64 %rd14,%rd12,63;\n"
        " setp.ne.s64 %p2,%rd13,%rd14; @!%p2 bra NOERR; ld.param.u64 %rd16,[error];\n"
        " mov.u32 %r5,1; atom.global.exch.b32 %r5,[%rd16],%r5; NOERR:";
    std::string text = ".version 6.0\n.target sm_50\n.address_size 64\n";
    text += binaryPtx("thiran_add_i64", false, "add", addOverflow);
    text += binaryPtx("thiran_sub_i64", false, "sub", subOverflow);
    text += binaryPtx("thiran_mul_i64", false, "mul.lo", mulOverflow);
    text += binaryPtx("thiran_add_f32", true, "add", "");
    text += binaryPtx("thiran_sub_f32", true, "sub", "");
    text += binaryPtx("thiran_mul_f32", true, "mul", "");
    text += unaryPtx("thiran_neg_i64", false);
    text += unaryPtx("thiran_neg_f32", true);
    text += indexPtx("thiran_index_i64", false);
    text += indexPtx("thiran_index_f32", true);
    return text;
}

const RegionNode& regionNode(const TensorRegion& region, semantic::ValueId id) {
    const auto found = std::find_if(region.nodes.begin(), region.nodes.end(),
        [&](const RegionNode& candidate) { return candidate.id == id; });
    if (found == region.nodes.end())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-PLAN", "planned value is absent from TensorRegion");
    return *found;
}

semantic::ValueId canonicalAlias(const TensorRegion& region, semantic::ValueId id) {
    std::set<semantic::ValueId> seen;
    while (seen.insert(id).second) {
        const auto& current = regionNode(region, id);
        if (current.op != RegionOp::Alias || current.dependencies.size() != 1) break;
        id = current.dependencies.front();
    }
    return id;
}

bool groupElementwise(const TensorRegion& region, const FusionGroup& group) {
    if (group.nodes.empty()) return false;
    const auto op = regionNode(region, group.nodes.front()).op;
    return op == RegionOp::Negate || op == RegionOp::Add ||
           op == RegionOp::Subtract || op == RegionOp::ElementMultiply;
}

std::vector<semantic::ValueId> groupInputs(const TensorRegion& region, const FusionGroup& group) {
    std::set<semantic::ValueId> inside(group.nodes.begin(), group.nodes.end());
    std::set<semantic::ValueId> seen;
    std::vector<semantic::ValueId> inputs;
    for (auto id : group.nodes)
        for (auto dependency : regionNode(region, id).dependencies) {
            dependency = canonicalAlias(region, dependency);
            if (!inside.contains(dependency) && seen.insert(dependency).second) inputs.push_back(dependency);
        }
    return inputs;
}

std::string groupKernelName(const FusionGroup& group, storage::DType dtype) {
    return "thiran_group_" + std::to_string(group.id) +
           (dtype == storage::DType::I64 ? "_i64" : "_f32");
}

std::string groupPtx(const TensorRegion& region, const FusionGroup& group) {
    const bool isF32 = group.type.elements.at(0).kind == semantic::TypeKind::F32;
    const auto dtype = isF32 ? storage::DType::F32 : storage::DType::I64;
    const auto inputs = groupInputs(region, group);
    std::map<semantic::ValueId, unsigned> registers;
    unsigned next = isF32 ? 1U : 32U;
    for (auto id : inputs) registers[id] = next++;
    for (auto id : group.nodes) registers[id] = next++;
    std::ostringstream out;
    out << ".visible .entry " << groupKernelName(group, dtype) << "(\n .param .u64 out";
    for (std::size_t index = 0; index < inputs.size(); ++index)
        out << ", .param .u64 input" << index;
    out << ", .param .u64 count, .param .u64 error) {\n"
        << " .reg .pred %p<4>; .reg .b32 %r<8>; .reg .b64 %rd<"
        << std::max<unsigned>(64U, next + 1U) << ">;";
    if (isF32) out << " .reg .f32 %f<" << std::max<unsigned>(8U, next + 1U) << ">;";
    const std::string done = "DONE_G" + std::to_string(group.id);
    out << "\n mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;\n"
        << " mad.lo.s32 %r4,%r1,%r2,%r3; cvt.u64.u32 %rd1,%r4;\n"
        << " ld.param.u64 %rd2,[count]; setp.ge.u64 %p1,%rd1,%rd2; @%p1 bra " << done << ";\n"
        << " mul.lo.u64 %rd3,%rd1," << (isF32 ? 4 : 8) << ";\n"
        << " ld.param.u64 %rd4,[out]; add.u64 %rd5,%rd4,%rd3;\n";
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        out << " ld.param.u64 %rd10,[input" << index << "]; add.u64 %rd11,%rd10,%rd3;\n";
        if (isF32) out << " ld.global.f32 %f" << registers.at(inputs[index]) << ",[%rd11];\n";
        else out << " ld.global.s64 %rd" << registers.at(inputs[index]) << ",[%rd11];\n";
    }
    auto registerName = [&](semantic::ValueId id) {
        id = canonicalAlias(region, id);
        return std::string(isF32 ? "%f" : "%rd") + std::to_string(registers.at(id));
    };
    for (std::size_t index = 0; index < group.nodes.size(); ++index) {
        const auto& operation = regionNode(region, group.nodes[index]);
        const auto destination = registerName(operation.id);
        const auto left = registerName(operation.dependencies[0]);
        const std::string label = "NOERR_G" + std::to_string(group.id) + "_N" + std::to_string(index);
        if (isF32) {
            if (operation.op == RegionOp::Negate) out << " neg.f32 " << destination << ',' << left << ";\n";
            else {
                const char* instruction = operation.op == RegionOp::Add ? "add" :
                    operation.op == RegionOp::Subtract ? "sub" : "mul";
                out << ' ' << instruction << ".rn.f32 " << destination << ',' << left << ','
                    << registerName(operation.dependencies[1]) << ";\n";
            }
            continue;
        }
        if (operation.op == RegionOp::Negate) {
            out << " neg.s64 " << destination << ',' << left << ";\n"
                << " setp.eq.s64 %p2," << left << ",0x8000000000000000; @!%p2 bra " << label << ";\n";
        } else if (operation.op == RegionOp::Add) {
            const auto right = registerName(operation.dependencies[1]);
            out << " add.s64 " << destination << ',' << left << ',' << right << ";\n"
                << " xor.b64 %rd12," << destination << ',' << left << "; xor.b64 %rd13,"
                << destination << ',' << right << "; and.b64 %rd14,%rd12,%rd13;\n"
                << " setp.lt.s64 %p2,%rd14,0; @!%p2 bra " << label << ";\n";
        } else if (operation.op == RegionOp::Subtract) {
            const auto right = registerName(operation.dependencies[1]);
            out << " sub.s64 " << destination << ',' << left << ',' << right << ";\n"
                << " xor.b64 %rd12," << left << ',' << right << "; xor.b64 %rd13,"
                << destination << ',' << left << "; and.b64 %rd14,%rd12,%rd13;\n"
                << " setp.lt.s64 %p2,%rd14,0; @!%p2 bra " << label << ";\n";
        } else {
            const auto right = registerName(operation.dependencies[1]);
            out << " mul.lo.s64 " << destination << ',' << left << ',' << right << ";\n"
                << " mul.hi.s64 %rd12," << left << ',' << right << "; shr.s64 %rd13,"
                << destination << ",63; setp.ne.s64 %p2,%rd12,%rd13; @!%p2 bra " << label << ";\n";
        }
        out << " ld.param.u64 %rd15,[error]; mov.u32 %r5,1; atom.global.exch.b32 %r6,[%rd15],%r5;\n"
            << label << ":\n";
    }
    const auto terminal = registerName(group.output);
    if (isF32) out << " st.global.f32 [%rd5]," << terminal << ";\n";
    else out << " st.global.u64 [%rd5]," << terminal << ";\n";
    out << done << ": ret; }\n";
    return out.str();
}

std::string plannedPtx(const TensorRegion& region, const PhysicalPlan& plan) {
    std::string result = ptx();
    for (const auto& group : plan.fusionGroups)
        if (groupElementwise(region, group)) result += groupPtx(region, group);
    return result;
}

class Module final {
public:
    Module(Driver& driver, Context& context, std::string source)
        : driver_(driver), context_(context.get()), source_(std::move(source)) {
        context.activate();
        driver_.check(driver_.cuModuleLoadDataEx(&module_, source_.c_str(), 0, nullptr, nullptr),
                      "cuModuleLoadDataEx", GpuErrorCategory::BackendUnsupported);
    }
    ~Module() {
        if (module_) {
            (void)driver_.cuCtxSetCurrent(context_);
            (void)driver_.cuModuleUnload(module_);
        }
    }
    CUfunction function(const std::string& name) {
        driver_.check(driver_.cuCtxSetCurrent(context_), "cuCtxSetCurrent");
        CUfunction function = nullptr;
        driver_.check(driver_.cuModuleGetFunction(&function, module_, name.c_str()), "cuModuleGetFunction");
        return function;
    }
private:
    Driver& driver_;
    CUcontext context_ = nullptr;
    std::string source_;
    CUmodule module_ = nullptr;
};

struct DeviceTensor {
    storage::DType dtype = storage::DType::Invalid;
    std::vector<std::uint64_t> shape;
    DeviceAllocationPtr allocation;
};

using DeviceValue = std::variant<std::int64_t, float, DeviceTensor>;

std::size_t bytes(const DeviceTensor& tensor) {
    return storage::checkedByteCount(storage::checkedElementCount(tensor.shape), tensor.dtype);
}

std::string kernelName(RegionOp op, storage::DType dtype) {
    std::string name = "thiran_";
    if (op == RegionOp::Add) name += "add_";
    else if (op == RegionOp::Subtract) name += "sub_";
    else if (op == RegionOp::ElementMultiply) name += "mul_";
    else if (op == RegionOp::Negate) name += "neg_";
    else if (op == RegionOp::Index) name += "index_";
    else fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-OP", "unsupported GPU operation");
    name += dtype == storage::DType::I64 ? "i64" : "f32";
    return name;
}

[[maybe_unused]] void launchUnary(Driver& driver, Context& context, Stream& stream, Module& module,
                 const DeviceTensor& input, DeviceTensor& output,
                 const DeviceAllocationPtr& error, GpuExecutionEvidence& evidence) {
    const auto count = storage::checkedElementCount(output.shape);
    if (count == 0) return;
    constexpr std::uint64_t block = 256;
    const auto grid64 = count / block + static_cast<std::uint64_t>(count % block != 0);
    if (grid64 > std::numeric_limits<unsigned int>::max())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-LAUNCH-OVERFLOW", "GPU grid dimension overflow");
    context.activate();
    CUdeviceptr outPointer = output.allocation->pointer;
    CUdeviceptr inputPointer = input.allocation->pointer;
    CUdeviceptr errorPointer = error ? error->pointer : 0;
    std::uint64_t elements = count;
    void* arguments[] = {&outPointer, &inputPointer, &elements, &errorPointer};
    driver.check(driver.cuLaunchKernel(module.function(kernelName(RegionOp::Negate, output.dtype)),
                                      static_cast<unsigned int>(grid64), 1, 1,
                                      static_cast<unsigned int>(block), 1, 1,
                                      0, stream.get(), arguments, nullptr), "cuLaunchKernel(negate)");
    ++evidence.kernelLaunches;
    ++evidence.streamSubmissions;
}

void copyToDevice(Driver& driver, Context& context, Stream& stream,
                  const DeviceAllocationPtr& allocation, const void* source,
                  std::size_t count, GpuExecutionEvidence& evidence) {
    if (count == 0) return;
    if (!source || !allocation || allocation->bytes != count)
        fail(GpuErrorCategory::RuntimeFailure, "GPU-TRANSFER-SIZE", "invalid host-to-device transfer size");
    context.activate();
    driver.check(driver.cuMemcpyHtoDAsync(allocation->pointer, source, count, stream.get()),
                 "cuMemcpyHtoDAsync");
    ++evidence.hostToDeviceCopies;
    ++evidence.streamSubmissions;
}

void copyFromDevice(Driver& driver, Context& context, Stream& stream, void* destination,
                    const DeviceAllocationPtr& allocation, std::size_t count,
                    GpuExecutionEvidence& evidence) {
    if (count == 0) return;
    if (!destination || !allocation || allocation->bytes != count)
        fail(GpuErrorCategory::RuntimeFailure, "GPU-TRANSFER-SIZE", "invalid device-to-host transfer size");
    context.activate();
    driver.check(driver.cuMemcpyDtoHAsync(destination, allocation->pointer, count, stream.get()),
                 "cuMemcpyDtoHAsync");
    ++evidence.deviceToHostCopies;
    ++evidence.streamSubmissions;
}

DeviceTensor upload(Driver& driver, Context& context, Stream& stream,
                    const storage::Tensor& tensor,
                    std::deque<std::vector<std::byte>>& staging,
                    GpuExecutionEvidence& evidence,
                    DeviceAllocationPtr destination = {}) {
    const auto& descriptor = tensor.descriptor();
    storage::verifyDescriptor(descriptor);
    if ((descriptor.dtype != storage::DType::I64 && descriptor.dtype != storage::DType::F32) ||
        (descriptor.shape.size() != 1 && descriptor.shape.size() != 2))
        fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-DTYPE-RANK",
             "native GPU accepts only rank-1/2 i64/f32 tensors");
    if (descriptor.view || descriptor.elementOffset != 0 || !tensor.isContiguousRowMajor())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-LAYOUT",
             "native GPU requires materialized offset-zero contiguous row-major tensors");
    if (destination && destination->bytes != descriptor.storage.byteLength())
        fail(GpuErrorCategory::RuntimeFailure, "GPU-PLAN-CAPACITY",
             "planned device allocation has incompatible byte capacity");
    DeviceTensor device{descriptor.dtype, descriptor.shape,
                        destination ? std::move(destination) :
                                      allocate(driver, context, descriptor.storage.byteLength(), evidence)};
    if (descriptor.storage.byteLength() == 0) return device;
    staging.emplace_back(descriptor.storage.byteLength());
    if (descriptor.dtype == storage::DType::I64) {
        const auto values = tensor.logicalI64Values();
        std::memcpy(staging.back().data(), values.data(), descriptor.storage.byteLength());
    } else {
        const auto values = tensor.logicalF32Values();
        std::memcpy(staging.back().data(), values.data(), descriptor.storage.byteLength());
    }
    copyToDevice(driver, context, stream, device.allocation, staging.back().data(),
                 descriptor.storage.byteLength(), evidence);
    ++evidence.inputStorageUploads;
    return device;
}

[[maybe_unused]] void launchBinary(Driver& driver, Context& context, Stream& stream, Module& module,
                  RegionOp op, const DeviceTensor& lhs, const DeviceTensor& rhs,
                  DeviceTensor& output, const DeviceAllocationPtr& error,
                  GpuExecutionEvidence& evidence) {
    if (lhs.dtype != rhs.dtype || lhs.dtype != output.dtype || lhs.shape != rhs.shape || lhs.shape != output.shape)
        fail(GpuErrorCategory::BackendUnsupported, "GPU-RUNTIME-SHAPE",
             "runtime elementwise inputs require equal dtype and shape");
    const auto count = storage::checkedElementCount(output.shape);
    if (count == 0) return;
    constexpr std::uint64_t block = 256;
    const auto grid64 = count / block + static_cast<std::uint64_t>(count % block != 0);
    if (grid64 > std::numeric_limits<unsigned int>::max())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-LAUNCH-OVERFLOW", "GPU grid dimension overflow");
    context.activate();
    CUdeviceptr outPointer = output.allocation->pointer;
    CUdeviceptr lhsPointer = lhs.allocation->pointer;
    CUdeviceptr rhsPointer = rhs.allocation->pointer;
    CUdeviceptr errorPointer = error ? error->pointer : 0;
    std::uint64_t elements = count;
    void* arguments[] = {&outPointer, &lhsPointer, &rhsPointer, &elements, &errorPointer};
    driver.check(driver.cuLaunchKernel(module.function(kernelName(op, output.dtype)),
                                      static_cast<unsigned int>(grid64), 1, 1,
                                      static_cast<unsigned int>(block), 1, 1,
                                      0, stream.get(), arguments, nullptr), "cuLaunchKernel");
    ++evidence.kernelLaunches;
    ++evidence.streamSubmissions;
}

void launchIndex(Driver& driver, Context& context, Stream& stream, Module& module,
                 const DeviceTensor& input, DeviceTensor& output, std::uint64_t index,
                 GpuExecutionEvidence& evidence) {
    context.activate();
    CUdeviceptr outPointer = output.allocation->pointer;
    CUdeviceptr inputPointer = input.allocation->pointer;
    void* arguments[] = {&outPointer, &inputPointer, &index};
    driver.check(driver.cuLaunchKernel(module.function(kernelName(RegionOp::Index, input.dtype)),
                                      1, 1, 1, 1, 1, 1, 0, stream.get(), arguments, nullptr),
                 "cuLaunchKernel(index)");
    ++evidence.kernelLaunches;
    ++evidence.streamSubmissions;
}

void copyDevice(Driver& driver, Context& context, Stream& stream,
                const DeviceTensor& input, DeviceTensor& output,
                GpuExecutionEvidence& evidence) {
    if (input.dtype != output.dtype || input.shape != output.shape ||
        input.allocation->bytes != output.allocation->bytes)
        fail(GpuErrorCategory::RuntimeFailure, "GPU-PLAN-COPY", "planned device copy is incompatible");
    if (input.allocation->bytes == 0) return;
    context.activate();
    driver.check(driver.cuMemcpyDtoDAsync(output.allocation->pointer, input.allocation->pointer,
                                         input.allocation->bytes, stream.get()),
                 "cuMemcpyDtoDAsync");
    ++evidence.deviceToDeviceCopies;
    ++evidence.streamSubmissions;
}

void launchGroup(Driver& driver, Context& context, Stream& stream, Module& module,
                 const TensorRegion& region, const FusionGroup& group,
                 const std::map<semantic::ValueId, DeviceValue>& values,
                 DeviceTensor& output, const DeviceAllocationPtr& error,
                 GpuExecutionEvidence& evidence) {
    const auto inputs = groupInputs(region, group);
    if (inputs.empty())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-PLAN",
             "elementwise fusion group has no materialized input");
    std::vector<const DeviceTensor*> tensors;
    tensors.reserve(inputs.size());
    for (auto id : inputs) {
        const auto found = values.find(id);
        if (found == values.end() || !std::holds_alternative<DeviceTensor>(found->second))
            fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-PLAN",
                 "fusion input is not materialized before launch");
        tensors.push_back(&std::get<DeviceTensor>(found->second));
    }
    for (const auto* input : tensors)
        if (input->dtype != output.dtype || input->shape != output.shape)
            fail(GpuErrorCategory::BackendUnsupported, "GPU-RUNTIME-SHAPE",
                 "runtime fused elementwise inputs require equal dtype and shape");
    const auto count = storage::checkedElementCount(output.shape);
    if (count == 0) return;
    constexpr std::uint64_t block = 256;
    const auto grid64 = count / block + static_cast<std::uint64_t>(count % block != 0);
    if (grid64 > std::numeric_limits<unsigned int>::max())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-LAUNCH-OVERFLOW", "GPU grid dimension overflow");
    context.activate();
    CUdeviceptr outPointer = output.allocation->pointer;
    std::vector<CUdeviceptr> inputPointers;
    inputPointers.reserve(tensors.size());
    for (const auto* input : tensors) inputPointers.push_back(input->allocation->pointer);
    std::uint64_t elements = count;
    CUdeviceptr errorPointer = error ? error->pointer : 0;
    std::vector<void*> arguments;
    arguments.reserve(inputPointers.size() + 3);
    arguments.push_back(&outPointer);
    for (auto& pointer : inputPointers) arguments.push_back(&pointer);
    arguments.push_back(&elements);
    arguments.push_back(&errorPointer);
    driver.check(driver.cuLaunchKernel(module.function(groupKernelName(group, output.dtype)),
                                      static_cast<unsigned int>(grid64), 1, 1,
                                      static_cast<unsigned int>(block), 1, 1,
                                      0, stream.get(), arguments.data(), nullptr),
                 "cuLaunchKernel(fusion-group)");
    ++evidence.kernelLaunches;
    ++evidence.streamSubmissions;
}

#endif

}

struct NativeGpuPendingState {
    TensorRegion region;
    PhysicalPlan plan;
    std::vector<GpuValue> inputs;
    runtime::AsyncResource output = runtime::AsyncResource::create();
    GpuExecutionEvidence evidence;
    std::optional<GpuValue> completedValue;
    std::optional<GpuError> failure;
#if THIRAN_ENABLE_NATIVE_GPU
    Driver driver;
    GpuDeviceInfo initializedDevice;
    Context context;
    Stream stream;
    Module module;
    Event completion;
    std::map<semantic::ValueId, DeviceValue> values;
    std::map<std::uint64_t, DeviceAllocationPtr> uploaded;
    std::map<PhysicalSlotId, DeviceAllocationPtr> slotAllocations;
    std::deque<std::vector<std::byte>> uploadStaging;
    DeviceAllocationPtr errorAllocation;
    std::uint32_t errorHost = 0;
    std::optional<DeviceTensor> deviceOutput;
    std::optional<GpuValue> immediateOutput;
    semantic::Type outputType;
    std::vector<std::byte> outputBytes;
    bool eventRecorded = false;
    bool finalized = false;

    NativeGpuPendingState(TensorRegion selectedRegion, PhysicalPlan selectedPlan,
                          std::vector<GpuValue> selectedInputs, int ordinal,
                          std::optional<std::string> selectedPtx = {})
        : region(std::move(selectedRegion)), plan(std::move(selectedPlan)), inputs(std::move(selectedInputs)),
          driver(), initializedDevice(initialize(driver, ordinal)), context(driver, ordinal),
          stream(driver, context),
          module(driver, context, selectedPtx ? std::move(*selectedPtx) : plannedPtx(region, plan)),
          completion(driver, context) {
        evidence.device = initializedDevice;
        evidence.logicalTensorValues = plan.values.size();
        evidence.physicalSlots = plan.slots.size();
        evidence.fusionGroups = plan.fusionGroups.size();
        for (const auto& node : region.nodes)
            if (node.type.kind == semantic::TypeKind::Tensor && node.op != RegionOp::Input &&
                node.op != RegionOp::Alias && node.id != region.output)
                ++evidence.logicalIntermediates;
        for (const auto& value : plan.values) {
            if (value.value == value.root && value.materialized &&
                value.classification == PhysicalValueClass::Temporary)
                ++evidence.materializedIntermediates;
        }
        for (const auto& slot : plan.slots) {
            if (slot.reusable && !slot.external) ++evidence.physicalTemporarySlots;
            std::uint64_t roots = 0;
            for (auto id : slot.values) {
                const auto* value = plan.value(id);
                roots += value && value->value == value->root;
            }
            if (roots > 1) evidence.reusedSlotAssignments += roots - 1;
        }
        for (const auto& group : plan.fusionGroups) evidence.fusedKernelGroups += group.fused;
    }

    DeviceAllocationPtr plannedAllocation(semantic::ValueId value, std::size_t byteCount) {
        const auto* mapping = plan.value(value);
        if (byteCount == 0) return allocate(driver, context, 0, evidence);
        if (!mapping || !mapping->slot)
            fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-PLAN",
                 "materialized tensor has no planned slot");
        const auto* slot = plan.slot(*mapping->slot);
        if (!slot || slot->device != PhysicalDevice::Gpu ||
            (slot->capacityBytes && *slot->capacityBytes < byteCount))
            fail(GpuErrorCategory::BackendUnsupported, "GPU-PLAN-CAPACITY",
                 "planned slot cannot satisfy runtime device allocation");
        auto found = slotAllocations.find(slot->id);
        if (found != slotAllocations.end()) {
            if (found->second->bytes != byteCount)
                fail(GpuErrorCategory::BackendUnsupported, "GPU-PLAN-CAPACITY",
                     "reused dynamic slot has incompatible runtime byte size");
            return found->second;
        }
        auto allocation = allocate(driver, context, byteCount, evidence);
        slotAllocations.emplace(slot->id, allocation);
        ++evidence.plannerOwnedAllocations;
        evidence.retainedPlannerAllocations = slotAllocations.size();
        return allocation;
    }

    void releaseDeviceStorage() noexcept {
        deviceOutput.reset();
        values.clear();
        uploaded.clear();
        slotAllocations.clear();
        errorAllocation.reset();
        uploadStaging.clear();
        evidence.retainedPlannerAllocations = 0;
    }

    void ensureErrorBuffer() {
        if (errorAllocation) return;
        errorAllocation = allocate(driver, context, sizeof(errorHost), evidence);
        errorHost = 0;
        copyToDevice(driver, context, stream, errorAllocation, &errorHost, sizeof(errorHost), evidence);
    }

    void submitWork() {
        context.activate();
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            const auto id = region.inputs[index];
            const auto node = std::find_if(region.nodes.begin(), region.nodes.end(),
                                           [&](const auto& candidate) { return candidate.id == id; });
            if (node == region.nodes.end())
                fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-REGION", "missing GPU input node");
            if (const auto* tensor = std::get_if<storage::Tensor>(&inputs[index])) {
                const auto& descriptor = tensor->descriptor();
                const auto key = tensor->storageId().value;
                if (!uploaded.contains(key)) {
                    auto deviceTensor = upload(driver, context, stream, *tensor, uploadStaging, evidence);
                    uploaded.emplace(key, deviceTensor.allocation);
                    values[id] = std::move(deviceTensor);
                } else {
                    if (uploaded.at(key)->bytes != descriptor.storage.byteLength())
                        fail(GpuErrorCategory::RuntimeFailure, "GPU-ALIASED-STORAGE-SIZE",
                             "aliased host descriptors disagree on physical byte length");
                    values[id] = DeviceTensor{descriptor.dtype, descriptor.shape, uploaded.at(key)};
                }
            } else if (const auto* integer = std::get_if<std::int64_t>(&inputs[index])) {
                values[id] = *integer;
            } else {
                values[id] = std::get<float>(inputs[index]);
            }
        }

        std::set<FusionGroupId> executedGroups;
        for (const auto& node : region.nodes) {
            if (node.op == RegionOp::Input) continue;
            if (node.op == RegionOp::Integer) values[node.id] = *node.integer;
            else if (node.op == RegionOp::Float) values[node.id] = *node.floating;
            else if (node.op == RegionOp::Alias) {
                const auto* mapping = plan.value(node.id);
                if (!mapping || mapping->materialized)
                    values[node.id] = values.at(canonicalAlias(region, node.dependencies.at(0)));
            }
            else if (node.op == RegionOp::TensorLiteral) {
                std::vector<std::uint64_t> shape;
                for (auto extent : node.shape.extents) {
                    if (!extent || *extent < 0)
                        fail(GpuErrorCategory::BackendUnsupported, "GPU-DYNAMIC-LITERAL",
                             "GPU literal shape is not concrete");
                    shape.push_back(static_cast<std::uint64_t>(*extent));
                }
                if (node.type.elements.at(0).kind == semantic::TypeKind::I64) {
                    std::vector<std::int64_t> host;
                    for (auto dependency : node.dependencies)
                        host.push_back(std::get<std::int64_t>(values.at(dependency)));
                    auto hostTensor = storage::Tensor::materializeI64(shape, host);
                    values[node.id] = upload(driver, context, stream,
                        hostTensor, uploadStaging, evidence,
                        plannedAllocation(node.id, hostTensor.descriptor().storage.byteLength()));
                } else {
                    std::vector<float> host;
                    for (auto dependency : node.dependencies)
                        host.push_back(std::get<float>(values.at(dependency)));
                    auto hostTensor = storage::Tensor::materializeF32(shape, host);
                    values[node.id] = upload(driver, context, stream,
                        hostTensor, uploadStaging, evidence,
                        plannedAllocation(node.id, hostTensor.descriptor().storage.byteLength()));
                }
            } else if (node.op == RegionOp::Copy) {
                const auto source = canonicalAlias(region, node.dependencies.at(0));
                const auto& input = std::get<DeviceTensor>(values.at(source));
                DeviceTensor outputTensor{input.dtype, input.shape,
                    plannedAllocation(node.id, bytes(input))};
                copyDevice(driver, context, stream, input, outputTensor, evidence);
                values[node.id] = std::move(outputTensor);
            } else if (node.op == RegionOp::Negate || node.op == RegionOp::Add ||
                       node.op == RegionOp::Subtract || node.op == RegionOp::ElementMultiply) {
                const auto* group = plan.groupFor(node.id);
                if (!group || group->nodes.front() != node.id || !executedGroups.insert(group->id).second)
                    continue;
                const auto groupInputIds = groupInputs(region, *group);
                if (groupInputIds.empty())
                    fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-PLAN",
                         "fusion group has no runtime tensor input");
                const auto& input = std::get<DeviceTensor>(values.at(groupInputIds.front()));
                DeviceTensor outputTensor{input.dtype, input.shape,
                    plannedAllocation(group->output, bytes(input))};
                if (input.dtype == storage::DType::I64 && storage::checkedElementCount(input.shape) != 0)
                    ensureErrorBuffer();
                launchGroup(driver, context, stream, module, region, *group, values,
                            outputTensor, errorAllocation, evidence);
                values[group->output] = std::move(outputTensor);
            } else if (node.op == RegionOp::Index) {
                const auto& input = std::get<DeviceTensor>(
                    values.at(canonicalAlias(region, node.dependencies.at(0))));
                std::uint64_t linear = 0;
                for (std::size_t axis = 0; axis < node.indices.size(); ++axis) {
                    const auto coordinate = std::get<std::int64_t>(values.at(node.indices[axis]));
                    if (coordinate < 0 || static_cast<std::uint64_t>(coordinate) >= input.shape.at(axis))
                        fail(GpuErrorCategory::SemanticFailure, "TH-SPEC-BOUNDS", "GPU index is out of bounds");
                    linear = storage::checkedAdd(storage::checkedMultiply(linear, input.shape.at(axis)),
                                                 static_cast<std::uint64_t>(coordinate));
                }
                DeviceTensor scalar{input.dtype, {},
                    allocate(driver, context, storage::elementWidth(input.dtype), evidence)};
                launchIndex(driver, context, stream, module, input, scalar, linear, evidence);
                values[node.id] = std::move(scalar);
            } else {
                fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-OP",
                     "TensorRegion operation has no native GPU kernel");
            }
        }

        const auto outputNode = std::find_if(region.nodes.begin(), region.nodes.end(),
            [&](const auto& node) { return node.id == region.output; });
        if (outputNode == region.nodes.end())
            fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-REGION", "missing GPU output node");
        outputType = outputNode->type;
        const auto& result = values.at(region.output);
        if (const auto* tensor = std::get_if<DeviceTensor>(&result)) {
            deviceOutput = *tensor;
            outputBytes.resize(bytes(*tensor));
            copyFromDevice(driver, context, stream, outputBytes.data(), tensor->allocation,
                           outputBytes.size(), evidence);
            if (!outputBytes.empty()) ++evidence.resultDownloads;
        } else if (const auto* integer = std::get_if<std::int64_t>(&result)) {
            immediateOutput = *integer;
        } else {
            immediateOutput = std::get<float>(result);
        }
        if (errorAllocation)
            copyFromDevice(driver, context, stream, &errorHost, errorAllocation, sizeof(errorHost), evidence);
        context.activate();
        driver.check(driver.cuEventRecord(completion.get(), stream.get()), "cuEventRecord");
        eventRecorded = true;
        ++evidence.eventRecords;
        ++evidence.streamSubmissions;
    }

    void finalizeValue() {
        if (finalized) return;
        if (errorAllocation && errorHost != 0)
            fail(GpuErrorCategory::SemanticFailure, "TH-SPEC-I64-OVERFLOW",
                 "checked i64 GPU arithmetic overflow");
        if (immediateOutput) completedValue = *immediateOutput;
        else if (deviceOutput) {
            const auto count = storage::checkedElementCount(deviceOutput->shape);
            if (outputType.kind == semantic::TypeKind::Tensor) {
                if (deviceOutput->dtype == storage::DType::I64) {
                    std::vector<std::int64_t> host(count);
                    if (!outputBytes.empty()) std::memcpy(host.data(), outputBytes.data(), outputBytes.size());
                    completedValue = storage::Tensor::materializeI64(deviceOutput->shape, host);
                } else {
                    std::vector<float> host(count);
                    if (!outputBytes.empty()) std::memcpy(host.data(), outputBytes.data(), outputBytes.size());
                    completedValue = storage::Tensor::materializeF32(deviceOutput->shape, host);
                }
            } else if (deviceOutput->dtype == storage::DType::I64) {
                std::int64_t host = 0;
                std::memcpy(&host, outputBytes.data(), sizeof(host));
                completedValue = host;
            } else {
                float host = 0;
                std::memcpy(&host, outputBytes.data(), sizeof(host));
                completedValue = host;
            }
        }
        if (!completedValue)
            fail(GpuErrorCategory::RuntimeFailure, "GPU-OUTPUT-MISSING", "GPU output was not materialized");
        finalized = true;
    }
#else
    NativeGpuPendingState(TensorRegion selectedRegion, PhysicalPlan selectedPlan,
                          std::vector<GpuValue> selectedInputs, int,
                          std::optional<std::string> = {})
        : region(std::move(selectedRegion)), plan(std::move(selectedPlan)),
          inputs(std::move(selectedInputs)) {}
#endif
};

namespace {

#if THIRAN_ENABLE_NATIVE_GPU
GpuError gpuInternal(const std::exception& error) {
    const std::string message = error.what();
    if (message == "TH007-SIZE-OVERFLOW" || message == "TH-SPEC-SHAPE")
        return {GpuErrorCategory::BackendUnsupported, "GPU-SIZE-OVERFLOW", message};
    return {GpuErrorCategory::RuntimeFailure, "GPU-INTERNAL", message};
}

runtime::AsyncError asyncError(const GpuError& error) { return {error.code, error.message}; }
#endif

void mergeEvidence(GpuExecutionEvidence& evidence, const runtime::AsyncOperationEvidence& operation) {
    evidence.submittedOperations = operation.submitted;
    evidence.pendingOperations = operation.pending;
    evidence.activeReadReservations = operation.activeReadReservations;
    evidence.activeWriteReservations = operation.activeWriteReservations;
    evidence.observations = operation.observations;
    evidence.droppedDrains = operation.droppedDrains;
    evidence.releasedReservations = operation.releasedReservations;
}

}

std::string emitNativeGpuPtx() {
#if THIRAN_ENABLE_NATIVE_GPU
    return ptx();
#else
    return {};
#endif
}

std::string emitNativeGpuPtx(const TensorRegion& region, PhysicalPlanOptions options) {
#if THIRAN_ENABLE_NATIVE_GPU
    auto planned = buildPhysicalPlan(region, PhysicalDevice::Gpu, options);
    if (!planned.ok())
        throw std::invalid_argument(planned.errors.empty() ? "GPU-INVALID-PLAN" : planned.errors.front());
    return plannedPtx(region, *planned.plan);
#else
    (void)region;
    (void)options;
    return {};
#endif
}

GpuDeviceInfo probeNativeGpu(int device) noexcept {
    GpuDeviceInfo info;
    info.backendBuilt = nativeGpuBackendBuilt();
#if THIRAN_ENABLE_NATIVE_GPU
    try {
        Driver driver;
        info.driverLoaded = true;
        return initialize(driver, device);
    } catch (const Failure& failure) {
        info.error = failure.error();
        return info;
    } catch (const std::exception& error) {
        info.error = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-INTERNAL", error.what()};
        return info;
    }
#else
    (void)device;
    info.error = GpuError{GpuErrorCategory::BackendUnavailable, "GPU-BACKEND-NOT-BUILT",
                          "native GPU backend was disabled at build time"};
    return info;
#endif
}

PendingGpuExecution::PendingGpuExecution(std::shared_ptr<NativeGpuPendingState> state,
                                         runtime::AsyncOperation operation)
    : state_(std::move(state)), operation_(std::move(operation)) {}

bool PendingGpuExecution::valid() const noexcept { return state_ && operation_.valid(); }
runtime::AsyncOperationState PendingGpuExecution::state() const noexcept { return operation_.state(); }
bool PendingGpuExecution::observed() const noexcept { return operation_.observed(); }
std::optional<GpuValue> PendingGpuExecution::value() const {
    if (!state_ || !operation_.observed() || state_->failure) return {};
    return state_->completedValue;
}
runtime::AsyncResource PendingGpuExecution::outputResource() const {
    return state_ ? state_->output : runtime::AsyncResource{};
}
GpuExecutionEvidence PendingGpuExecution::evidence() const noexcept {
    if (!state_) return {};
    auto result = state_->evidence;
    mergeEvidence(result, operation_.evidence());
    return result;
}
GpuExecutionResult PendingGpuExecution::observe() noexcept {
    GpuExecutionResult result;
    if (!valid()) {
        result.error = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-INVALID-ASYNC-HANDLE",
                                "native GPU async handle is invalid"};
        return result;
    }
    const auto observedResult = operation_.observe();
    if (!observedResult.success) {
        if (state_->failure) result.error = state_->failure;
        else if (observedResult.error)
            result.error = GpuError{GpuErrorCategory::RuntimeFailure, observedResult.error->code,
                                    observedResult.error->message};
        else
            result.error = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-ASYNC-OBSERVE",
                                    "native GPU observation failed"};
    } else if (!state_->completedValue) {
        result.error = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-OUTPUT-MISSING",
                                "native GPU observation produced no value"};
    } else {
        result.value = state_->completedValue;
    }
    result.evidence = evidence();
    return result;
}

#if THIRAN_ENABLE_NATIVE_GPU
namespace {
struct PreparedGpuSubmission {
    std::shared_ptr<NativeGpuPendingState> state;
    std::optional<runtime::AsyncOperation> operation;
    std::optional<GpuError> error;
};

PreparedGpuSubmission prepareGpuSubmission(std::shared_ptr<NativeGpuPendingState> state,
                                           const std::vector<GpuValue>& inputs) {
    std::vector<runtime::AsyncResourceAccess> accesses;
    for (const auto& input : inputs)
        if (const auto* tensor = std::get_if<storage::Tensor>(&input))
            accesses.push_back({tensor->asyncResource(), runtime::AsyncReservationKind::Read});
    accesses.push_back({state->output, runtime::AsyncReservationKind::Write});

    runtime::AsyncBackendCallbacks callbacks;
    callbacks.submit = [state]() -> std::optional<runtime::AsyncError> {
        try { state->submitWork(); return {}; }
        catch (const Failure& failure) { state->failure = failure.error(); }
        catch (const std::exception& error) { state->failure = gpuInternal(error); }
        catch (...) {
            state->failure = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-INTERNAL",
                                      "unknown native GPU submission failure"};
        }
        return asyncError(*state->failure);
    };
    callbacks.poll = [state] {
        try {
            if (!state->eventRecorded) return runtime::AsyncPollResult{};
            state->context.activate();
            const auto status = state->driver.cuEventQuery(state->completion.get());
            if (status == cudaErrorNotReady) return runtime::AsyncPollResult{};
            state->driver.check(status, "cuEventQuery");
            if (state->errorAllocation && state->errorHost != 0) {
                state->failure = GpuError{GpuErrorCategory::SemanticFailure, "TH-SPEC-I64-OVERFLOW",
                                          "checked i64 GPU arithmetic overflow"};
                return runtime::AsyncPollResult{runtime::AsyncPollState::Failed,
                                                asyncError(*state->failure)};
            }
            return runtime::AsyncPollResult{runtime::AsyncPollState::Completed, {}};
        } catch (const Failure& failure) { state->failure = failure.error(); }
        catch (const std::exception& error) { state->failure = gpuInternal(error); }
        return runtime::AsyncPollResult{runtime::AsyncPollState::Failed, asyncError(*state->failure)};
    };
    callbacks.wait = [state]() -> std::optional<runtime::AsyncError> {
        try {
            state->context.activate();
            if (state->eventRecorded) {
                state->driver.check(state->driver.cuEventSynchronize(state->completion.get()),
                                    "cuEventSynchronize");
                ++state->evidence.synchronizations;
            }
            if (state->failure) {
                auto error = asyncError(*state->failure);
                state->releaseDeviceStorage();
                return error;
            }
            state->finalizeValue();
            state->releaseDeviceStorage();
            return {};
        } catch (const Failure& failure) {
            state->failure = failure.error();
            state->releaseDeviceStorage();
        } catch (const std::exception& error) {
            state->failure = gpuInternal(error);
            state->releaseDeviceStorage();
        } catch (...) {
            state->failure = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-INTERNAL",
                                      "unknown native GPU observation failure"};
            state->releaseDeviceStorage();
        }
        return asyncError(*state->failure);
    };
    callbacks.abort = [state]() -> std::optional<runtime::AsyncError> {
        try {
            state->context.activate();
            if (state->evidence.streamSubmissions != 0) {
                state->driver.check(state->driver.cuStreamSynchronize(state->stream.get()),
                                    "cuStreamSynchronize(abort)");
                ++state->evidence.synchronizations;
            }
            state->releaseDeviceStorage();
            return {};
        } catch (const Failure& failure) {
            state->releaseDeviceStorage();
            return asyncError(failure.error());
        } catch (const std::exception& error) {
            state->releaseDeviceStorage();
            return asyncError(gpuInternal(error));
        } catch (...) {
            state->releaseDeviceStorage();
            return runtime::AsyncError{"GPU-INTERNAL", "unknown native GPU abort failure"};
        }
    };

    auto submitted = runtime::submitAsyncOperation(std::move(accesses), {}, std::move(callbacks));
    if (!submitted.ok()) {
        auto error = state->failure.value_or(GpuError{GpuErrorCategory::RuntimeFailure,
            submitted.error ? submitted.error->code : "GPU-ASYNC-SUBMIT",
            submitted.error ? submitted.error->message : "native GPU async submission failed"});
        return {std::move(state), {}, std::move(error)};
    }
    return {std::move(state), std::move(submitted.operation), {}};
}
} // namespace
#endif

GpuAsyncSubmission submitNativeGpuAsync(const TensorRegion& region,
                                        const std::vector<GpuValue>& inputs,
                                        int device,
                                        PhysicalPlanOptions options) noexcept {
    GpuAsyncSubmission result;
    result.evidence.device.backendBuilt = nativeGpuBackendBuilt();
#if THIRAN_ENABLE_NATIVE_GPU
    try {
        preflight(region, inputs);
        auto planned = buildPhysicalPlan(region, PhysicalDevice::Gpu, options);
        if (!planned.ok())
            fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-PLAN",
                 planned.errors.empty() ? "physical planning failed" : planned.errors.front());
        auto state = std::make_shared<NativeGpuPendingState>(
            region, std::move(*planned.plan), inputs, device);
        auto prepared = prepareGpuSubmission(std::move(state), inputs);
        if (prepared.error) {
            result.error = std::move(prepared.error);
            result.evidence = prepared.state->evidence;
            return result;
        }
        result.pending = PendingGpuExecution(prepared.state, std::move(*prepared.operation));
        result.evidence = result.pending->evidence();
        return result;
    } catch (const Failure& failure) {
        result.error = failure.error();
    } catch (const std::exception& error) {
        result.error = gpuInternal(error);
    } catch (...) {
        result.error = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-INTERNAL",
                                "unknown native GPU setup failure"};
    }
#else
    (void)region;
    (void)inputs;
    (void)device;
    (void)options;
    result.error = GpuError{GpuErrorCategory::BackendUnavailable, "GPU-BACKEND-NOT-BUILT",
                            "native GPU backend was disabled at build time"};
#endif
    return result;
}

GpuAsyncSubmission submitNativeGpuPayloadAsync(const TensorRegion& region,
                                               const PhysicalPlan& plan,
                                               std::string ptxSource,
                                               const std::vector<GpuValue>& inputs,
                                               int device) noexcept {
    GpuAsyncSubmission result;
    result.evidence.device.backendBuilt = nativeGpuBackendBuilt();
#if THIRAN_ENABLE_NATIVE_GPU
    try {
        preflight(region, inputs);
        if (ptxSource.empty())
            fail(GpuErrorCategory::BackendUnsupported, "GPU-MISSING-PTX",
                 "persistent GPU payload is empty");
        if (plan.device != PhysicalDevice::Gpu)
            fail(GpuErrorCategory::BackendUnsupported, "GPU-PLAN-TARGET",
                 "persistent plan does not target GPU");
        const auto verified = verifyPhysicalPlan(region, plan);
        if (!verified.ok())
            fail(GpuErrorCategory::BackendUnsupported, "GPU-INVALID-PLAN",
                 verified.errors.empty() ? "persistent physical plan failed verification" :
                                           verified.errors.front());
        auto state = std::make_shared<NativeGpuPendingState>(
            region, plan, inputs, device, std::move(ptxSource));
        auto prepared = prepareGpuSubmission(std::move(state), inputs);
        if (prepared.error) {
            result.error = std::move(prepared.error);
            result.evidence = prepared.state->evidence;
            return result;
        }
        result.pending = PendingGpuExecution(prepared.state, std::move(*prepared.operation));
        result.evidence = result.pending->evidence();
        return result;
    } catch (const Failure& failure) {
        result.error = failure.error();
    } catch (const std::exception& error) {
        result.error = gpuInternal(error);
    } catch (...) {
        result.error = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-INTERNAL",
                                "unknown persistent GPU setup failure"};
    }
#else
    (void)region;
    (void)plan;
    (void)ptxSource;
    (void)inputs;
    (void)device;
    result.error = GpuError{GpuErrorCategory::BackendUnavailable, "GPU-BACKEND-NOT-BUILT",
                            "native GPU backend was disabled at build time"};
#endif
    return result;
}

GpuExecutionResult executeNativeGpu(const TensorRegion& region,
                                    const std::vector<GpuValue>& inputs,
                                    int device,
                                    PhysicalPlanOptions options) noexcept {
    auto submitted = submitNativeGpuAsync(region, inputs, device, options);
    if (!submitted.ok()) return {{}, submitted.error, submitted.evidence};
    return submitted.pending->observe();
}

GpuExecutionResult executeNativeGpuPayload(const TensorRegion& region,
                                           const PhysicalPlan& plan,
                                           std::string ptxSource,
                                           const std::vector<GpuValue>& inputs,
                                           int device) noexcept {
    auto submitted = submitNativeGpuPayloadAsync(
        region, plan, std::move(ptxSource), inputs, device);
    if (!submitted.ok()) return {{}, submitted.error, submitted.evidence};
    return submitted.pending->observe();
}

}
