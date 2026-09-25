#include "backend/v0/NativeGpu.hpp"

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <map>
#include <memory>
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
using CUcontext = CUctx_st*;
using CUmodule = CUmod_st*;
using CUfunction = CUfunc_st*;
using CUstream = void*;
using CUresult = int;

constexpr CUresult cudaSuccess = 0;

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
        load(cuMemAlloc, "cuMemAlloc_v2");
        load(cuMemFree, "cuMemFree_v2");
        load(cuMemcpyHtoD, "cuMemcpyHtoD_v2");
        load(cuMemcpyDtoH, "cuMemcpyDtoH_v2");
        load(cuModuleLoadDataEx, "cuModuleLoadDataEx");
        load(cuModuleUnload, "cuModuleUnload");
        load(cuModuleGetFunction, "cuModuleGetFunction");
        load(cuLaunchKernel, "cuLaunchKernel");
        load(cuCtxSynchronize, "cuCtxSynchronize");
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
    CUresult (*cuMemAlloc)(CUdeviceptr*, std::size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, std::size_t) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, std::size_t) = nullptr;
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, void*, void*) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, CUstream, void**, void**) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;
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
    ~Context() { if (context_) (void)driver_.cuCtxDestroy(context_); }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
private:
    Driver& driver_;
    CUcontext context_ = nullptr;
};

struct DeviceAllocation final {
    Driver* driver = nullptr;
    CUdeviceptr pointer = 0;
    std::size_t bytes = 0;
    ~DeviceAllocation() { if (pointer && driver) (void)driver->cuMemFree(pointer); }
};

using DeviceAllocationPtr = std::shared_ptr<DeviceAllocation>;

DeviceAllocationPtr allocate(Driver& driver, std::size_t bytes, GpuExecutionEvidence& evidence) {
    auto allocation = std::make_shared<DeviceAllocation>();
    allocation->driver = &driver;
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

class Module final {
public:
    explicit Module(Driver& driver) : driver_(driver) {
        source_ = ptx();
        driver_.check(driver_.cuModuleLoadDataEx(&module_, source_.c_str(), 0, nullptr, nullptr),
                      "cuModuleLoadDataEx", GpuErrorCategory::BackendUnsupported);
    }
    ~Module() { if (module_) (void)driver_.cuModuleUnload(module_); }
    CUfunction function(const std::string& name) {
        CUfunction function = nullptr;
        driver_.check(driver_.cuModuleGetFunction(&function, module_, name.c_str()), "cuModuleGetFunction");
        return function;
    }
private:
    Driver& driver_;
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

void launchUnary(Driver& driver, Module& module, const DeviceTensor& input,
                 DeviceTensor& output, const DeviceAllocationPtr& error,
                 GpuExecutionEvidence& evidence) {
    const auto count = storage::checkedElementCount(output.shape);
    if (count == 0) return;
    constexpr std::uint64_t block = 256;
    const auto grid64 = count / block + static_cast<std::uint64_t>(count % block != 0);
    if (grid64 > std::numeric_limits<unsigned int>::max())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-LAUNCH-OVERFLOW", "GPU grid dimension overflow");
    std::uint32_t zero = 0;
    if (output.dtype == storage::DType::I64)
        driver.check(driver.cuMemcpyHtoD(error->pointer, &zero, sizeof(zero)), "cuMemcpyHtoD(error)");
    CUdeviceptr outPointer = output.allocation->pointer;
    CUdeviceptr inputPointer = input.allocation->pointer;
    CUdeviceptr errorPointer = error ? error->pointer : 0;
    std::uint64_t elements = count;
    void* arguments[] = {&outPointer, &inputPointer, &elements, &errorPointer};
    driver.check(driver.cuLaunchKernel(module.function(kernelName(RegionOp::Negate, output.dtype)),
                                      static_cast<unsigned int>(grid64), 1, 1,
                                      static_cast<unsigned int>(block), 1, 1,
                                      0, nullptr, arguments, nullptr), "cuLaunchKernel(negate)");
    ++evidence.kernelLaunches;
    driver.check(driver.cuCtxSynchronize(), "cuCtxSynchronize(negate)");
    ++evidence.synchronizations;
    if (output.dtype == storage::DType::I64) {
        std::uint32_t overflow = 0;
        driver.check(driver.cuMemcpyDtoH(&overflow, error->pointer, sizeof(overflow)), "cuMemcpyDtoH(error)");
        if (overflow)
            fail(GpuErrorCategory::SemanticFailure, "TH-SPEC-I64-OVERFLOW",
                 "checked i64 GPU negate overflow");
    }
}

void copyToDevice(Driver& driver, const DeviceAllocationPtr& allocation,
                  const void* source, std::size_t count, GpuExecutionEvidence& evidence) {
    if (count == 0) return;
    if (!source || !allocation || allocation->bytes != count)
        fail(GpuErrorCategory::RuntimeFailure, "GPU-TRANSFER-SIZE", "invalid host-to-device transfer size");
    driver.check(driver.cuMemcpyHtoD(allocation->pointer, source, count), "cuMemcpyHtoD");
    ++evidence.hostToDeviceCopies;
}

void copyFromDevice(Driver& driver, void* destination, const DeviceAllocationPtr& allocation,
                    std::size_t count, GpuExecutionEvidence& evidence) {
    if (count == 0) return;
    if (!destination || !allocation || allocation->bytes != count)
        fail(GpuErrorCategory::RuntimeFailure, "GPU-TRANSFER-SIZE", "invalid device-to-host transfer size");
    driver.check(driver.cuMemcpyDtoH(destination, allocation->pointer, count), "cuMemcpyDtoH");
    ++evidence.deviceToHostCopies;
}

DeviceTensor upload(Driver& driver, const storage::Tensor& tensor, GpuExecutionEvidence& evidence) {
    const auto& descriptor = tensor.descriptor();
    storage::verifyDescriptor(descriptor);
    if ((descriptor.dtype != storage::DType::I64 && descriptor.dtype != storage::DType::F32) ||
        (descriptor.shape.size() != 1 && descriptor.shape.size() != 2))
        fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-DTYPE-RANK",
             "native GPU accepts only rank-1/2 i64/f32 tensors");
    if (descriptor.view || descriptor.elementOffset != 0 || !tensor.isContiguousRowMajor())
        fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-LAYOUT",
             "native GPU requires materialized offset-zero contiguous row-major tensors");
    DeviceTensor device{descriptor.dtype, descriptor.shape, allocate(driver, descriptor.storage.byteLength(), evidence)};
    if (descriptor.dtype == storage::DType::I64) {
        const auto values = tensor.logicalI64Values();
        copyToDevice(driver, device.allocation, values.data(), descriptor.storage.byteLength(), evidence);
    } else {
        const auto values = tensor.logicalF32Values();
        copyToDevice(driver, device.allocation, values.data(), descriptor.storage.byteLength(), evidence);
    }
    return device;
}

void launchBinary(Driver& driver, Module& module, RegionOp op, const DeviceTensor& lhs,
                  const DeviceTensor& rhs, DeviceTensor& output, const DeviceAllocationPtr& error,
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
    std::uint32_t zero = 0;
    if (output.dtype == storage::DType::I64)
        driver.check(driver.cuMemcpyHtoD(error->pointer, &zero, sizeof(zero)), "cuMemcpyHtoD(error)");
    CUdeviceptr outPointer = output.allocation->pointer;
    CUdeviceptr lhsPointer = lhs.allocation->pointer;
    CUdeviceptr rhsPointer = rhs.allocation->pointer;
    CUdeviceptr errorPointer = error ? error->pointer : 0;
    std::uint64_t elements = count;
    void* arguments[] = {&outPointer, &lhsPointer, &rhsPointer, &elements, &errorPointer};
    driver.check(driver.cuLaunchKernel(module.function(kernelName(op, output.dtype)),
                                      static_cast<unsigned int>(grid64), 1, 1,
                                      static_cast<unsigned int>(block), 1, 1,
                                      0, nullptr, arguments, nullptr), "cuLaunchKernel");
    ++evidence.kernelLaunches;
    driver.check(driver.cuCtxSynchronize(), "cuCtxSynchronize");
    ++evidence.synchronizations;
    if (output.dtype == storage::DType::I64) {
        std::uint32_t overflow = 0;
        driver.check(driver.cuMemcpyDtoH(&overflow, error->pointer, sizeof(overflow)), "cuMemcpyDtoH(error)");
        if (overflow)
            fail(GpuErrorCategory::SemanticFailure, "TH-SPEC-I64-OVERFLOW",
                 "checked i64 GPU arithmetic overflow");
    }
}

void launchIndex(Driver& driver, Module& module, const DeviceTensor& input,
                 DeviceTensor& output, std::uint64_t index, GpuExecutionEvidence& evidence) {
    CUdeviceptr outPointer = output.allocation->pointer;
    CUdeviceptr inputPointer = input.allocation->pointer;
    void* arguments[] = {&outPointer, &inputPointer, &index};
    driver.check(driver.cuLaunchKernel(module.function(kernelName(RegionOp::Index, input.dtype)),
                                      1, 1, 1, 1, 1, 1, 0, nullptr, arguments, nullptr),
                 "cuLaunchKernel(index)");
    ++evidence.kernelLaunches;
    driver.check(driver.cuCtxSynchronize(), "cuCtxSynchronize(index)");
    ++evidence.synchronizations;
}

#endif

}

std::string emitNativeGpuPtx() {
#if THIRAN_ENABLE_NATIVE_GPU
    return ptx();
#else
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

GpuExecutionResult executeNativeGpu(const TensorRegion& region,
                                    const std::vector<GpuValue>& inputs,
                                    int device) noexcept {
    GpuExecutionResult result;
    result.evidence.device.backendBuilt = nativeGpuBackendBuilt();
#if THIRAN_ENABLE_NATIVE_GPU
    try {
        preflight(region, inputs);
        Driver driver;
        result.evidence.device.driverLoaded = true;
        result.evidence.device = initialize(driver, device);
        Context context(driver, device);
        std::map<semantic::ValueId, DeviceValue> values;
        std::map<std::uint64_t, DeviceAllocationPtr> uploaded;
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
                    auto deviceTensor = upload(driver, *tensor, result.evidence);
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

        Module module(driver);
        DeviceAllocationPtr errorAllocation;
        for (const auto& node : region.nodes) {
            if (node.op == RegionOp::Input) continue;
            if (node.op == RegionOp::Integer) values[node.id] = *node.integer;
            else if (node.op == RegionOp::Float) values[node.id] = *node.floating;
            else if (node.op == RegionOp::Alias) values[node.id] = values.at(node.dependencies.at(0));
            else if (node.op == RegionOp::TensorLiteral) {
                std::vector<std::uint64_t> shape;
                for (auto extent : node.shape.extents) {
                    if (!extent || *extent < 0)
                        fail(GpuErrorCategory::BackendUnsupported, "GPU-DYNAMIC-LITERAL", "GPU literal shape is not concrete");
                    shape.push_back(static_cast<std::uint64_t>(*extent));
                }
                if (node.type.elements.at(0).kind == semantic::TypeKind::I64) {
                    std::vector<std::int64_t> host;
                    for (auto dependency : node.dependencies) host.push_back(std::get<std::int64_t>(values.at(dependency)));
                    values[node.id] = upload(driver, storage::Tensor::materializeI64(shape, host), result.evidence);
                } else {
                    std::vector<float> host;
                    for (auto dependency : node.dependencies) host.push_back(std::get<float>(values.at(dependency)));
                    values[node.id] = upload(driver, storage::Tensor::materializeF32(shape, host), result.evidence);
                }
            } else if (node.op == RegionOp::Negate) {
                const auto& input = std::get<DeviceTensor>(values.at(node.dependencies.at(0)));
                DeviceTensor output{input.dtype, input.shape, allocate(driver, bytes(input), result.evidence)};
                if (input.dtype == storage::DType::I64 && storage::checkedElementCount(input.shape) != 0 &&
                    !errorAllocation)
                    errorAllocation = allocate(driver, sizeof(std::uint32_t), result.evidence);
                launchUnary(driver, module, input, output, errorAllocation, result.evidence);
                values[node.id] = std::move(output);
            } else if (node.op == RegionOp::Add || node.op == RegionOp::Subtract ||
                       node.op == RegionOp::ElementMultiply) {
                const auto& lhs = std::get<DeviceTensor>(values.at(node.dependencies.at(0)));
                const auto& rhs = std::get<DeviceTensor>(values.at(node.dependencies.at(1)));
                DeviceTensor output{lhs.dtype, lhs.shape, allocate(driver, bytes(lhs), result.evidence)};
                if (lhs.dtype == storage::DType::I64 && storage::checkedElementCount(lhs.shape) != 0 &&
                    !errorAllocation)
                    errorAllocation = allocate(driver, sizeof(std::uint32_t), result.evidence);
                launchBinary(driver, module, node.op, lhs, rhs, output, errorAllocation, result.evidence);
                values[node.id] = std::move(output);
            } else if (node.op == RegionOp::Index) {
                const auto& input = std::get<DeviceTensor>(values.at(node.dependencies.at(0)));
                std::uint64_t linear = 0;
                for (std::size_t axis = 0; axis < node.indices.size(); ++axis) {
                    const auto coordinate = std::get<std::int64_t>(values.at(node.indices[axis]));
                    if (coordinate < 0 || static_cast<std::uint64_t>(coordinate) >= input.shape.at(axis))
                        fail(GpuErrorCategory::SemanticFailure, "TH-SPEC-BOUNDS", "GPU index is out of bounds");
                    linear = storage::checkedAdd(storage::checkedMultiply(linear, input.shape.at(axis)),
                                                 static_cast<std::uint64_t>(coordinate));
                }
                DeviceTensor scalar{input.dtype, {}, allocate(driver, storage::elementWidth(input.dtype), result.evidence)};
                launchIndex(driver, module, input, scalar, linear, result.evidence);
                if (input.dtype == storage::DType::I64) {
                    std::int64_t host = 0;
                    copyFromDevice(driver, &host, scalar.allocation, sizeof(host), result.evidence);
                    values[node.id] = host;
                } else {
                    float host = 0;
                    copyFromDevice(driver, &host, scalar.allocation, sizeof(host), result.evidence);
                    values[node.id] = host;
                }
            } else {
                fail(GpuErrorCategory::BackendUnsupported, "GPU-UNSUPPORTED-OP",
                     "TensorRegion operation has no native GPU kernel");
            }
        }

        const auto& output = values.at(region.output);
        if (const auto* tensor = std::get_if<DeviceTensor>(&output)) {
            if (tensor->dtype == storage::DType::I64) {
                std::vector<std::int64_t> host(storage::checkedElementCount(tensor->shape));
                copyFromDevice(driver, host.data(), tensor->allocation, bytes(*tensor), result.evidence);
                result.value = storage::Tensor::materializeI64(tensor->shape, host);
            } else {
                std::vector<float> host(storage::checkedElementCount(tensor->shape));
                copyFromDevice(driver, host.data(), tensor->allocation, bytes(*tensor), result.evidence);
                result.value = storage::Tensor::materializeF32(tensor->shape, host);
            }
        } else if (const auto* integer = std::get_if<std::int64_t>(&output)) {
            result.value = *integer;
        } else {
            result.value = std::get<float>(output);
        }
    } catch (const Failure& failure) {
        result.error = failure.error();
    } catch (const std::exception& error) {
        const std::string message = error.what();
        if (message == "TH007-SIZE-OVERFLOW" || message == "TH-SPEC-SHAPE")
            result.error = GpuError{GpuErrorCategory::BackendUnsupported, "GPU-SIZE-OVERFLOW", message};
        else
            result.error = GpuError{GpuErrorCategory::RuntimeFailure, "GPU-INTERNAL", message};
    }
#else
    (void)region;
    (void)inputs;
    (void)device;
    result.error = GpuError{GpuErrorCategory::BackendUnavailable, "GPU-BACKEND-NOT-BUILT",
                            "native GPU backend was disabled at build time"};
#endif
    return result;
}

}
