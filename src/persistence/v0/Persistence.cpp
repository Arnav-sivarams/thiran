#include "persistence/v0/Persistence.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace thiran::v0::persistence {

void BinaryWriter::u8(std::uint8_t value) { bytes_.push_back(static_cast<std::byte>(value)); }
void BinaryWriter::u32(std::uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) u8(static_cast<std::uint8_t>(value >> shift));
}
void BinaryWriter::u64(std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) u8(static_cast<std::uint8_t>(value >> shift));
}
void BinaryWriter::text(std::string_view value) {
    u64(value.size());
    raw(reinterpret_cast<const std::byte*>(value.data()), value.size());
}
void BinaryWriter::raw(const std::byte* data, std::size_t size) {
    if (size != 0) bytes_.insert(bytes_.end(), data, data + size);
}

BinaryReader::BinaryReader(const std::vector<std::byte>& bytes, std::string errorPrefix)
    : bytes_(bytes), prefix_(std::move(errorPrefix)) {}

void BinaryReader::need(std::uint64_t count) {
    if (count > bytes_.size() || offset_ > bytes_.size() - static_cast<std::size_t>(count))
        throw std::runtime_error(prefix_ + "-TRUNCATED: checked read exceeds file size");
}
std::uint8_t BinaryReader::u8() {
    need(1);
    return std::to_integer<std::uint8_t>(bytes_[offset_++]);
}
std::uint32_t BinaryReader::u32() {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8)
        value |= static_cast<std::uint32_t>(u8()) << shift;
    return value;
}
std::uint64_t BinaryReader::u64() {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8)
        value |= static_cast<std::uint64_t>(u8()) << shift;
    return value;
}
std::string BinaryReader::text(std::uint64_t limit) {
    const auto count = u64();
    if (count > limit) throw std::runtime_error(prefix_ + "-LENGTH: string exceeds V0 limit");
    need(count);
    std::string value(reinterpret_cast<const char*>(bytes_.data() + offset_),
                      static_cast<std::size_t>(count));
    offset_ += static_cast<std::size_t>(count);
    return value;
}
std::vector<std::byte> BinaryReader::raw(std::uint64_t count, std::uint64_t limit) {
    if (count > limit) throw std::runtime_error(prefix_ + "-LENGTH: byte payload exceeds V0 limit");
    need(count);
    auto first = bytes_.begin() + static_cast<std::ptrdiff_t>(offset_);
    std::vector<std::byte> value(first, first + static_cast<std::ptrdiff_t>(count));
    offset_ += static_cast<std::size_t>(count);
    return value;
}

std::uint8_t dtypeCode(storage::DType dtype) {
    if (dtype == storage::DType::I64) return 1;
    if (dtype == storage::DType::F32) return 2;
    return 0;
}
storage::DType decodeDtype(std::uint8_t code) {
    if (code == 1) return storage::DType::I64;
    if (code == 2) return storage::DType::F32;
    return storage::DType::Invalid;
}

std::optional<std::string> validateStoredValue(const StoredValue& value) {
    if (value.dtype != storage::DType::I64 && value.dtype != storage::DType::F32)
        return "unsupported stored dtype";
    if (value.kind == StoredValueKind::Scalar) {
        if (!value.shape.empty()) return "scalar stored value has a shape";
        if (value.bytes.size() != storage::elementWidth(value.dtype))
            return "scalar stored value has an invalid byte length";
        return {};
    }
    if (value.kind != StoredValueKind::Tensor) return "unknown stored value kind";
    if (value.shape.empty() || value.shape.size() > 2) return "stored tensor rank is outside V0";
    try {
        const auto count = storage::checkedElementCount(value.shape);
        const auto expected = storage::checkedByteCount(count, value.dtype);
        if (value.bytes.size() != expected) return "stored tensor payload length disagrees with shape";
    } catch (const std::exception&) {
        return "stored tensor size arithmetic failed";
    }
    return {};
}

void writeStoredValue(BinaryWriter& writer, const StoredValue& value) {
    writer.u8(static_cast<std::uint8_t>(value.kind));
    writer.u8(dtypeCode(value.dtype));
    writer.u32(static_cast<std::uint32_t>(value.shape.size()));
    for (auto extent : value.shape) writer.u64(extent);
    writer.u64(value.bytes.size());
    writer.raw(value.bytes.data(), value.bytes.size());
}

StoredValue readStoredValue(BinaryReader& reader) {
    StoredValue value;
    value.kind = static_cast<StoredValueKind>(reader.u8());
    value.dtype = decodeDtype(reader.u8());
    const auto rank = reader.u32();
    if (rank > 64) throw std::runtime_error("PERSISTENCE-RANK: stored rank exceeds V0 limit");
    value.shape.reserve(rank);
    for (std::uint32_t axis = 0; axis < rank; ++axis) value.shape.push_back(reader.u64());
    value.bytes = reader.raw(reader.u64(), maxValueBytes);
    if (const auto invalid = validateStoredValue(value))
        throw std::runtime_error("PERSISTENCE-VALUE: " + *invalid);
    return value;
}

void writeStoredParameter(BinaryWriter& writer, const StoredParameter& parameter) {
    writer.u64(parameter.id);
    writer.text(parameter.name);
    writeStoredValue(writer, parameter.value);
}

StoredParameter readStoredParameter(BinaryReader& reader) {
    StoredParameter parameter;
    parameter.id = reader.u64();
    parameter.name = reader.text();
    parameter.value = readStoredValue(reader);
    return parameter;
}

std::uint64_t fnv1a64(const std::byte* bytes, std::size_t size) noexcept {
    std::uint64_t value = 1469598103934665603ULL;
    for (std::size_t index = 0; index < size; ++index) {
        value ^= std::to_integer<std::uint8_t>(bytes[index]);
        value *= 1099511628211ULL;
    }
    return value;
}

std::string formatDigest(std::uint64_t value) {
    std::ostringstream out;
    out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

std::string digestBytes(const std::byte* bytes, std::size_t size) {
    return formatDigest(fnv1a64(bytes, size));
}

std::string parameterDigest(const std::vector<StoredParameter>& parameters) {
    BinaryWriter writer;
    writer.u32(static_cast<std::uint32_t>(parameters.size()));
    for (const auto& parameter : parameters) writeStoredParameter(writer, parameter);
    return digestBytes(writer.bytes().data(), writer.bytes().size());
}

std::vector<std::byte> readFile(const std::filesystem::path& path, std::uint64_t limit) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open file");
    const auto end = input.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > limit)
        throw std::runtime_error("file size exceeds V0 limit");
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!bytes.empty()) input.read(reinterpret_cast<char*>(bytes.data()), end);
    if (!input) throw std::runtime_error("short file read");
    return bytes;
}

std::optional<std::string> transactionalWrite(const std::filesystem::path& path,
                                              const std::vector<std::byte>& bytes) {
    const auto absolute = std::filesystem::absolute(path);
    const auto parent = absolute.parent_path();
    if (!std::filesystem::is_directory(parent)) return "output parent does not exist";
    auto temporary = absolute;
    temporary += ".tmp";
    int descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0) return "cannot create exclusive temporary file: " + std::string(std::strerror(errno));
    bool renamed = false;
    auto fail = [&](const std::string& message) -> std::optional<std::string> {
        const int saved = errno;
        if (descriptor >= 0) (void)::close(descriptor);
        if (!renamed) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
        }
        return message + ": " + std::strerror(saved);
    };
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (written < 0) {
            if (errno == EINTR) continue;
            return fail("temporary file write failed");
        }
        if (written == 0) { errno = EIO; return fail("temporary file write made no progress"); }
        offset += static_cast<std::size_t>(written);
    }
    if (::fsync(descriptor) != 0) return fail("temporary file fsync failed");
    if (::close(descriptor) != 0) { descriptor = -1; return fail("temporary file close failed"); }
    descriptor = -1;
    if (::rename(temporary.c_str(), absolute.c_str()) != 0) return fail("atomic rename failed");
    renamed = true;
    const int directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory < 0) return "file installed but parent directory could not be opened for fsync";
    const bool synced = ::fsync(directory) == 0;
    const int syncError = errno;
    (void)::close(directory);
    if (!synced) return "file installed but parent directory fsync failed: " + std::string(std::strerror(syncError));
    return {};
}

} // namespace thiran::v0::persistence
