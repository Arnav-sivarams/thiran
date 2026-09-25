#pragma once

#include "storage/v0/Storage.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace thiran::v0::persistence {

inline constexpr std::uint64_t maxStringBytes = 1U << 20;
inline constexpr std::uint64_t maxItems = 1U << 20;
inline constexpr std::uint64_t maxValueBytes = 64U << 20;

enum class StoredValueKind : std::uint8_t { Scalar = 1, Tensor = 2 };

struct StoredValue {
    StoredValueKind kind = StoredValueKind::Scalar;
    storage::DType dtype = storage::DType::Invalid;
    std::vector<std::uint64_t> shape;
    std::vector<std::byte> bytes;
    bool operator==(const StoredValue&) const = default;
};

struct StoredParameter {
    std::uint64_t id = 0;
    std::string name;
    StoredValue value;
    bool operator==(const StoredParameter&) const = default;
};

class BinaryWriter {
public:
    void u8(std::uint8_t);
    void u32(std::uint32_t);
    void u64(std::uint64_t);
    void text(std::string_view);
    void raw(const std::byte*, std::size_t);
    const std::vector<std::byte>& bytes() const noexcept { return bytes_; }
    std::vector<std::byte> take() { return std::move(bytes_); }
private:
    std::vector<std::byte> bytes_;
};

class BinaryReader {
public:
    BinaryReader(const std::vector<std::byte>&, std::string errorPrefix);
    std::uint8_t u8();
    std::uint32_t u32();
    std::uint64_t u64();
    std::string text(std::uint64_t limit = maxStringBytes);
    std::vector<std::byte> raw(std::uint64_t count, std::uint64_t limit);
    bool done() const noexcept { return offset_ == bytes_.size(); }
    std::size_t offset() const noexcept { return offset_; }
private:
    void need(std::uint64_t);
    const std::vector<std::byte>& bytes_;
    std::string prefix_;
    std::size_t offset_ = 0;
};

std::uint8_t dtypeCode(storage::DType);
storage::DType decodeDtype(std::uint8_t);
std::optional<std::string> validateStoredValue(const StoredValue&);
void writeStoredValue(BinaryWriter&, const StoredValue&);
StoredValue readStoredValue(BinaryReader&);
void writeStoredParameter(BinaryWriter&, const StoredParameter&);
StoredParameter readStoredParameter(BinaryReader&);

std::uint64_t fnv1a64(const std::byte*, std::size_t) noexcept;
std::string formatDigest(std::uint64_t);
std::string digestBytes(const std::byte*, std::size_t);
std::string parameterDigest(const std::vector<StoredParameter>&);

std::vector<std::byte> readFile(const std::filesystem::path&, std::uint64_t limit);
// Uses a same-directory exclusive temporary file, fsync, close, and atomic
// rename. An existing temporary file is never overwritten or removed.
std::optional<std::string> transactionalWrite(const std::filesystem::path&,
                                              const std::vector<std::byte>&);

} // namespace thiran::v0::persistence
