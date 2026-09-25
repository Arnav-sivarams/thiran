#pragma once

#include "persistence/v0/Persistence.hpp"
#include "training/v0/Training.hpp"

#include <filesystem>
#include <optional>

namespace thiran::v0::training {

inline constexpr std::uint32_t checkpointFormatVersion = 0;
inline constexpr std::uint32_t trainingCheckpointAbiVersion = 1;

struct CheckpointError {
    std::string code;
    std::string message;
};

struct CheckpointMetadata {
    std::string modelIdentity;
    std::string parameterDigest;
    std::uint64_t step = 0;
    OptimizerSpec optimizer;
};

struct CheckpointParameterSchema {
    ParameterId id;
    semantic::Type type;
    std::vector<std::int64_t> shape;
};

struct TrainingCheckpointSchema {
    std::string modelIdentity;
    std::uint64_t planSignature = 0;
    std::vector<CheckpointParameterSchema> parameters;
};

struct CheckpointSchemaResult {
    std::optional<TrainingCheckpointSchema> schema;
    std::optional<CheckpointError> error;
    bool ok() const noexcept { return schema.has_value() && !error.has_value(); }
};

struct CheckpointSaveResult {
    std::optional<CheckpointMetadata> metadata;
    std::optional<CheckpointError> error;
    bool ok() const noexcept { return metadata.has_value() && !error.has_value(); }
};

struct CheckpointLoadResult {
    std::optional<TrainingState> state;
    std::optional<CheckpointMetadata> metadata;
    std::optional<CheckpointError> error;
    bool ok() const noexcept { return state.has_value() && metadata.has_value() && !error.has_value(); }
};

std::vector<persistence::StoredParameter> storedParameters(const TrainingPlan&,
                                                           const TrainingState&);
CheckpointSchemaResult createTrainingCheckpointSchema(const std::string& modelIdentity,
                                                       const TrainingPlan&,
                                                       const TrainingState& initialState);
CheckpointSaveResult saveTrainingCheckpoint(const std::filesystem::path&,
                                             const TrainingCheckpointSchema&,
                                             const TrainingPlan&,
                                             const TrainingState&);
CheckpointLoadResult loadTrainingCheckpoint(const std::filesystem::path&,
                                             const TrainingCheckpointSchema&,
                                             const TrainingPlan&);

} // namespace thiran::v0::training
