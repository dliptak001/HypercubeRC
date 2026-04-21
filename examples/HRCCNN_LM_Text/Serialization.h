#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ESN.h"

namespace hrccnn_lm_text {

/// Training metadata captured at save time — informational; not required
/// to reconstruct the model but useful for auditing a checkpoint.
struct TrainingMetadata
{
    std::uint64_t training_seed    = 0;
    std::uint32_t training_positions = 0;
    std::uint32_t training_passes  = 0;
    std::string   git_sha;           ///< up to 40 chars; truncated on save
};

/// Pure-data mirror of ESN<DIM>::ReadoutState (template-free so it can
/// cross compilation units without instantiating ESN<DIM>).
struct SerialReadoutState
{
    std::vector<double> weights;
    double              bias = 0.0;
    std::vector<float>  feature_mean;
    std::vector<float>  feature_scale;
    std::vector<double> target_mean;  ///< Per-output target centering (regression).
};

/// Full serialized model payload.
///   - `dim` is the hypercube dimension the model was trained at.
///   - `vocab` is the char-level vocabulary; position in the string is the
///     class index.  Embedded in the file so eval/infer don't need to
///     re-derive it from the corpus (which would desync if the corpus
///     ever changes).
struct ModelFile
{
    std::uint32_t        dim = 0;
    std::string          vocab;
    TrainingMetadata     meta;
    ReservoirConfig      reservoir_cfg{};
    HCNNReadoutConfig     cnn_cfg{};
    SerialReadoutState   readout;
};

inline constexpr char          kMagic[8]       = {'H','C','N','N','L','M','T','X'};
inline constexpr std::uint32_t kFormatVersion  = 2;

/// Write a ModelFile to disk.  Returns true on success.
bool SaveModelFile(const std::string& path, const ModelFile& mf);

/// Load a ModelFile from disk.  Returns true on success.  Populates
/// `err` with a short diagnostic on failure.
bool LoadModelFile(const std::string& path, ModelFile& mf, std::string* err = nullptr);

/// Convert between ESN<DIM>::ReadoutState and SerialReadoutState.
template <std::size_t DIM>
SerialReadoutState ToSerial(const typename ESN<DIM>::ReadoutState& s);

template <std::size_t DIM>
typename ESN<DIM>::ReadoutState FromSerial(const SerialReadoutState& s);

}  // namespace hrccnn_lm_text
