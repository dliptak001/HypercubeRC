#include "Model.h"
#include "Config.h"
#include "Presets.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace lm {

// =========================================================================
//  Binary serialization helpers (same format as LM_Text for compatibility)
// =========================================================================

namespace {

inline constexpr char     kMagic[8]       = {'H','C','N','N','L','M','T','X'};
inline constexpr uint32_t kFormatVersion  = 4;
inline constexpr uint64_t kMaxVecBytes    = 1ULL << 30;

template <typename T>
void WritePOD(std::ostream& os, const T& v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
bool ReadPOD(std::istream& is, T& v)
{
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return static_cast<bool>(is);
}

template <typename T>
void WriteVec(std::ostream& os, const std::vector<T>& v)
{
    uint64_t n = v.size();
    WritePOD(os, n);
    if (n > 0)
        os.write(reinterpret_cast<const char*>(v.data()),
                 static_cast<std::streamsize>(n * sizeof(T)));
}

template <typename T>
bool ReadVec(std::istream& is, std::vector<T>& v)
{
    uint64_t n = 0;
    if (!ReadPOD(is, n)) return false;
    if (n > kMaxVecBytes / sizeof(T)) return false;
    v.resize(static_cast<std::size_t>(n));
    if (n > 0)
        is.read(reinterpret_cast<char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(is);
}

void WriteStr(std::ostream& os, const std::string& s)
{
    uint32_t n = static_cast<uint32_t>(s.size());
    WritePOD(os, n);
    if (n > 0) os.write(s.data(), n);
}

bool ReadStr(std::istream& is, std::string& s)
{
    uint32_t n = 0;
    if (!ReadPOD(is, n)) return false;
    if (n > (1u << 20)) return false;
    s.resize(n);
    if (n > 0 && !is.read(s.data(), n)) return false;
    return static_cast<bool>(is);
}

// --- Field-by-field struct serialization (portable across compilers) ---

void WriteReservoirConfig(std::ostream& os, const ReservoirConfig& c)
{
    WritePOD(os, c.seed);
    WritePOD(os, c.alpha);
    WritePOD(os, c.spectral_radius);
    WritePOD(os, c.leak_rate);
    WritePOD(os, c.input_scaling);
    WritePOD(os, c.coupling_scaling);
    WritePOD(os, static_cast<int32_t>(c.coupling_mode));
    WritePOD(os, static_cast<uint64_t>(c.num_inputs));
    WritePOD(os, c.output_fraction);
}

bool ReadReservoirConfig(std::istream& is, ReservoirConfig& c)
{
    int32_t coupling_mode; uint64_t num_inputs;
    if (!ReadPOD(is, c.seed))             return false;
    if (!ReadPOD(is, c.alpha))            return false;
    if (!ReadPOD(is, c.spectral_radius))  return false;
    if (!ReadPOD(is, c.leak_rate))        return false;
    if (!ReadPOD(is, c.input_scaling))    return false;
    if (!ReadPOD(is, c.coupling_scaling)) return false;
    if (!ReadPOD(is, coupling_mode))      return false;
    if (!ReadPOD(is, num_inputs))         return false;
    if (!ReadPOD(is, c.output_fraction))  return false;
    c.coupling_mode = static_cast<CouplingMode>(coupling_mode);
    c.num_inputs    = static_cast<size_t>(num_inputs);
    return true;
}

void WriteReadoutArchConfig(std::ostream& os, const ReadoutArchConfig& c)
{
    WritePOD(os, static_cast<int32_t>(c.num_outputs));
    WritePOD(os, static_cast<int32_t>(c.task));
    WritePOD(os, static_cast<int32_t>(c.num_layers));
    WritePOD(os, static_cast<int32_t>(c.conv_channels));
    WritePOD(os, static_cast<int32_t>(c.input_channels));
    WritePOD(os, static_cast<uint32_t>(c.seed));
}

bool ReadReadoutArchConfig(std::istream& is, ReadoutArchConfig& c)
{
    int32_t num_outputs, task, num_layers, conv_channels, input_channels;
    uint32_t seed;
    if (!ReadPOD(is, num_outputs))    return false;
    if (!ReadPOD(is, task))           return false;
    if (!ReadPOD(is, num_layers))     return false;
    if (!ReadPOD(is, conv_channels))  return false;
    if (!ReadPOD(is, input_channels)) return false;
    if (!ReadPOD(is, seed))           return false;
    c.num_outputs    = num_outputs;
    c.task           = static_cast<ReadoutTask>(task);
    c.num_layers     = num_layers;
    c.conv_channels  = conv_channels;
    c.input_channels = input_channels;
    c.seed           = seed;
    return true;
}

void WriteReadoutTrainConfig(std::ostream& os, const ReadoutTrainConfig& c)
{
    WritePOD(os, static_cast<int32_t>(c.epochs));
    WritePOD(os, static_cast<int32_t>(c.batch_size));
    WritePOD(os, c.lr_max);
    WritePOD(os, c.lr_min_frac);
    WritePOD(os, static_cast<int32_t>(c.lr_decay_epochs));
    WritePOD(os, c.weight_decay);
    WritePOD(os, static_cast<uint8_t>(c.verbose));
    WritePOD(os, static_cast<uint8_t>(c.verbose_train_acc));
}

bool ReadReadoutTrainConfig(std::istream& is, ReadoutTrainConfig& c)
{
    int32_t epochs, batch_size, lr_decay_epochs;
    uint8_t verbose, verbose_train_acc;
    if (!ReadPOD(is, epochs))            return false;
    if (!ReadPOD(is, batch_size))        return false;
    if (!ReadPOD(is, c.lr_max))          return false;
    if (!ReadPOD(is, c.lr_min_frac))     return false;
    if (!ReadPOD(is, lr_decay_epochs))   return false;
    if (!ReadPOD(is, c.weight_decay))    return false;
    if (!ReadPOD(is, verbose))           return false;
    if (!ReadPOD(is, verbose_train_acc)) return false;
    c.epochs           = epochs;
    c.batch_size       = batch_size;
    c.lr_decay_epochs  = lr_decay_epochs;
    c.verbose          = verbose;
    c.verbose_train_acc = verbose_train_acc;
    return true;
}

}  // namespace

// =========================================================================
//  Construction
// =========================================================================

template <std::size_t DIM>
Model<DIM>::Model(const ReservoirConfig& reservoir_cfg,
                  const ReadoutArchConfig& cnn_arch,
                  std::size_t depth)
    : esn_(depth, reservoir_cfg),
      reservoir_cfg_(reservoir_cfg),
      cnn_arch_(cnn_arch),
      depth_(depth)
{
}

// =========================================================================
//  Reservoir driving
// =========================================================================

template <std::size_t DIM>
void Model<DIM>::Reset()
{
    esn_.ResetReservoirOnly();
}

template <std::size_t DIM>
void Model<DIM>::Step(char ch)
{
    Vocabulary::EncodeBipolar(ch, step_bits_);
    esn_.Warmup(step_bits_, 1);
}

template <std::size_t DIM>
void Model<DIM>::Warmup(const std::string& text)
{
    for (char ch : text)
        Step(ch);
}

// =========================================================================
//  Readout — online training
// =========================================================================

template <std::size_t DIM>
void Model<DIM>::InitReadout(const std::string& text,
                             const ReadoutArchConfig& arch)
{
    cnn_arch_ = arch;

    std::vector<float> bits(text.size() * Vocabulary::kInputBits);
    Vocabulary::EncodeBipolar(text, bits.data());
    esn_.InitOnline(bits.data(), text.size(), arch);
}

template <std::size_t DIM>
void Model<DIM>::CopyLiveState(float* out) const
{
    esn_.CopyLiveState(out);
}

template <std::size_t DIM>
void Model<DIM>::TrainBatch(const float* states, const int* targets,
                            std::size_t count, float lr)
{
    esn_.TrainLiveBatch(states, targets, count, lr);
}

// =========================================================================
//  Readout — prediction
// =========================================================================

template <std::size_t DIM>
void Model<DIM>::Predict(float* logits) const
{
    esn_.PredictLiveRaw(logits);
}

// =========================================================================
//  State snapshot
// =========================================================================

template <std::size_t DIM>
void Model<DIM>::SaveState(std::vector<float>& state,
                           std::vector<float>& output) const
{
    state.resize(OutputSize());
    output.resize(OutputSize());
    esn_.SaveReservoirState(state.data(), output.data());
}

template <std::size_t DIM>
void Model<DIM>::RestoreState(const std::vector<float>& state,
                              const std::vector<float>& output)
{
    esn_.RestoreReservoirState(state.data(), output.data());
}

// =========================================================================
//  Persistence — Save
// =========================================================================

template <std::size_t DIM>
void Model<DIM>::Save(const std::string& path,
                      const Vocabulary& vocab,
                      const TrainingMetadata& meta) const
{
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os)
        throw std::runtime_error("cannot open file for writing: " + path);

    os.write(kMagic, 8);
    WritePOD(os, kFormatVersion);
    WritePOD(os, static_cast<uint32_t>(DIM));
    WriteStr(os, vocab.Tokens());

    WritePOD(os, meta.training_seed);
    WritePOD(os, meta.training_positions);
    WritePOD(os, meta.training_passes);

    char sha[40] = {};
    std::size_t n = std::min(meta.git_sha.size(), std::size_t{40});
    std::memcpy(sha, meta.git_sha.data(), n);
    os.write(sha, 40);

    WritePOD(os, static_cast<uint32_t>(depth_));

    WriteReservoirConfig(os, esn_.GetConfig());
    WriteReadoutArchConfig(os, cnn_arch_);
    WriteReadoutTrainConfig(os, ReadoutTrainConfig{});

    auto readout_state = esn_.GetReadoutState();
    WriteVec(os, readout_state.weights);
    WritePOD(os, readout_state.bias);
    WriteVec(os, readout_state.feature_mean);
    WriteVec(os, readout_state.feature_scale);
    WriteVec(os, readout_state.target_mean);

    if (!os)
        throw std::runtime_error("write error saving model to: " + path);
}

// =========================================================================
//  Persistence — Load
// =========================================================================

template <std::size_t DIM>
Model<DIM> Model<DIM>::Load(const std::string& path, const Vocabulary& vocab)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("cannot open model file: " + path);

    char magic[8] = {};
    is.read(magic, 8);
    if (!is || std::memcmp(magic, kMagic, 8) != 0)
        throw std::runtime_error("bad magic in model file");

    uint32_t version = 0;
    if (!ReadPOD(is, version) || version != kFormatVersion)
        throw std::runtime_error("unsupported model format version");

    uint32_t file_dim = 0;
    if (!ReadPOD(is, file_dim))
        throw std::runtime_error("short read (dim)");
    if (file_dim != DIM)
        throw std::runtime_error("model DIM mismatch: file has "
                                 + std::to_string(file_dim)
                                 + ", binary compiled for "
                                 + std::to_string(DIM));

    std::string file_vocab;
    if (!ReadStr(is, file_vocab))
        throw std::runtime_error("short read (vocab)");

    if (file_vocab != vocab.Tokens())
        throw std::runtime_error("model vocab does not match expected vocabulary");

    // Skip training metadata (informational only).
    uint64_t seed; uint32_t positions, passes;
    ReadPOD(is, seed); ReadPOD(is, positions); ReadPOD(is, passes);
    char sha[40]; is.read(sha, 40);

    uint32_t file_depth = 0;
    if (!ReadPOD(is, file_depth))
        throw std::runtime_error("short read (depth)");

    ReservoirConfig rcfg;
    if (!ReadReservoirConfig(is, rcfg))
        throw std::runtime_error("short read (reservoir_cfg)");

    ReadoutArchConfig acfg;
    if (!ReadReadoutArchConfig(is, acfg))
        throw std::runtime_error("short read (cnn_arch)");

    ReadoutTrainConfig tcfg;
    if (!ReadReadoutTrainConfig(is, tcfg))
        throw std::runtime_error("short read (cnn_train)");

    // Build the model with loaded config.
    Model<DIM> model(rcfg, acfg, static_cast<std::size_t>(file_depth));
    model.esn_.SetCNNConfig(acfg);

    // Read readout state.
    typename ESN<DIM>::ReadoutState rs;
    if (!ReadVec(is, rs.weights))       throw std::runtime_error("short read (weights)");
    if (!ReadPOD(is, rs.bias))          throw std::runtime_error("short read (bias)");
    if (!ReadVec(is, rs.feature_mean))  throw std::runtime_error("short read (feature_mean)");
    if (!ReadVec(is, rs.feature_scale)) throw std::runtime_error("short read (feature_scale)");
    if (!ReadVec(is, rs.target_mean))   throw std::runtime_error("short read (target_mean)");
    rs.is_trained = !rs.weights.empty();

    model.esn_.SetReadoutState(rs);
    return model;
}

// =========================================================================
//  Explicit instantiation
// =========================================================================

using namespace lm::config;
template class Model<kDIM>;

}  // namespace lm
