#include "Serialization.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

namespace hrccnn_llm_math {

// Serialization POD-dumps ReservoirConfig and CNNReadoutConfig directly.
// Adding any non-trivially-copyable member (std::string, std::vector, etc.)
// to either struct would silently break save/load — catch it at compile time.
static_assert(std::is_trivially_copyable_v<ReservoirConfig>,
              "ReservoirConfig must stay trivially copyable for POD serialization");
static_assert(std::is_trivially_copyable_v<CNNReadoutConfig>,
              "CNNReadoutConfig must stay trivially copyable for POD serialization");

namespace {

// Upper bound for any serialized vector payload. 1 GiB is orders of magnitude
// beyond any readout state this library produces; trips only on a corrupt or
// hostile file.
constexpr std::uint64_t kMaxVecBytes = 1ULL << 30;

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
    std::uint64_t n = v.size();
    WritePOD(os, n);
    if (n > 0) os.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
}

template <typename T>
bool ReadVec(std::istream& is, std::vector<T>& v)
{
    std::uint64_t n = 0;
    if (!ReadPOD(is, n)) return false;
    if (n > kMaxVecBytes / sizeof(T)) return false;
    v.resize(static_cast<std::size_t>(n));
    if (n > 0) {
        is.read(reinterpret_cast<char*>(v.data()),
                static_cast<std::streamsize>(n * sizeof(T)));
    }
    return static_cast<bool>(is);
}

}  // namespace

bool SaveModelFile(const std::string& path, const ModelFile& mf)
{
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;

    os.write(kMagic, 8);
    WritePOD(os, kFormatVersion);
    WritePOD(os, mf.dim);

    WritePOD(os, mf.meta.training_seed);
    WritePOD(os, mf.meta.training_samples);
    WritePOD(os, mf.meta.training_epochs);

    char sha[40] = {};
    std::size_t n = std::min(mf.meta.git_sha.size(), std::size_t{40});
    std::memcpy(sha, mf.meta.git_sha.data(), n);
    os.write(sha, 40);

    WritePOD(os, mf.reservoir_cfg);
    WritePOD(os, mf.cnn_cfg);

    WriteVec(os, mf.readout.weights);
    WritePOD(os, mf.readout.bias);
    WriteVec(os, mf.readout.feature_mean);
    WriteVec(os, mf.readout.feature_scale);

    return static_cast<bool>(os);
}

bool LoadModelFile(const std::string& path, ModelFile& mf, std::string* err)
{
    auto fail = [&](const char* msg) {
        if (err) *err = msg;
        return false;
    };

    std::ifstream is(path, std::ios::binary);
    if (!is) return fail("could not open file");

    char magic[8] = {};
    is.read(magic, 8);
    if (!is || std::memcmp(magic, kMagic, 8) != 0) return fail("bad magic");

    std::uint32_t version = 0;
    if (!ReadPOD(is, version)) return fail("short read (version)");
    if (version != kFormatVersion) return fail("unsupported format_version");

    if (!ReadPOD(is, mf.dim)) return fail("short read (dim)");

    if (!ReadPOD(is, mf.meta.training_seed))    return fail("short read (seed)");
    if (!ReadPOD(is, mf.meta.training_samples)) return fail("short read (samples)");
    if (!ReadPOD(is, mf.meta.training_epochs))  return fail("short read (epochs)");

    char sha[40] = {};
    is.read(sha, 40);
    if (!is) return fail("short read (git_sha)");
    mf.meta.git_sha.assign(sha, std::find(sha, sha + 40, '\0'));

    if (!ReadPOD(is, mf.reservoir_cfg)) return fail("short read (reservoir_cfg)");
    if (!ReadPOD(is, mf.cnn_cfg))       return fail("short read (cnn_cfg)");

    if (!ReadVec(is, mf.readout.weights))      return fail("short read or oversized (weights)");
    if (!ReadPOD(is, mf.readout.bias))         return fail("short read (bias)");
    if (!ReadVec(is, mf.readout.feature_mean)) return fail("short read or oversized (feature_mean)");
    if (!ReadVec(is, mf.readout.feature_scale))return fail("short read or oversized (feature_scale)");

    return true;
}

template <std::size_t DIM>
SerialReadoutState ToSerial(const typename ESN<DIM>::ReadoutState& s)
{
    SerialReadoutState r;
    r.weights       = s.weights;
    r.bias          = s.bias;
    r.feature_mean  = s.feature_mean;
    r.feature_scale = s.feature_scale;
    return r;
}

template <std::size_t DIM>
typename ESN<DIM>::ReadoutState FromSerial(const SerialReadoutState& s)
{
    typename ESN<DIM>::ReadoutState r;
    r.weights       = s.weights;
    r.bias          = s.bias;
    r.feature_mean  = s.feature_mean;
    r.feature_scale = s.feature_scale;
    r.is_trained    = !s.weights.empty();
    return r;
}

template SerialReadoutState ToSerial<12>(const ESN<12>::ReadoutState&);
template ESN<12>::ReadoutState FromSerial<12>(const SerialReadoutState&);

}  // namespace hrccnn_llm_math
