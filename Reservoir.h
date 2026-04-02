#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/// Reservoir configuration. Defaults are scale-invariant — SR=0.90 and
/// input_scaling=0.02 are optimal across all DIMs (see ScaleInvariance.md).
/// Construct with defaults, override what you need.
struct ReservoirConfig
{
    uint64_t              seed             = 0;
    float                 alpha            = 1.0f;
    float                 spectral_radius  = 0.9f;     // scale-invariant optimum (see ScaleInvariance.md)
    float                 leak_rate        = 1.0f;     // 1.0 = full replacement, <1.0 = leaky integrator
    float                 input_scaling    = 0.02f;    // scale-invariant optimum (see ScaleInvariance.md)
    size_t                num_inputs       = 1;
    float                 output_fraction  = 1.0f;     // fraction of N vertices used as readout features (0.0, 1.0]
};

/// @brief Echo-state reservoir whose neurons live on a Boolean hypercube.
///
/// A Boolean hypercube of dimension DIM is a graph with N = 2^DIM vertices,
/// where each vertex is addressed by a DIM-bit binary index. The natural
/// edges of the hypercube connect vertices that differ by one bit, but this
/// reservoir's connectivity extends beyond the graph's natural edges to
/// include multi-bit shell connections (see below).
///
/// This class places one neuron at every vertex. At each timestep, every
/// neuron computes a weighted sum of its neighbors' previous outputs,
/// applies tanh(alpha * sum), and writes the result to its state slot.
/// The full N-dimensional state vector is then available to a downstream
/// readout (see ESN for the complete pipeline).
///
/// **Connectivity.** Each neuron receives from (2*DIM - 2) neighbors,
/// organized into two families:
///
///   - **Shell connections** (cumulative-bit masks 3, 7, 15, ...):
///     DIM-2 connections using masks ShellMask(1) through ShellMask(DIM-2).
///     The distance-1 shell (mask 1) is omitted (covered by nearest
///     neighbors) and the antipodal shell (mask 2^DIM - 1, all bits set)
///     is omitted to limit maximum reach.
///
///   - **Nearest-neighbor connections** (single-bit flips 1, 2, 4, 8, ...):
///     DIM connections, each vertex to its Hamming-distance-1 neighbors,
///     providing local coupling along every dimension of the hypercube.
///
/// Every connection has its own fixed random weight, giving N * (2*DIM - 2)
/// total recurrent weights. Neighbor addresses are computed inline from
/// the loop index (v XOR mask) — no adjacency storage is needed.
///
/// **Input injection.** External input is projected onto neuron states via
/// per-vertex random weights (W_in), drawn from U(-input_scaling, +input_scaling).
/// Inputs are clamped to [-1, +1]. In multi-input mode (K channels), input k
/// is mapped to every K-th vertex starting at offset k (v where v % K == k),
/// distributing each channel evenly across the hypercube.
///
/// **Spectral radius.** After random initialization, recurrent weights are
/// rescaled so the spectral norm (estimated via power iteration) matches
/// the target spectral radius specified in ReservoirConfig. The optimal
/// SR (0.90) is scale-invariant across all DIMs — a property of the
/// hypercube's vertex-transitive topology (see ScaleInvariance.md).
///
/// **Usage.** Construct via the static Create() factory, then alternate
/// InjectInput() and Step() calls. Read the N-dimensional state from
/// Outputs() after each Step().
template <size_t DIM>
class Reservoir
{
    static_assert(DIM >= 5 && DIM <= 12, "DIM must be in 5 <= DIM <= 12");

    static constexpr size_t N = 1ULL << DIM;
    static constexpr size_t NUM_SHELL = DIM - 2;          // shells 3, 7, ..., 2^(DIM-1)-1
    static constexpr size_t NUM_CONNECTIONS = NUM_SHELL + DIM;

public:
    static constexpr size_t dim = DIM;

    /// Inline neighbor mask computation — no stored adjacency.
    /// Shells [1..DIM-1):  mask = (1 << (i+1)) - 1      → 3, 7, 15, ...  (skip distance-1 and antipodal)
    /// Nearest [0..DIM):   mask = 1 << i                 → 1, 2, 4, 8, ...
    static constexpr uint32_t ShellMask(size_t i) { return (1u << (i + 1)) - 1; }
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    /// @brief Create a reservoir from a fully resolved config.
    static std::unique_ptr<Reservoir> Create(const ReservoirConfig& cfg)
    {
        return std::unique_ptr<Reservoir>(new Reservoir(cfg));
    }

    Reservoir(const Reservoir&) = delete;
    Reservoir& operator=(const Reservoir&) = delete;

    void Step();

    /// @brief Inject a scalar input into one channel before Step().
    /// @param channel  Channel index in [0, num_inputs). Channel k owns vertices k, k+K, k+2K, ...
    /// @param input    Scalar value. Clamped to [-1, 1].
    void InjectInput(size_t channel, float input);

    [[nodiscard]] const float* Outputs() const { return vtx_output_; }
    [[nodiscard]] float GetAlpha() const { return alpha_; }
    [[nodiscard]] uint64_t GetSeed() const { return rng_seed_; }
    [[nodiscard]] float GetSpectralRadius() const { return spectral_radius_; }
    [[nodiscard]] float GetLeakRate() const { return leak_rate_; }
    [[nodiscard]] float GetInputScaling() const { return input_scaling_; }

private:
    explicit Reservoir(const ReservoirConfig& cfg);
    uint64_t rng_seed_;

    alignas(64) float vtx_state_[N]{};
    alignas(64) float vtx_output_[N]{};
    std::vector<float> vtx_input_weight_; // flat [N] — one W_in weight per vertex
    std::vector<float> vtx_weight_; // flat [N * NUM_CONNECTIONS]

    size_t num_inputs_ = 1;
    float alpha_ = 1.0f;
    float spectral_radius_ = 0.9f;
    float leak_rate_ = 1.0f;
    float input_scaling_ = 0.02f;

    void Initialize();
    void UpdateState(size_t v);
    [[nodiscard]] float EstimateSpectralRadius() const;
};
