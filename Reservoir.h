#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/// Concrete reservoir configuration. No sentinels, no defaults.
/// Callers are responsible for populating all fields.
struct ReservoirConfig
{
    uint64_t              seed             = 0;
    float                 alpha            = 1.0f;
    float                 spectral_radius  = 0.9f;
    float                 leak_rate        = 1.0f;     // 1.0 = full replacement, <1.0 = leaky integrator
    std::vector<float>    block_scaling    = {0.05f};   // size = num_inputs
};

/// @brief Echo-state reservoir whose neurons live on a Boolean hypercube.
///
/// A Boolean hypercube of dimension DIM is a graph with N = 2^DIM vertices,
/// where each vertex is addressed by a DIM-bit binary index. Two vertices
/// are neighbors when their indices differ by exactly one bit — the edge
/// set is defined entirely by XOR, so no adjacency list is stored.
///
/// This class places one neuron at every vertex. At each timestep, every
/// neuron computes a weighted sum of its neighbors' previous outputs,
/// applies tanh(alpha * sum), and writes the result to its state slot.
/// The full N-dimensional state vector is then available to a downstream
/// readout (see ESN for the complete pipeline).
///
/// **Connectivity.** Each neuron receives from 2*DIM neighbors, organized
/// into two families of DIM connections each:
///
///   - **Shell connections** (cumulative-bit masks 1, 3, 7, 15, ...):
///     connect vertices whose low-order bits are progressively scrambled,
///     mixing information across multiple bit positions at once.
///
///   - **Nearest-neighbor connections** (single-bit flips 1, 2, 4, 8, ...):
///     connect each vertex to its DIM Hamming-distance-1 neighbors,
///     providing local coupling along every dimension of the hypercube.
///
/// Every connection has its own learned weight, giving N * 2*DIM total
/// recurrent weights. Neighbor addresses are computed inline from the
/// loop index (v XOR mask) — no adjacency storage is needed.
///
/// **Input injection.** External input is projected onto neuron states via
/// per-vertex random weights (W_in). Inputs are clamped to [-1, +1].
/// In multi-input mode (K channels), the N vertices are block-partitioned
/// into K contiguous groups, each driven by one input channel.
///
/// **Spectral radius.** After random initialization, recurrent weights are
/// rescaled so the spectral norm (estimated via power iteration) matches
/// the target spectral radius specified in ReservoirConfig.
///
/// **Usage.** Construct via the static Create() factory, then alternate
/// InjectInput() and Step() calls. Read the N-dimensional state from
/// Outputs() after each Step().
template <size_t DIM>
class Reservoir
{
    static_assert(DIM >= 4 && DIM <= 10, "DIM must be in 4 <= DIM <= 10");

    static constexpr size_t N = 1ULL << DIM;
    static constexpr size_t NUM_CONNECTIONS = 2 * DIM;

public:
    static constexpr size_t dim = DIM;

    /// Inline neighbor mask computation — no stored adjacency.
    /// Shells [0..DIM):    mask = (1 << (i+1)) - 1      → 1, 3, 7, 15, ...
    /// Nearest [0..DIM):   mask = 1 << i                 → 1, 2, 4, 8, ...
    static constexpr uint32_t ShellMask(size_t i) { return (1u << (i + 1)) - 1; }
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    /// @brief Create a reservoir from a fully resolved config.
    static std::unique_ptr<Reservoir> Create(const ReservoirConfig& cfg)
    {
        return std::unique_ptr<Reservoir>(
            new Reservoir(cfg.seed, cfg.alpha, cfg.spectral_radius,
                          cfg.leak_rate, cfg.block_scaling));
    }

    Reservoir(const Reservoir&) = delete;
    Reservoir& operator=(const Reservoir&) = delete;

    void Step();

    /// @brief Inject a scalar input into one block before Step().
    /// @param block  Block index in [0, num_inputs). Each block owns vertices [k*N/K, (k+1)*N/K).
    /// @param input  Scalar value. Clamped to [-1, 1].
    void InjectInput(size_t block, float input);

    [[nodiscard]] const float* Outputs() const { return vtx_output_; }
    [[nodiscard]] float GetAlpha() const { return alpha_; }

private:
    explicit Reservoir(uint64_t rng_seed, float alpha,
                       float spectral_radius, float leak_rate,
                       std::vector<float> block_scaling);
    uint64_t rng_seed_;

    alignas(64) float vtx_state_[N]{};
    alignas(64) float vtx_output_[N]{};
    std::vector<float> vtx_input_weight_; // flat [N] — one weight per vertex (block-partitioned)
    std::vector<float> vtx_weight_; // flat [N * NUM_CONNECTIONS]

    size_t num_inputs_ = 1;
    float alpha_ = 1.0f;
    float spectral_radius_ = 0.9f;
    float leak_rate_ = 1.0f;
    std::vector<float> block_scaling_; // per-block W_in scaling [num_inputs]

    void Initialize();
    void UpdateState(size_t v);
    [[nodiscard]] float EstimateSpectralRadius() const;
};
