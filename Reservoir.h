#pragma once

#include <cstddef>
#include <memory>
#include <vector>

/// Feature mode selector for per-DIM default resolution.
/// Raw: optimized for N-dim raw readout.
/// Translation: optimized for 2.5N-dim readout via TranslationLayer.
enum class FeatureMode { Raw, Translation };

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
/// a target spectral radius. Two sets of per-DIM optimized defaults are
/// provided — one for raw N-feature readout, one for 2.5N translation
/// readout — selected automatically by the FeatureMode parameter.
/// Explicit values override the defaults.
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

    // ----- Raw-feature defaults -----
    // Jointly optimized on MG h=1 + NARMA-10 + MC, raw N-dim readout.
    // Balanced across all tasks, not single-task optimal.

    static constexpr float RawSpectralRadius()
    {
        if constexpr (DIM == 4) return 0.95f;
        if constexpr (DIM == 5) return 0.80f;
        if constexpr (DIM == 6) return 0.90f;
        if constexpr (DIM == 7) return 0.88f;
        if constexpr (DIM == 8) return 0.88f;
        if constexpr (DIM == 9) return 0.88f;   // extrapolated from DIM 8, not sweep-verified
        if constexpr (DIM == 10) return 0.88f;  // extrapolated from DIM 8, not sweep-verified
        return 0.88f;
    }

    static constexpr float RawInputScaling()
    {
        if constexpr (DIM == 4) return 0.05f;
        if constexpr (DIM == 5) return 0.10f;
        if constexpr (DIM == 6) return 0.05f;
        if constexpr (DIM == 7) return 0.03f;
        if constexpr (DIM == 8) return 0.02f;
        if constexpr (DIM == 9) return 0.02f;   // extrapolated from DIM 8, not sweep-verified
        if constexpr (DIM == 10) return 0.02f;  // extrapolated from DIM 8, not sweep-verified
        return 0.02f;
    }

    // ----- Translation-layer defaults -----
    // Jointly optimized on MG h=1 + NARMA-10 + MC, translation 2.5N readout.
    // Balanced across all tasks. Very low input scaling (0.02-0.04) —
    // translation features amplify dynamics, so less drive is needed.

    static constexpr float TranslationSpectralRadius()
    {
        if constexpr (DIM == 4) return 0.88f;
        if constexpr (DIM == 5) return 0.80f;
        if constexpr (DIM == 6) return 0.92f;
        if constexpr (DIM == 7) return 0.92f;
        if constexpr (DIM == 8) return 0.95f;
        if constexpr (DIM == 9) return 0.95f;   // extrapolated from DIM 8, not sweep-verified
        if constexpr (DIM == 10) return 0.95f;  // extrapolated from DIM 8, not sweep-verified
        return 0.95f;
    }

    static constexpr float TranslationInputScaling()
    {
        if constexpr (DIM == 4) return 0.02f;
        if constexpr (DIM == 5) return 0.04f;
        if constexpr (DIM == 6) return 0.02f;
        if constexpr (DIM == 7) return 0.04f;
        if constexpr (DIM == 8) return 0.02f;
        if constexpr (DIM == 9) return 0.02f;   // extrapolated from DIM 8, not sweep-verified
        if constexpr (DIM == 10) return 0.02f;  // extrapolated from DIM 8, not sweep-verified
        return 0.02f;
    }

    /// Resolve the default spectral radius for a given feature mode.
    static constexpr float DefaultSpectralRadius(FeatureMode mode = FeatureMode::Raw)
    {
        return (mode == FeatureMode::Translation) ? TranslationSpectralRadius() : RawSpectralRadius();
    }

    /// Resolve the default input scaling for a given feature mode.
    static constexpr float DefaultInputScaling(FeatureMode mode = FeatureMode::Raw)
    {
        return (mode == FeatureMode::Translation) ? TranslationInputScaling() : RawInputScaling();
    }

    /// Inline neighbor mask computation — no stored adjacency.
    /// Shells [0..DIM):    mask = (1 << (i+1)) - 1      → 1, 3, 7, 15, ...
    /// Nearest [0..DIM):   mask = 1 << i                 → 1, 2, 4, 8, ...
    static constexpr uint32_t ShellMask(size_t i) { return (1u << (i + 1)) - 1; }
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    /// @brief Create a reservoir.
    /// @param rng_seed         Deterministic initialization seed.
    /// @param mode             Feature mode — selects per-DIM defaults (Raw or Translation).
    /// @param alpha            Tanh steepness (1.0 universally optimal).
    /// @param spectral_radius  Target SR (-1 = per-DIM default for the selected mode).
    /// @param block_scaling    Per-block W_in scaling, K values. null = mode default for all blocks.
    ///                         Pass K floats, one per block. -1.0f entries resolve to mode default.
    /// @param num_inputs       Number of input blocks (1 = single-input, 2+ = cascade/multi).
    static std::unique_ptr<Reservoir> Create(uint64_t rng_seed,
                                             FeatureMode mode = FeatureMode::Raw,
                                             float alpha = 1.0f,
                                             float spectral_radius = -1.0f,
                                             const float* block_scaling = nullptr,
                                             size_t num_inputs = 1)
    {
        if (spectral_radius < 0.0f)
            spectral_radius = DefaultSpectralRadius(mode);

        // Build resolved per-block scaling array
        std::vector<float> scaling(num_inputs);
        for (size_t k = 0; k < num_inputs; k++)
        {
            float s = (block_scaling && block_scaling[k] >= 0.0f)
                          ? block_scaling[k]
                          : DefaultInputScaling(mode);
            scaling[k] = s;
        }

        return std::unique_ptr<Reservoir>(new Reservoir(rng_seed, alpha,
                                                        spectral_radius, std::move(scaling)));
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
                       float spectral_radius, std::vector<float> block_scaling);
    uint64_t rng_seed_;

    alignas(64) float vtx_state_[N]{};
    alignas(64) float vtx_output_[N]{};
    std::vector<float> vtx_input_weight_; // flat [N] — one weight per vertex (block-partitioned)
    std::vector<float> vtx_weight_; // flat [N * NUM_CONNECTIONS]

    size_t num_inputs_ = 1;
    float alpha_ = 1.0f;
    float spectral_radius_ = RawSpectralRadius();
    std::vector<float> block_scaling_; // per-block W_in scaling [num_inputs]

    void Initialize();
    void UpdateState(size_t v);
    [[nodiscard]] float EstimateSpectralRadius() const;
};
