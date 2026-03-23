#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <random>

/// Continuous echo-state reservoir on a DIM-dimensional Boolean hypercube (N = 2^DIM vertices).
///
/// Each vertex gathers weighted outputs from 2*DIM neighbors:
///   - DIM Hamming-shell connections: cumulative-bit selectors (1, 3, 7, 15, ...).
///   - DIM nearest-neighbor connections: single-bit flips (1<<0, 1<<1, ...).
///
/// Neighbor masks are computed inline from the loop index — no adjacency storage.
/// Each vertex has fully independent weights (N * 2*DIM total recurrent weights).
/// The weighted sum is passed through tanh(alpha * sum) with exponential leak.
///
/// Recurrent weights are rescaled via power iteration to the target spectral radius.
/// Input is injected into vtx_output_ via per-vertex random projection weights (W_in matrix)
/// distributed across the hypercube by bit-reversal permutation.
/// Supports single-input (float) and multi-input (float*, K channels).
/// Call InjectInput() before Step().
///
/// Readout uses Outputs() which exposes N floats after the synchronous swap.
template <size_t DIM>
class Reservoir
{
    static_assert(DIM >= 5 && DIM <= 10, "DIM must be in 5 <= DIM <= 10");

    static constexpr size_t N = 1ULL << DIM;
    static constexpr size_t MASK = N - 1;
    static constexpr size_t NUM_CONNECTIONS = 2 * DIM;

public:
    static constexpr size_t dim = DIM;
    static constexpr size_t num_vertices = N;
    static constexpr size_t num_connections = NUM_CONNECTIONS;

    /// Per-DIM spectral radius defaults — jointly optimized with input_scaling on
    /// Mackey-Glass h=1, full translation layer, LinearReadout. Alpha=1.0 universal.
    /// Higher DIM → higher SR (more internal dynamics to sustain).
    static constexpr float DefaultSpectralRadius()
    {
        if constexpr (DIM == 5)  return 0.90f;
        if constexpr (DIM == 6)  return 1.00f;
        if constexpr (DIM == 7)  return 1.05f;
        if constexpr (DIM == 8)  return 1.10f;
        if constexpr (DIM == 9)  return 1.00f;
        if constexpr (DIM == 10) return 1.05f;
        return 1.00f;
    }

    /// Per-DIM input scaling defaults — jointly optimized with SR.
    /// Smaller reservoirs need stronger input injection.
    static constexpr float DefaultInputScaling()
    {
        if constexpr (DIM == 5)  return 1.50f;
        if constexpr (DIM == 6)  return 1.00f;
        if constexpr (DIM == 7)  return 0.80f;
        if constexpr (DIM == 8)  return 0.50f;
        if constexpr (DIM == 9)  return 0.40f;
        if constexpr (DIM == 10) return 0.40f;
        return 0.10f;
    }

    static constexpr float default_spectral_radius = DefaultSpectralRadius();
    static constexpr float default_input_scaling = DefaultInputScaling();

    /// Inline neighbor mask computation — no stored adjacency.
    /// Shells [0..DIM):  mask = (1 << (i+1)) - 1  → 1, 3, 7, 15, ...
    /// Nearest [0..DIM): mask = 1 << i             → 1, 2, 4, 8, ...
    static constexpr uint32_t ShellMask(size_t i) { return (1u << (i + 1)) - 1; }
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    static std::unique_ptr<Reservoir> Create(uint64_t rng_seed,
                                          float alpha = 1.0f,
                                          float leak = 1.0f,
                                          float spectral_radius = 0.0f,
                                          float input_scaling = 0.0f,
                                          size_t num_inputs = 1)
    {
        if (spectral_radius <= 0.0f)
            spectral_radius = DefaultSpectralRadius();
        if (input_scaling <= 0.0f)
            input_scaling = DefaultInputScaling();
        return std::unique_ptr<Reservoir>(new Reservoir(rng_seed, alpha, leak,
                                                  spectral_radius, input_scaling,
                                                  num_inputs));
    }

    Reservoir(const Reservoir&) = delete;
    Reservoir& operator=(const Reservoir&) = delete;

    void Step();

    /// @brief Inject single-channel input before Step().
    /// @warning Values outside [-1, 1] are silently clamped.
    void InjectInput(float input);

    /// @brief Inject multi-channel input before Step().
    /// @param inputs Array of num_inputs floats. Values outside [-1, 1] are silently clamped.
    void InjectInput(const float* inputs);

    [[nodiscard]] const float* Outputs() const { return vtx_output_; }

private:
    explicit Reservoir(uint64_t rng_seed, float alpha, float leak,
                    float spectral_radius, float input_scaling,
                    size_t num_inputs);
    uint64_t rng_seed_;

    alignas(64) float vtx_state_[N]{};
    alignas(64) float vtx_output_[N]{};
    std::vector<float> vtx_input_weight_;  // flat [N * num_inputs_] W_in matrix
    std::vector<float> vtx_weight_;        // flat [N * NUM_CONNECTIONS]

    size_t num_inputs_ = 1;
    float alpha_ = 1.0f;
    float leak_ = 1.0f;
    float retain_ = 0.0f;
    float spectral_radius_ = default_spectral_radius;
    float input_scaling_ = default_input_scaling;

    void Initialize();
    void UpdateState(size_t v);
    [[nodiscard]] float EstimateSpectralRadius() const;
};
