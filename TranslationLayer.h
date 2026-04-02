#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

/// @brief Translation layer: expand reservoir states into richer features
///        that help a linear readout decode information trapped inside tanh.
///
/// **Why this exists.** Every reservoir neuron outputs tanh(sum), which
/// compresses its internal dynamics into [-1, +1]. A linear readout can
/// only form weighted sums of these outputs — it cannot "undo" the tanh
/// to recover products or squared terms that the reservoir computed
/// internally. The translation layer creates those nonlinear features
/// explicitly, giving the readout access to information that would
/// otherwise be invisible.
///
/// **Three feature classes** are produced from N raw vertex states:
///
///   1. **x** (N features) — the raw tanh outputs, unchanged.
///
///   2. **x²** (N features) — element-wise squares. These expose the
///      magnitude of each neuron's activation regardless of sign,
///      capturing energy-like information that tanh folds symmetrically.
///
///   3. **x*x'** (N/2 features) — antipodal products. For the first N/2
///      vertices (each antipodal pair counted once), vertex v is multiplied
///      by its antipodal partner v XOR (N-1), the vertex at the opposite
///      corner of the hypercube. These cross-products mix information from
///      maximally distant neurons, capturing long-range correlations across
///      the full diameter of the hypercube.
///
/// **Output layout per sample:**
///   [x_0 .. x_{N-1},  x_0² .. x_{N-1}²,  x_0*x_0' .. x_{N/2-1}*x_{N/2-1}']
///
/// **Total:** N + N + N/2 = 2.5N features per sample.
///
/// A stride-selected variant (`TranslationTransformSelected`) produces the
/// same three feature classes for a subset of M vertices (selected by
/// stride from the full N-state vector). Antipodal partners are looked up
/// from the full N-vector, not from the selected subset. Feature count
/// scales as M + M + M/2 = 2.5M.
///
/// In practice, the translation layer reduces NRMSE by 20-70% on standard
/// benchmarks (largest gains on NARMA-10, which demands nonlinear mixing).
template <size_t DIM>
std::vector<float> TranslationTransform(const float* states, size_t num_samples)
{
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t HALF = N / 2;
    constexpr size_t OUT = N + N + HALF;  // 2.5N
    constexpr size_t COMPLEMENT = N - 1;

    std::vector<float> out(num_samples * OUT);

    for (size_t s = 0; s < num_samples; ++s)
    {
        const float* src = states + s * N;
        float* dst = out.data() + s * OUT;

        // x: raw states
        memcpy(dst, src, N * sizeof(float));

        // x²: squared states
        for (size_t v = 0; v < N; ++v)
            dst[N + v] = src[v] * src[v];

        // x*x': antipodal products
        for (size_t v = 0; v < HALF; ++v)
            dst[N + N + v] = src[v] * src[v ^ COMPLEMENT];
    }

    return out;
}

/// @brief Number of features produced by TranslationTransform: N + N + N/2 = 2.5N.
template <size_t DIM>
constexpr size_t TranslationFeatureCount()
{
    constexpr size_t N = 1ULL << DIM;
    return N + N + N / 2;  // 2.5N
}

/// @brief Number of features produced by TranslationTransformSelected: M + M + M/2 = 2.5M.
inline size_t TranslationFeatureCountSelected(size_t M)
{
    return M + M + M / 2;
}

/// @brief Translation transform for stride-selected output vertices.
///
/// Operates on the full N-state buffer (needed for antipodal lookups) but only
/// produces features for vertices 0, stride, 2*stride, ... giving M output
/// vertices per sample.
///
/// Output layout per sample: [x_sel, x_sel², x_sel*x_anti'] where x_sel has M
/// entries and x_sel*x_anti' has M/2 entries (antipodal partners looked up from
/// the full N-state vector, not from the selected subset).
/// Note: when M is odd, the last selected vertex has no antipodal product.
template <size_t DIM>
std::vector<float> TranslationTransformSelected(const float* full_states,
                                                 size_t num_samples,
                                                 size_t stride,
                                                 size_t M)
{
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t COMPLEMENT = N - 1;
    const size_t HALF_M = M / 2;  // odd M: last selected vertex gets no antipodal product
    const size_t OUT = M + M + HALF_M;

    std::vector<float> out(num_samples * OUT);

    for (size_t s = 0; s < num_samples; ++s)
    {
        const float* src = full_states + s * N;
        float* dst = out.data() + s * OUT;

        // x, x², x*x' in a single pass over selected vertices
        size_t j = 0;
        for (size_t v = 0; v < N; v += stride)
        {
            dst[j] = src[v];                           // raw
            dst[M + j] = src[v] * src[v];              // squared
            if (j < HALF_M)
                dst[M + M + j] = src[v] * src[v ^ COMPLEMENT];  // antipodal
            ++j;
        }
    }

    return out;
}
