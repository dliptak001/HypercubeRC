#pragma once

#include <cstddef>
#include <cstring>
#include <vector>
#include <omp.h>

/// @brief Translation layer: transform reservoir states into expanded feature set.
///
/// Produces three feature classes from N raw vertex states:
///   - x:    N raw states (identity)
///   - x²:   N element-wise squared states
///   - x*x': N/2 antipodal products (state[v] * state[v XOR (N-1)])
///
/// Output layout per sample: [x_0..x_{N-1}, x_0²..x_{N-1}², x_0*x_0'..x_{N/2-1}*x_{N/2-1}']
/// Total features: N + N + N/2 = 2.5N per sample.

/// Full translation: x + x² + x*x'. Returns 2.5N features per sample.
template <size_t DIM>
std::vector<float> TranslationTransform(const float* states, size_t num_samples)
{
    constexpr size_t N = 1ULL << DIM;
    constexpr size_t HALF = N / 2;
    constexpr size_t OUT = N + N + HALF;  // 2.5N
    constexpr size_t COMPLEMENT = N - 1;

    std::vector<float> out(num_samples * OUT);

    #pragma omp parallel for schedule(static) if(num_samples >= 256)
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

/// Feature count for the full translation layer.
template <size_t DIM>
constexpr size_t TranslationFeatureCount()
{
    constexpr size_t N = 1ULL << DIM;
    return N + N + N / 2;  // 2.5N
}
