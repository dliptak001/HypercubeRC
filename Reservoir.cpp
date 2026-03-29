#include "Reservoir.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>
#include <stdexcept>


template <size_t DIM>
Reservoir<DIM>::Reservoir(const uint64_t rng_seed,
                          const float alpha,
                          const float spectral_radius,
                          std::vector<float> block_scaling)
    : rng_seed_(rng_seed),
      num_inputs_(block_scaling.size()),
      alpha_(alpha),
      spectral_radius_(spectral_radius),
      block_scaling_(std::move(block_scaling))
{
    if (alpha_ <= 0.0f)
        throw std::invalid_argument("alpha must be positive");

    Initialize();
}

template <size_t DIM>
void Reservoir<DIM>::Initialize()
{
    std::mt19937_64 rng(rng_seed_);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    memset(vtx_state_, 0, N * sizeof(float));
    memset(vtx_output_, 0, N * sizeof(float));

    // N * NUM_CONNECTIONS fully independent weights
    const size_t num_weights = N * NUM_CONNECTIONS;
    vtx_weight_.resize(num_weights);
    const float w_scale = 1.0f / std::sqrt(static_cast<float>(NUM_CONNECTIONS));
    for (size_t i = 0; i < num_weights; i++)
        vtx_weight_[i] = static_cast<float>(dist(rng)) * w_scale;

    // Rescale recurrent weights to target spectral radius
    float current_sr = EstimateSpectralRadius();
    if (current_sr > 1e-6f)
    {
        float scale = spectral_radius_ / current_sr;
        for (size_t i = 0; i < vtx_weight_.size(); i++)
            vtx_weight_[i] *= scale;
    }

    // Initialize W_in — one weight per vertex, scaling determined by block assignment
    vtx_input_weight_.resize(N);
    const size_t block_size = N / num_inputs_;
    for (size_t k = 0; k < num_inputs_; k++)
    {
        const size_t start = k * block_size;
        const size_t end = (k + 1 == num_inputs_) ? N : start + block_size;
        for (size_t v = start; v < end; v++)
            vtx_input_weight_[v] = static_cast<float>(dist(rng)) * block_scaling_[k];
    }
}

template <size_t DIM>
void Reservoir<DIM>::Step()
{
    for (size_t v = 0; v < N; v++)
        UpdateState(v);

    memcpy(vtx_output_, vtx_state_, N * sizeof(float));
}

template <size_t DIM>
void Reservoir<DIM>::UpdateState(size_t v)
{
    const float* w = vtx_weight_.data() + v * NUM_CONNECTIONS;
    float s = 0.0f;

    // Recurrent: Hamming shells (1, 3, 7, 15, ...)
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[v ^ ShellMask(i)] * w[i];

    // Recurrent: Nearest neighbors (1, 2, 4, 8, ...)
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[v ^ NearestMask(i)] * w[DIM + i];

    vtx_state_[v] = std::tanh(alpha_ * s);
}

template <size_t DIM>
float Reservoir<DIM>::EstimateSpectralRadius() const
{
    std::vector<float> x(N), y(N);

    std::mt19937_64 rng(rng_seed_ + 12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    float norm = 0.0f;
    for (size_t v = 0; v < N; v++)
    {
        x[v] = static_cast<float>(dist(rng));
        norm += x[v] * x[v];
    }
    norm = std::sqrt(norm);
    for (size_t v = 0; v < N; v++) x[v] /= norm;

    float eigenvalue = 0.0f;
    float prev_eigenvalue = 0.0f;
    for (int iter = 0; iter < 100; iter++)
    {
        for (size_t v = 0; v < N; v++)
        {
            float s = 0.0f;
            const float* w = vtx_weight_.data() + v * NUM_CONNECTIONS;

            for (size_t i = 0; i < DIM; i++)
                s += w[i] * x[v ^ ShellMask(i)];
            for (size_t i = 0; i < DIM; i++)
                s += w[DIM + i] * x[v ^ NearestMask(i)];

            y[v] = s;
        }

        norm = 0.0f;
        for (size_t v = 0; v < N; v++) norm += y[v] * y[v];
        norm = std::sqrt(norm);
        eigenvalue = norm;

        if (norm > 1e-12f)
            for (size_t v = 0; v < N; v++) x[v] = y[v] / norm;

        if (iter > 5 && std::abs(eigenvalue - prev_eigenvalue) < eigenvalue * 1e-6f)
            break;
        prev_eigenvalue = eigenvalue;
    }

    return eigenvalue;
}

template <size_t DIM>
void Reservoir<DIM>::InjectInput(size_t block, float input)
{
    assert(block < num_inputs_ && "InjectInput: block index out of range");
    if (input < -1.0f) input = -1.0f;
    else if (input > 1.0f) input = 1.0f;

    const size_t block_size = N / num_inputs_;
    const size_t start = block * block_size;
    const size_t end = (block + 1 == num_inputs_) ? N : start + block_size;
    for (size_t v = start; v < end; v++)
        vtx_output_[v] += vtx_input_weight_[v] * input;
}

// Explicit template instantiations (DIM 4-10)
template class Reservoir<4>;
template class Reservoir<5>;
template class Reservoir<6>;
template class Reservoir<7>;
template class Reservoir<8>;
template class Reservoir<9>;
template class Reservoir<10>;
