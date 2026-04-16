#include "Reservoir.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>
#include <stdexcept>


template <size_t DIM>
Reservoir<DIM>::Reservoir(const ReservoirConfig& cfg)
    : rng_seed_(cfg.seed),
      num_inputs_(cfg.num_inputs),
      alpha_(cfg.alpha),
      spectral_radius_(cfg.spectral_radius),
      leak_rate_(cfg.leak_rate),
      input_scaling_(cfg.input_scaling)
{
    if (alpha_ <= 0.0f)
        throw std::invalid_argument("alpha must be positive");
    if (spectral_radius_ <= 0.0f)
        throw std::invalid_argument("spectral_radius must be positive");
    if (leak_rate_ <= 0.0f || leak_rate_ > 1.0f)
        throw std::invalid_argument("leak_rate must be in (0.0, 1.0]");
    if (num_inputs_ == 0)
        throw std::invalid_argument("num_inputs must be >= 1");
    // Validate here so ESN doesn't need to — Reservoir doesn't use output_fraction itself
    if (cfg.output_fraction <= 0.0f || cfg.output_fraction > 1.0f)
        throw std::invalid_argument("output_fraction must be in (0.0, 1.0]");

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

    // Initialize W_in — one random weight per vertex, uniform scaling
    vtx_input_weight_.resize(N);
    for (size_t v = 0; v < N; ++v)
        vtx_input_weight_[v] = static_cast<float>(dist(rng)) * input_scaling_;
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

    // Recurrent: Hamming shells (3, 7, 15, ...) — skip distance-1 and antipodal
    for (size_t i = 0; i < NUM_SHELL; i++)
        s += vtx_output_[v ^ ShellMask(i + 1)] * w[i];

    // Recurrent: Nearest neighbors (1, 2, 4, 8, ...)
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[v ^ NearestMask(i)] * w[NUM_SHELL + i];

    const float activation = std::tanh(alpha_ * s);
    vtx_state_[v] = (1.0f - leak_rate_) * vtx_output_[v] + leak_rate_ * activation;
}

// Power iteration on the (non-symmetric) recurrent weight matrix.
// This computes the spectral norm (largest singular value), which is the
// standard proxy for the spectral radius in reservoir computing literature.
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

            for (size_t i = 0; i < NUM_SHELL; i++)
                s += w[i] * x[v ^ ShellMask(i + 1)];
            for (size_t i = 0; i < DIM; i++)
                s += w[NUM_SHELL + i] * x[v ^ NearestMask(i)];

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
void Reservoir<DIM>::InjectInput(size_t channel, float input)
{
    assert(channel < num_inputs_ && "InjectInput: channel index out of range");
    if (input < -1.0f) input = -1.0f;
    else if (input > 1.0f) input = 1.0f;

    for (size_t v = channel; v < N; v += num_inputs_)
        vtx_output_[v] += vtx_input_weight_[v] * input;
}

template <size_t DIM>
void Reservoir<DIM>::Reset()
{
    memset(vtx_state_, 0, N * sizeof(float));
    memset(vtx_output_, 0, N * sizeof(float));
}

// Explicit template instantiations (DIM 5-16)
template class Reservoir<5>;
template class Reservoir<6>;
template class Reservoir<7>;
template class Reservoir<8>;
template class Reservoir<9>;
template class Reservoir<10>;
template class Reservoir<11>;
template class Reservoir<12>;
template class Reservoir<13>;
template class Reservoir<14>;
template class Reservoir<15>;
template class Reservoir<16>;
