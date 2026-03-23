#include "Reservoir.h"
#include <cstring>
#include <cmath>
#include <stdexcept>


template <size_t DIM>
Reservoir<DIM>::Reservoir(const uint64_t rng_seed,
                    const float alpha,
                    const float leak,
                    const float spectral_radius,
                    const float input_scaling,
                    const size_t num_inputs)
    : rng_seed_(rng_seed),
      num_inputs_(num_inputs),
      alpha_(alpha),
      leak_(leak),
      retain_(1.0f - leak),
      spectral_radius_(spectral_radius),
      input_scaling_(input_scaling)
{
    if (alpha_ <= 0.0f)
        throw std::invalid_argument("alpha must be positive");
    if (leak_ < 0.0f || leak_ > 1.0f)
        throw std::invalid_argument("leak must be in [0.0, 1.0]");

    Initialize();
}

template <size_t DIM>
void Reservoir<DIM>::Initialize()
{
    std::mt19937_64 rng(rng_seed_); std::uniform_real_distribution<double> dist(-1.0, 1.0);

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

    // Initialize W_in — all N vertices receive input
    vtx_input_weight_.resize(N * num_inputs_);
    for (size_t i = 0; i < N * num_inputs_; i++)
        vtx_input_weight_[i] = static_cast<float>(dist(rng)) * input_scaling_;
}

template <size_t DIM>
void Reservoir<DIM>::UpdateState(size_t v)
{
    const float* w = vtx_weight_.data() + v * NUM_CONNECTIONS;
    float s = 0.0f;

    // Hamming shells: masks (1<<(i+1))-1 → 1, 3, 7, 15, ...
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[(v ^ ShellMask(i)) & MASK] * w[i];

    // Nearest neighbors: masks 1<<i → 1, 2, 4, 8, ...
    for (size_t i = 0; i < DIM; i++)
        s += vtx_output_[(v ^ NearestMask(i)) & MASK] * w[DIM + i];

    const float raw = std::tanh(alpha_ * s);
    vtx_state_[v] = retain_ * vtx_state_[v] + leak_ * raw;
}


template <size_t DIM>
float Reservoir<DIM>::EstimateSpectralRadius() const
{
    std::vector<float> x(N), y(N);

    std::mt19937_64 rng(rng_seed_ + 12345); std::uniform_real_distribution<double> dist(-1.0, 1.0);
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
                s += w[i] * x[(v ^ ShellMask(i)) & MASK];
            for (size_t i = 0; i < DIM; i++)
                s += w[DIM + i] * x[(v ^ NearestMask(i)) & MASK];

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
void Reservoir<DIM>::InjectInput(float input)
{
    if (input < -1.0f) input = -1.0f;
    else if (input > 1.0f) input = 1.0f;

    for (size_t v = 0; v < N; v++)
        vtx_output_[v] += vtx_input_weight_[v * num_inputs_] * input;
}

template <size_t DIM>
void Reservoir<DIM>::InjectInput(const float* inputs)
{
    float clamped[64];
    std::vector<float> clamped_heap;
    float* inp;
    if (num_inputs_ <= 64)
        inp = clamped;
    else
    {
        clamped_heap.resize(num_inputs_);
        inp = clamped_heap.data();
    }
    for (size_t k = 0; k < num_inputs_; k++)
    {
        float v = inputs[k];
        if (v < -1.0f) v = -1.0f;
        else if (v > 1.0f) v = 1.0f;
        inp[k] = v;
    }

    for (size_t v = 0; v < N; v++)
    {
        float sum = 0.0f;
        const float* w = vtx_input_weight_.data() + v * num_inputs_;
        for (size_t k = 0; k < num_inputs_; k++)
            sum += w[k] * inp[k];
        vtx_output_[v] += sum;
    }
}

template <size_t DIM>
void Reservoir<DIM>::Step()
{
    #pragma omp parallel for schedule(static) if(N >= 4096)
    for (size_t v = 0; v < N; v++)
        UpdateState(v);
    memcpy(vtx_output_, vtx_state_, N * sizeof(float));
}

// Explicit template instantiations (DIM 5-10)
template class Reservoir<5>;
template class Reservoir<6>;
template class Reservoir<7>;
template class Reservoir<8>;
template class Reservoir<9>;
template class Reservoir<10>;
