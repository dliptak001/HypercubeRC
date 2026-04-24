#include "ReservoirCascade.h"
#include <stdexcept>


template <size_t DIM>
ReservoirCascade<DIM>::ReservoirCascade(size_t depth, const ReservoirConfig& cfg)
{
    if (depth == 0)
        throw std::invalid_argument("cascade depth must be >= 1");

    reservoirs_.reserve(depth);
    input_rotations_.reserve(depth);

    for (size_t i = 0; i < depth; ++i)
    {
        reservoirs_.push_back(Reservoir<DIM>::Create(cfg));
        input_rotations_.push_back(i * (N / depth));
    }

    output_buf_.resize(depth * N, 0.0f);
    coupling_mode_ = cfg.coupling_mode;
    coupling_scratch_.resize(N, 0.0f);
}

template <size_t DIM>
void ReservoirCascade<DIM>::Step()
{
    reservoirs_[0]->Step();
    for (size_t i = 1; i < reservoirs_.size(); ++i)
    {
        const float* src = reservoirs_[i - 1]->Outputs();
        const float* signal = src;
        if (coupling_mode_ != CouplingMode::Raw)
        {
            ConditionSignal(src, coupling_scratch_.data());
            signal = coupling_scratch_.data();
        }
        reservoirs_[i]->InjectState(signal, input_rotations_[i]);
        reservoirs_[i]->Step();
    }
}

template <size_t DIM>
void ReservoirCascade<DIM>::ConditionSignal(const float* src, float* dst) const
{
    switch (coupling_mode_)
    {
    case CouplingMode::Binarize:
        for (size_t v = 0; v < N; ++v)
            dst[v] = (src[v] >= 0.0f) ? 1.0f : -1.0f;
        break;

    case CouplingMode::Normalize:
    {
        float lo = src[0], hi = src[0];
        for (size_t v = 1; v < N; ++v) {
            if (src[v] < lo) lo = src[v];
            if (src[v] > hi) hi = src[v];
        }
        float range = hi - lo;
        if (range < 1e-12f) {
            for (size_t v = 0; v < N; ++v) dst[v] = 0.0f;
        } else {
            float inv = 2.0f / range;
            for (size_t v = 0; v < N; ++v)
                dst[v] = (src[v] - lo) * inv - 1.0f;
        }
        break;
    }

    case CouplingMode::Center:
    {
        float sum = 0.0f;
        for (size_t v = 0; v < N; ++v) sum += src[v];
        float mean = sum / static_cast<float>(N);
        for (size_t v = 0; v < N; ++v)
            dst[v] = src[v] - mean;
        break;
    }

    default:
        break;
    }
}

template <size_t DIM>
void ReservoirCascade<DIM>::InjectInput(size_t channel, float input)
{
    reservoirs_[0]->InjectInput(channel, input);
}

template <size_t DIM>
const float* ReservoirCascade<DIM>::Outputs() const
{
    for (size_t i = 0; i < reservoirs_.size(); ++i)
        std::memcpy(output_buf_.data() + i * N, reservoirs_[i]->Outputs(), N * sizeof(float));
    return output_buf_.data();
}

template <size_t DIM>
void ReservoirCascade<DIM>::SaveState(float* state_out, float* output_out) const
{
    for (size_t i = 0; i < reservoirs_.size(); ++i)
        reservoirs_[i]->SaveState(state_out + i * N, output_out + i * N);
}

template <size_t DIM>
void ReservoirCascade<DIM>::RestoreState(const float* state_in, const float* output_in)
{
    for (size_t i = 0; i < reservoirs_.size(); ++i)
        reservoirs_[i]->RestoreState(state_in + i * N, output_in + i * N);
}

template <size_t DIM>
void ReservoirCascade<DIM>::Reset()
{
    for (auto& r : reservoirs_)
        r->Reset();
}

// Explicit template instantiations (DIM 5-16)
template class ReservoirCascade<5>;
template class ReservoirCascade<6>;
template class ReservoirCascade<7>;
template class ReservoirCascade<8>;
template class ReservoirCascade<9>;
template class ReservoirCascade<10>;
template class ReservoirCascade<11>;
template class ReservoirCascade<12>;
template class ReservoirCascade<13>;
template class ReservoirCascade<14>;
template class ReservoirCascade<15>;
template class ReservoirCascade<16>;
