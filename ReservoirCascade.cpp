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
}

template <size_t DIM>
void ReservoirCascade<DIM>::Step()
{
    reservoirs_[0]->Step();
    for (size_t i = 1; i < reservoirs_.size(); ++i)
    {
        reservoirs_[i]->InjectState(reservoirs_[i - 1]->Outputs(), input_rotations_[i]);
        reservoirs_[i]->Step();
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
