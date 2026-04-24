#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include "Reservoir.h"

template <size_t DIM>
class ReservoirCascade
{

    static_assert(DIM >= 5 && DIM <= 16, "DIM must be in 5 <= DIM <= 16");

    static constexpr size_t N = 1ULL << DIM;

public:

    static constexpr size_t dim = DIM;

    static std::unique_ptr<ReservoirCascade> Create(size_t depth, const ReservoirConfig& cfg)
    {
        return std::unique_ptr<ReservoirCascade>(new ReservoirCascade(depth, cfg));
    }

    ReservoirCascade(const ReservoirCascade&) = delete;
    ReservoirCascade& operator=(const ReservoirCascade&) = delete;

    void Step();
    void InjectInput(size_t channel, float input);
    void Reset();

    [[nodiscard]] const float* Outputs() const;
    [[nodiscard]] size_t TotalOutputSize() const { return reservoirs_.size() * N; }
    [[nodiscard]] size_t Depth() const { return reservoirs_.size(); }

private:

    explicit ReservoirCascade(size_t depth, const ReservoirConfig& cfg);

    std::vector<std::unique_ptr<Reservoir<DIM>>> reservoirs_;
    std::vector<size_t> input_rotations_;
    mutable std::vector<float> output_buf_;
};
