#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include "Reservoir.h"

template <size_t DIM>
class ReservoirCascade : public IReservoir<DIM>
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

    void Step() override;
    void InjectInput(size_t channel, float input) override;
    void Reset() override;

    [[nodiscard]] const float* Outputs() const override;
    [[nodiscard]] size_t OutputSize() const override { return reservoirs_.size() * N; }
    [[nodiscard]] size_t Depth() const { return reservoirs_.size(); }

    void SaveState(float* state_out, float* output_out) const override;
    void RestoreState(const float* state_in, const float* output_in) override;

    [[nodiscard]] uint64_t GetSeed() const override { return reservoirs_[0]->GetSeed(); }
    [[nodiscard]] float GetAlpha() const override { return reservoirs_[0]->GetAlpha(); }
    [[nodiscard]] float GetSpectralRadius() const override { return reservoirs_[0]->GetSpectralRadius(); }
    [[nodiscard]] float GetLeakRate() const override { return reservoirs_[0]->GetLeakRate(); }
    [[nodiscard]] float GetInputScaling() const override { return reservoirs_[0]->GetInputScaling(); }

private:

    explicit ReservoirCascade(size_t depth, const ReservoirConfig& cfg);

    std::vector<std::unique_ptr<Reservoir<DIM>>> reservoirs_;
    std::vector<size_t> input_rotations_;
    mutable std::vector<float> output_buf_;
};
