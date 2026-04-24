#pragma once

#include <cstddef>
#include <cstdint>

template <size_t DIM>
class IReservoir
{
public:
    virtual ~IReservoir() = default;

    virtual void Step() = 0;
    virtual void InjectInput(size_t channel, float input) = 0;
    virtual void Reset() = 0;

    [[nodiscard]] virtual const float* Outputs() const = 0;
    [[nodiscard]] virtual size_t OutputSize() const = 0;

    virtual void SaveState(float* state_out, float* output_out) const = 0;
    virtual void RestoreState(const float* state_in, const float* output_in) = 0;

    [[nodiscard]] virtual uint64_t GetSeed() const = 0;
    [[nodiscard]] virtual float GetAlpha() const = 0;
    [[nodiscard]] virtual float GetSpectralRadius() const = 0;
    [[nodiscard]] virtual float GetLeakRate() const = 0;
    [[nodiscard]] virtual float GetInputScaling() const = 0;
};
