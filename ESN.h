#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include "Reservoir.h"
#include "readout/LinearReadout.h"
#include "readout/RidgeRegression.h"

enum class ReadoutType { Linear, Ridge };

/// @brief Echo-state network wrapper implementing the three-stage pipeline:
///        Reservoir -> Translation -> Readout.
///
/// The reservoir (Reservoir) produces N-dimensional state vectors per timestep. These
/// raw states are collected by Run() and exposed via States(). A **translation layer**
/// (see TranslationLayer.h) transforms these states into 2.5N features (x, x², x*x')
/// before the linear readout, breaking through the tanh nonlinear encoding bottleneck.
///
/// The translation layer is external to this class. Callers access raw states via
/// States(), apply TranslationTransform(), and pass the expanded features to the
/// readout. FitAndEvaluate() operates on raw states only — it exists for simple
/// baselines.
///
/// **Pipeline:**
///
///   Reservoir (N states) -> TranslationLayer (2.5N features) -> Readout
///
/// **Workflow:**
///
/// 1. **Construct** — Creates the reservoir with fixed random weights.
///    Pass FeatureMode::Raw or FeatureMode::Translation to select per-DIM
///    optimized defaults for spectral radius and input scaling:
///
///        // Raw features (default):
///        ESN<8> esn(seed, ReadoutType::Linear);
///
///        // Translation features:
///        ESN<8> esn(seed, ReadoutType::Linear, FeatureMode::Translation);
///
/// 2. **Warmup** — Drive the reservoir to wash out initial conditions (200-500 steps).
///
///        esn.Warmup(inputs, 200);
///
/// 3. **Run** — Drive and record N-dimensional states per step.
///
///        esn.Run(inputs + 200, 1000);
///
/// 4. **Translate + Fit** — Apply TranslationTransform to States(), then train readout
///    on the 2.5N expanded features.
///
/// 5. **ClearStates** — Reset state buffer between independent evaluations.
///
/// **Design notes:**
///   - Inputs are clamped to [-1, +1] by InjectInput.
///   - States are tanh outputs in [-1, +1].
///   - Targets and predictions are continuous (unbounded for regression).
///   - The reservoir produces N = 2^DIM state features per step.
///   - States() exposes the raw flat buffer for external analysis.
///
/// @tparam DIM Hypercube dimension (4-10). Vertex count is 2^DIM.
template <size_t DIM>
class ESN
{
    static constexpr size_t N = 1ULL << DIM;

public:
    ESN(uint64_t rng_seed,
        ReadoutType readout_type = ReadoutType::Linear,
        FeatureMode mode = FeatureMode::Raw,
        float alpha = 1.0f,
        float spectral_radius = -1.0f,
        const float* block_scaling = nullptr,
        size_t num_inputs = 1)
        : reservoir_(Reservoir<DIM>::Create(rng_seed, mode, alpha, spectral_radius,
                                            block_scaling, num_inputs)),
          readout_type_(readout_type)
    {
    }

    /// @brief Drive the reservoir without recording.
    /// @param inputs Array of num_steps scalars (single-input mode).
    void Warmup(const float* inputs, size_t num_steps)
    {
        for (size_t s = 0; s < num_steps; ++s)
        {
            reservoir_->InjectInput(0, inputs[s]);
            reservoir_->Step();
        }
    }

    /// @brief Drive the reservoir and collect state snapshots.
    /// @param inputs Array of num_steps scalars (single-input mode).
    void Run(const float* inputs, size_t num_steps)
    {
        states_.resize((num_collected_ + num_steps) * N);
        for (size_t s = 0; s < num_steps; ++s)
        {
            reservoir_->InjectInput(0, inputs[s]);
            reservoir_->Step();

            const float* out = reservoir_->Outputs();
            memcpy(states_.data() + (num_collected_ + s) * N, out, N * sizeof(float));
        }
        num_collected_ += num_steps;
    }

    /// @brief Fit readout on first train_fraction of collected states, evaluate on the rest.
    /// @param targets         One target per collected step, aligned with states. {-1.0, +1.0}.
    /// @param train_fraction  Fraction of collected steps used for training (default 0.7).
    /// @return Test accuracy [0.0, 1.0].
    double FitAndEvaluate(const float* targets, double train_fraction = 0.7)
    {
        if (num_collected_ < 2)
            return 0.0;

        const size_t train_size = static_cast<size_t>(num_collected_ * train_fraction);
        const size_t test_size = num_collected_ - train_size;
        if (train_size == 0 || test_size == 0)
            return 0.0;

        const float* test_states = states_.data() + train_size * N;
        const float* test_targets = targets + train_size;

        if (readout_type_ == ReadoutType::Linear)
        {
            linear_.Train(states_.data(), targets, train_size, N);
            return linear_.Accuracy(test_states, test_targets, test_size);
        }
        else
        {
            ridge_.Train(states_.data(), targets, train_size, N);
            return ridge_.Accuracy(test_states, test_targets, test_size);
        }
    }

    /// @brief Clear collected states. Call between independent evaluations.
    void ClearStates()
    {
        states_.clear();
        num_collected_ = 0;
    }

    // --- Accessors ---
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] const float* States() const { return states_.data(); }
    [[nodiscard]] ReadoutType GetReadoutType() const { return readout_type_; }
    [[nodiscard]] const Reservoir<DIM>& GetReservoir() const { return *reservoir_; }
    [[nodiscard]] float GetAlpha() const { return reservoir_->GetAlpha(); }

private:
    std::unique_ptr<Reservoir<DIM>> reservoir_;
    ReadoutType readout_type_;
    LinearReadout linear_;
    RidgeRegression ridge_;

    std::vector<float> states_; // flat: num_collected_ * N floats
    size_t num_collected_ = 0;
};
