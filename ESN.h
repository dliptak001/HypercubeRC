#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include "Reservoir.h"

enum class ReadoutType { Linear, Ridge };

/// @brief Echo-state network wrapper implementing the three-stage pipeline:
///        Reservoir -> [Output Selection] -> Translation -> Readout.
///
/// The reservoir (Reservoir) produces N-dimensional state vectors per timestep. These
/// raw states are collected by Run() and exposed via States(). A **translation layer**
/// (see TranslationLayer.h) transforms states into 2.5M features (x, x², x*x_anti')
/// before the linear readout, breaking through the tanh nonlinear encoding bottleneck.
///
/// The translation layer is external to this class. Callers access raw states via
/// States(), apply TranslationTransformSelected(), and pass the expanded features to
/// the readout. Readouts (LinearReadout, RidgeRegression) are instantiated externally;
/// ESN stores the ReadoutType for callers to query via GetReadoutType().
///
/// **Pipeline:**
///
///   Reservoir (N states) -> M selected -> TranslationLayer (2.5M features) -> Readout
///
/// **Output selection.** The `output_fraction` parameter (from ReservoirConfig)
/// controls how many of the N vertices are used as readout features. M vertices
/// are selected by stride: 0, stride, 2*stride, ... where stride = N/M. At the
/// default output_fraction=1.0, M=N and all vertices are used. At 0.5, half are
/// used, cutting Ridge readout cost by ~4x (quadratic in feature count).
/// SelectedStates() extracts the M-vertex subset; TranslationTransformSelected()
/// builds 2.5M features from it (using the full N-state buffer for antipodal lookups).
///
/// **Workflow:**
///
/// 1. **Construct** — Creates the reservoir from a ReservoirConfig with concrete
///    parameter values. ReservoirConfig's defaults (SR=0.90, input=0.02)
///    are scale-invariant and work across all DIMs.
///
/// 2. **Warmup** — Drive the reservoir to wash out initial conditions (200-500 steps).
///
///        esn.Warmup(inputs, 200);
///
/// 3. **Run** — Drive and record N-dimensional states per step.
///
///        esn.Run(inputs + 200, 1000);
///
/// 4. **Extract features** — For raw readout, use SelectedStates() (M features).
///    For translation readout, use TranslationTransformSelected() with States(),
///    OutputStride(), and NumOutputVerts() (2.5M features).
///
/// 5. **ClearStates** — Reset state buffer between independent evaluations.
///
/// **Design notes:**
///   - Inputs are clamped to [-1, +1] by InjectInput.
///   - States are tanh outputs in [-1, +1].
///   - Targets and predictions are continuous (unbounded for regression).
///   - The reservoir produces N = 2^DIM state features per step.
///   - States() exposes the raw flat buffer (all N vertices, needed by translation).
///   - SelectedStates() returns only the M stride-selected vertices.
///
/// @tparam DIM Hypercube dimension (4-10). Vertex count is 2^DIM.
template <size_t DIM>
class ESN
{
    static constexpr size_t N = 1ULL << DIM;

public:
    explicit ESN(const ReservoirConfig& cfg,
                 ReadoutType readout_type = ReadoutType::Linear)
        : reservoir_(Reservoir<DIM>::Create(cfg)),
          readout_type_(readout_type)
    {
        output_fraction_ = cfg.output_fraction;
        size_t M = std::max<size_t>(1, static_cast<size_t>(std::round(N * output_fraction_)));
        output_stride_ = std::max<size_t>(1, N / M);
        // Actual count from the stride loop
        num_output_verts_ = (N + output_stride_ - 1) / output_stride_;
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

    /// @brief Clear collected states. Call between independent evaluations.
    void ClearStates()
    {
        states_.clear();
        num_collected_ = 0;
    }

    /// @brief Extract stride-selected vertices from collected states.
    /// Returns a flat buffer of num_collected * num_output_verts floats.
    /// At output_fraction=1.0 (stride=1), this is identical to States().
    [[nodiscard]] std::vector<float> SelectedStates() const
    {
        std::vector<float> selected(num_collected_ * num_output_verts_);
        for (size_t s = 0; s < num_collected_; ++s)
        {
            const float* src = states_.data() + s * N;
            float* dst = selected.data() + s * num_output_verts_;
            size_t j = 0;
            for (size_t v = 0; v < N; v += output_stride_)
                dst[j++] = src[v];
        }
        return selected;
    }

    // --- Accessors ---
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] const float* States() const { return states_.data(); }
    [[nodiscard]] float OutputFraction() const { return output_fraction_; }
    [[nodiscard]] size_t OutputStride() const { return output_stride_; }
    [[nodiscard]] size_t NumOutputVerts() const { return num_output_verts_; }
    [[nodiscard]] ReadoutType GetReadoutType() const { return readout_type_; }
    [[nodiscard]] const Reservoir<DIM>& GetReservoir() const { return *reservoir_; }
    [[nodiscard]] float GetAlpha() const { return reservoir_->GetAlpha(); }

private:
    std::unique_ptr<Reservoir<DIM>> reservoir_;
    ReadoutType readout_type_;
    float output_fraction_ = 1.0f;
    size_t output_stride_ = 1;
    size_t num_output_verts_ = N;

    std::vector<float> states_; // flat: num_collected_ * N floats
    size_t num_collected_ = 0;
};
