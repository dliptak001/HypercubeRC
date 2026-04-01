#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <cstring>
#include <memory>
#include <variant>
#include <vector>
#include "Reservoir.h"
#include "TranslationLayer.h"
#include "LinearReadout.h"
#include "RidgeRegression.h"

enum class ReadoutType { Linear, Ridge };
enum class FeatureMode { Raw, Translated };

/// @brief Echo-state network implementing the full pipeline:
///        Reservoir -> [Output Selection] -> [Translation] -> Readout.
///
/// ESN owns all stages of the pipeline. Construct with a ReservoirConfig,
/// a ReadoutType, and a FeatureMode, then drive, train, and predict:
///
///     ESN<6> esn(cfg, ReadoutType::Ridge, FeatureMode::Translated);
///     esn.Warmup(inputs, 200);
///     esn.Run(inputs + 200, total);
///     esn.Train(targets, train_size);
///     double r2 = esn.R2(targets, train_size, test_size);
///
/// **Feature modes.**
///   - Raw: M stride-selected vertex states (M features per timestep).
///   - Translated: M selected -> 2.5M features via [x | x² | x*x_antipodal].
///
/// **Training.** Train() uses sensible defaults for both readout types.
/// Power users can call the Ridge overload with a custom lambda, or the
/// Linear overload with custom SGD parameters (lr, epochs).
///
/// **State access.** States(), SelectedStates(), and Features() remain
/// available for direct access (diagnostics, analysis, custom readouts).
///
/// @tparam DIM Hypercube dimension (5-12). Vertex count is 2^DIM.
template <size_t DIM>
class ESN
{
    static constexpr size_t N = 1ULL << DIM;

public:
    explicit ESN(const ReservoirConfig& cfg,
                 ReadoutType readout_type = ReadoutType::Ridge,
                 FeatureMode feature_mode = FeatureMode::Translated)
        : reservoir_(Reservoir<DIM>::Create(cfg)),
          readout_type_(readout_type),
          feature_mode_(feature_mode)
    {
        output_fraction_ = cfg.output_fraction;
        size_t M = std::max<size_t>(1, static_cast<size_t>(std::round(N * output_fraction_)));
        output_stride_ = std::max<size_t>(1, N / M);
        num_output_verts_ = (N + output_stride_ - 1) / output_stride_;

        if (readout_type_ == ReadoutType::Ridge)
            readout_ = RidgeRegression{};
        else
            readout_ = LinearReadout{};
    }

    // ---------------------------------------------------------------
    //  Reservoir driving
    // ---------------------------------------------------------------

    /// @brief Drive the reservoir without recording.
    void Warmup(const float* inputs, size_t num_steps)
    {
        for (size_t s = 0; s < num_steps; ++s)
        {
            reservoir_->InjectInput(0, inputs[s]);
            reservoir_->Step();
        }
    }

    /// @brief Drive the reservoir and collect state snapshots.
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

    /// @brief Clear collected states and cached features.
    void ClearStates()
    {
        states_.clear();
        num_collected_ = 0;
        features_.clear();
        features_computed_ = 0;
    }

    // ---------------------------------------------------------------
    //  Training
    // ---------------------------------------------------------------

    /// @brief Train on the first train_size collected states with default parameters.
    void Train(const float* targets, size_t train_size)
    {
        EnsureFeatures();
        size_t nf = NumFeatures();
        std::visit([&](auto& r) {
            r.Train(features_.data(), targets, train_size, nf);
        }, readout_);
    }

    /// @brief Train Ridge readout with custom lambda.
    void Train(const float* targets, size_t train_size, double lambda)
    {
        assert(readout_type_ == ReadoutType::Ridge);
        EnsureFeatures();
        std::get<RidgeRegression>(readout_).Train(
            features_.data(), targets, train_size, NumFeatures(), lambda);
    }

    /// @brief Train Linear readout with custom SGD parameters.
    void Train(const float* targets, size_t train_size,
               float lr, size_t epochs,
               float weight_decay = 1e-4f, float lr_decay = 0.01f)
    {
        assert(readout_type_ == ReadoutType::Linear);
        EnsureFeatures();
        std::get<LinearReadout>(readout_).Train(
            features_.data(), targets, train_size, NumFeatures(),
            lr, epochs, weight_decay, lr_decay);
    }

    /// @brief Incremental training for streaming (Linear readout only).
    void TrainIncremental(const float* targets, size_t train_size,
                          float blend = 0.1f,
                          float lr = 0.0f, size_t epochs = 200,
                          float weight_decay = 1e-4f, float lr_decay = 0.01f)
    {
        assert(readout_type_ == ReadoutType::Linear);
        EnsureFeatures();
        std::get<LinearReadout>(readout_).TrainIncremental(
            features_.data(), targets, train_size, NumFeatures(),
            blend, lr, epochs, weight_decay, lr_decay);
    }

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    /// @brief Predict from collected state at a given timestep index.
    [[nodiscard]] float PredictRaw(size_t timestep) const
    {
        EnsureFeatures();
        assert(timestep < num_collected_);
        size_t nf = NumFeatures();
        const float* f = features_.data() + timestep * nf;
        return std::visit([f](const auto& r) { return r.PredictRaw(f); }, readout_);
    }

    /// @brief R² on a slice of collected states [start, start+count).
    /// Both features and targets are indexed from `start`.
    [[nodiscard]] double R2(const float* targets, size_t start, size_t count) const
    {
        EnsureFeatures();
        assert(start + count <= num_collected_);
        size_t nf = NumFeatures();
        const float* f = features_.data() + start * nf;
        return std::visit([&](const auto& r) { return r.R2(f, targets + start, count); }, readout_);
    }

    /// @brief NRMSE on a slice of collected states [start, start+count).
    /// Both features and targets are indexed from `start`.
    [[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const
    {
        EnsureFeatures();
        assert(start + count <= num_collected_);

        const float* tgt = targets + start;
        double mean = 0.0;
        for (size_t s = 0; s < count; ++s) mean += tgt[s];
        mean /= static_cast<double>(count);

        size_t nf = NumFeatures();
        const float* f = features_.data() + start * nf;
        double var = 0.0, mse = 0.0;
        std::visit([&](const auto& r) {
            for (size_t s = 0; s < count; ++s)
            {
                double y = tgt[s];
                double yh = r.PredictRaw(f + s * nf);
                var += (y - mean) * (y - mean);
                mse += (y - yh) * (y - yh);
            }
        }, readout_);
        if (var < 1e-12) return std::numeric_limits<double>::infinity();
        return std::sqrt(mse / count) / std::sqrt(var / count);
    }

    /// @brief Classification accuracy on a slice of collected states.
    /// Both features and labels are indexed from `start`.
    [[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const
    {
        EnsureFeatures();
        assert(start + count <= num_collected_);
        size_t nf = NumFeatures();
        const float* f = features_.data() + start * nf;
        return std::visit([&](const auto& r) { return r.Accuracy(f, labels + start, count); }, readout_);
    }

    // ---------------------------------------------------------------
    //  State & feature access
    // ---------------------------------------------------------------

    /// @brief Extract stride-selected vertices from collected states.
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

    /// @brief Build and cache features from collected states (incremental).
    /// Only computes features for states added since the last call.
    void EnsureFeatures() const
    {
        if (features_computed_ == num_collected_) return;
        size_t nf = NumFeatures();
        size_t old_count = features_computed_;
        size_t new_count = num_collected_ - old_count;
        features_.resize(num_collected_ * nf);
        if (feature_mode_ == FeatureMode::Translated)
        {
            auto new_feats = TranslationTransformSelected<DIM>(
                states_.data() + old_count * N, new_count, output_stride_, num_output_verts_);
            memcpy(features_.data() + old_count * nf, new_feats.data(), new_count * nf * sizeof(float));
        }
        else
        {
            for (size_t s = 0; s < new_count; ++s)
            {
                const float* src = states_.data() + (old_count + s) * N;
                float* dst = features_.data() + (old_count + s) * nf;
                size_t j = 0;
                for (size_t v = 0; v < N; v += output_stride_)
                    dst[j++] = src[v];
            }
        }
        features_computed_ = num_collected_;
    }

    /// @brief Number of features per timestep (M for Raw, 2.5M for Translated).
    [[nodiscard]] size_t NumFeatures() const
    {
        if (feature_mode_ == FeatureMode::Translated)
            return TranslationFeatureCountSelected(num_output_verts_);
        return num_output_verts_;
    }

    // --- Accessors ---
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] const float* States() const { return states_.data(); }
    [[nodiscard]] const float* Features() const { return features_.data(); }
    [[nodiscard]] float OutputFraction() const { return output_fraction_; }
    [[nodiscard]] size_t OutputStride() const { return output_stride_; }
    [[nodiscard]] size_t NumOutputVerts() const { return num_output_verts_; }
    [[nodiscard]] ReadoutType GetReadoutType() const { return readout_type_; }
    [[nodiscard]] FeatureMode GetFeatureMode() const { return feature_mode_; }
    [[nodiscard]] const Reservoir<DIM>& GetReservoir() const { return *reservoir_; }
    [[nodiscard]] float GetAlpha() const { return reservoir_->GetAlpha(); }

private:
    std::unique_ptr<Reservoir<DIM>> reservoir_;
    ReadoutType readout_type_;
    FeatureMode feature_mode_;
    std::variant<LinearReadout, RidgeRegression> readout_;

    float output_fraction_ = 1.0f;
    size_t output_stride_ = 1;
    size_t num_output_verts_ = N;

    std::vector<float> states_;      // flat: num_collected_ * N floats
    size_t num_collected_ = 0;

    mutable std::vector<float> features_;    // flat: num_collected_ * NumFeatures() floats
    mutable size_t features_computed_ = 0;   // incremental: features valid for [0, features_computed_)
};
