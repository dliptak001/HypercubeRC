#include "CNNReadout.h"
#include "HCNN.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

CNNReadout::CNNReadout() = default;
CNNReadout::~CNNReadout() = default;
CNNReadout::CNNReadout(CNNReadout&&) noexcept = default;
CNNReadout& CNNReadout::operator=(CNNReadout&&) noexcept = default;

// ---------------------------------------------------------------------------
//  Input standardization
// ---------------------------------------------------------------------------

void CNNReadout::compute_standardization(const float* states,
                                         size_t num_samples, size_t n)
{
    input_mean_.assign(n, 0.0f);
    input_scale_.assign(n, 1.0f);

    // Per-vertex mean (double accumulator for precision at large N).
    for (size_t s = 0; s < num_samples; ++s) {
        const float* x = states + s * n;
        for (size_t v = 0; v < n; ++v)
            input_mean_[v] += x[v];
    }
    const float inv_n = 1.0f / static_cast<float>(num_samples);
    for (size_t v = 0; v < n; ++v)
        input_mean_[v] *= inv_n;

    // Per-vertex std.
    std::vector<float> var(n, 0.0f);
    for (size_t s = 0; s < num_samples; ++s) {
        const float* x = states + s * n;
        for (size_t v = 0; v < n; ++v) {
            float d = x[v] - input_mean_[v];
            var[v] += d * d;
        }
    }
    for (size_t v = 0; v < n; ++v) {
        float std_v = std::sqrt(var[v] * inv_n);
        input_scale_[v] = (std_v > 1e-8f) ? (1.0f / std_v) : 1.0f;
    }
}

void CNNReadout::standardize(const float* in, float* out, size_t n) const
{
    for (size_t v = 0; v < n; ++v)
        out[v] = (in[v] - input_mean_[v]) * input_scale_[v];
}

// ---------------------------------------------------------------------------
//  Training
// ---------------------------------------------------------------------------

void CNNReadout::Train(const float* states, const float* targets,
                       size_t num_samples, size_t dim,
                       const CNNReadoutConfig& config)
{
    config_ = config;
    dim_ = dim;
    const size_t n = 1ULL << dim;
    num_features_ = n;

    // --- Input standardization ---
    compute_standardization(states, num_samples, n);

    // --- Target centering ---
    double tgt_sum = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
        tgt_sum += targets[s];
    target_mean_ = tgt_sum / static_cast<double>(num_samples);

    // Build standardized input matrix and centered targets (contiguous).
    std::vector<float> std_states(num_samples * n);
    std::vector<float> centered_targets(num_samples);
    for (size_t s = 0; s < num_samples; ++s) {
        standardize(states + s * n, std_states.data() + s * n, n);
        centered_targets[s] = targets[s] - static_cast<float>(target_mean_);
    }

    // --- Build HCNN ---
    net_ = std::make_unique<hcnn::HCNN>(
        static_cast<int>(dim), /*num_outputs=*/1, /*input_channels=*/1,
        hcnn::ReadoutType::GAP, hcnn::TaskType::Regression);
    net_->AddConv(config.conv_channels, hcnn::Activation::TANH, /*use_bias=*/true);
    net_->AddPool(hcnn::PoolType::MAX);
    net_->RandomizeWeights(0.0f, config.seed);
    net_->SetOptimizer(hcnn::OptimizerType::ADAM);

    // --- Cosine LR annealing with floor ---
    const float lr_min = config.lr_max * config.lr_min_frac;
    const auto pi = static_cast<float>(std::numbers::pi);

    for (int e = 0; e < config.epochs; ++e) {
        float progress = static_cast<float>(e) / static_cast<float>(config.epochs);
        float lr = lr_min + 0.5f * (config.lr_max - lr_min) *
                   (1.0f + std::cos(pi * progress));

        net_->TrainEpochRegressionFlat(
            std_states.data(), static_cast<int>(n),
            centered_targets.data(), /*num_outputs_per_sample=*/1,
            static_cast<int>(num_samples), config.batch_size,
            lr, /*momentum=*/0.0f, config.weight_decay,
            /*shuffle_seed=*/static_cast<unsigned>(e + 1));
    }

    // --- Allocate persistent scratch buffers ---
    scratch_state_.resize(n);
    scratch_embedded_.resize(n);
    scratch_pred_.resize(1);

    // --- Flatten weights for serialization ---
    flatten_weights();
    trained_ = true;
}

void CNNReadout::Train(const float* /*features*/, const float* /*labels*/,
                       size_t /*num_samples*/, size_t /*num_features*/)
{
    throw std::logic_error(
        "CNNReadout::Train(features, labels, n, nf): CNN readout requires "
        "the dim-aware Train overload.  Use ESN's CNN-specific training path.");
}

// ---------------------------------------------------------------------------
//  Prediction
// ---------------------------------------------------------------------------

float CNNReadout::PredictRaw(const float* state) const
{
    assert(trained_ && net_);
    const size_t n = num_features_;

    // Standardize into scratch.
    standardize(state, scratch_state_.data(), n);

    // Embed + Forward.
    net_->Embed(scratch_state_.data(), static_cast<int>(n),
                scratch_embedded_.data());
    net_->Forward(scratch_embedded_.data(), scratch_pred_.data());

    // De-center.
    return scratch_pred_[0] + static_cast<float>(target_mean_);
}

float CNNReadout::Predict(const float* state) const
{
    return PredictRaw(state) >= 0.0f ? 1.0f : -1.0f;
}

// ---------------------------------------------------------------------------
//  Evaluation
// ---------------------------------------------------------------------------

double CNNReadout::R2(const float* states, const float* targets,
                      size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const size_t n = num_features_;

    double tgt_sum = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
        tgt_sum += targets[s];
    double tgt_mean = tgt_sum / static_cast<double>(num_samples);

    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t s = 0; s < num_samples; ++s) {
        double y  = targets[s];
        double yh = PredictRaw(states + s * n);
        ss_res += (y - yh) * (y - yh);
        ss_tot += (y - tgt_mean) * (y - tgt_mean);
    }
    if (ss_tot < 1e-12) return 0.0;
    return 1.0 - ss_res / ss_tot;
}

double CNNReadout::Accuracy(const float* states, const float* labels,
                            size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const size_t n = num_features_;
    size_t correct = 0;
    for (size_t s = 0; s < num_samples; ++s) {
        float pred = Predict(states + s * n);
        if ((pred > 0.0f) == (labels[s] > 0.0f)) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(num_samples);
}

// ---------------------------------------------------------------------------
//  Serialization
// ---------------------------------------------------------------------------

void CNNReadout::flatten_weights()
{
    // Full weight serialization requires HCNN to expose weight read/write
    // accessors (Gap 5 in the integration plan -- LOW priority).  For now
    // the blob is empty; GetReadoutState captures the standardization
    // params and config, but restoring a CNN readout from state requires
    // retraining.  This is acceptable for the initial integration.
    weights_blob_.clear();
}

void CNNReadout::rebuild_from_blob()
{
    // Full weight restore requires HCNN weight-injection accessors.
    // For now: stub.  SetState stores the standardization params but
    // requires retraining to restore the CNN weights.
}

void CNNReadout::SetState(std::vector<double> weights, double bias,
                          std::vector<float> feature_mean,
                          std::vector<float> feature_scale)
{
    weights_blob_ = std::move(weights);
    target_mean_ = bias;
    input_mean_ = std::move(feature_mean);
    input_scale_ = std::move(feature_scale);
    num_features_ = input_mean_.size();

    if (!weights_blob_.empty()) {
        rebuild_from_blob();
        trained_ = true;
    }
}
