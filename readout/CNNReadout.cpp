#include "CNNReadout.h"
#include "HCNN.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <numeric>


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
//  Architecture
// ---------------------------------------------------------------------------

void CNNReadout::build_architecture()
{
    assert(dim_ >= 5);
    const size_t n = 1ULL << dim_;
    const int d = static_cast<int>(dim_);

    // Auto-size layers: min(DIM - 3, 4), at least 1.
    int layers = (config_.num_layers > 0)
                     ? config_.num_layers
                     : std::min(d - 3, 4);
    layers = std::max(layers, 1);
    assert(layers <= d - 3);

    auto task_type = (config_.task == HCNNTask::Classification)
                         ? hcnn::TaskType::Classification
                         : hcnn::TaskType::Regression;
    net_ = std::make_unique<hcnn::HCNN>(
        d, config_.num_outputs, /*input_channels=*/1,
        hcnn::ReadoutType::GAP, task_type);

    int ch = config_.conv_channels;
    for (int i = 0; i < layers; ++i) {
        net_->AddConv(ch, hcnn::Activation::TANH, /*use_bias=*/true);
        net_->AddPool(hcnn::PoolType::MAX);
        ch *= 2;
    }

    net_->RandomizeWeights(0.0f, config_.seed);

    scratch_state_.resize(n);
    scratch_embedded_.resize(n);
    scratch_pred_.resize(num_outputs_);
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
    num_outputs_ = static_cast<size_t>(config.num_outputs);
    const bool is_classification = (config.task == HCNNTask::Classification);

    // --- Input standardization ---
    compute_standardization(states, num_samples, n);

    // --- Standardize inputs ---
    std::vector<float> std_states(num_samples * n);
    for (size_t s = 0; s < num_samples; ++s)
        standardize(states + s * n, std_states.data() + s * n, n);

    // --- Build HCNN ---
    build_architecture();
    net_->SetOptimizer(hcnn::OptimizerType::ADAM);

    // --- Cosine LR annealing with floor ---
    const float lr_min = config.lr_max * config.lr_min_frac;
    const auto pi = static_cast<float>(std::numbers::pi);

    if (is_classification) {
        // Targets are float class indices — convert to int.
        std::vector<int> int_targets(num_samples);
        for (size_t s = 0; s < num_samples; ++s)
            int_targets[s] = static_cast<int>(targets[s]);

        target_mean_.clear();  // no centering for classification

        for (int e = 0; e < config.epochs; ++e) {
            float progress = static_cast<float>(e) / static_cast<float>(config.epochs);
            float lr = lr_min + 0.5f * (config.lr_max - lr_min) *
                       (1.0f + std::cos(pi * progress));

            net_->TrainEpoch(
                std_states.data(), static_cast<int>(n),
                int_targets.data(),
                static_cast<int>(num_samples), config.batch_size,
                lr, /*momentum=*/0.0f, config.weight_decay,
                /*class_weights=*/nullptr,
                /*shuffle_seed=*/static_cast<unsigned>(e + 1));
        }
    } else {
        // Regression: per-output target centering.
        target_mean_.assign(num_outputs_, 0.0);
        for (size_t s = 0; s < num_samples; ++s)
            for (size_t k = 0; k < num_outputs_; ++k)
                target_mean_[k] += targets[s * num_outputs_ + k];
        for (size_t k = 0; k < num_outputs_; ++k)
            target_mean_[k] /= static_cast<double>(num_samples);

        std::vector<float> centered_targets(num_samples * num_outputs_);
        for (size_t s = 0; s < num_samples; ++s)
            for (size_t k = 0; k < num_outputs_; ++k)
                centered_targets[s * num_outputs_ + k] =
                    targets[s * num_outputs_ + k] - static_cast<float>(target_mean_[k]);

        for (int e = 0; e < config.epochs; ++e) {
            float progress = static_cast<float>(e) / static_cast<float>(config.epochs);
            float lr = lr_min + 0.5f * (config.lr_max - lr_min) *
                       (1.0f + std::cos(pi * progress));

            net_->TrainEpochRegression(
                std_states.data(), static_cast<int>(n),
                centered_targets.data(),
                static_cast<int>(num_samples), config.batch_size,
                lr, /*momentum=*/0.0f, config.weight_decay,
                /*shuffle_seed=*/static_cast<unsigned>(e + 1));
        }
    }

    // --- Allocate persistent scratch buffers ---
    scratch_state_.resize(n);
    scratch_embedded_.resize(n);
    scratch_pred_.resize(num_outputs_);

    // --- Flatten weights for serialization ---
    flatten_weights();
    trained_ = true;
}

// ---------------------------------------------------------------------------
//  Prediction
// ---------------------------------------------------------------------------

void CNNReadout::PredictRaw(const float* state, float* output) const
{
    assert(trained_ && net_);
    const size_t n = num_features_;

    standardize(state, scratch_state_.data(), n);
    net_->Embed(scratch_state_.data(), static_cast<int>(n),
                scratch_embedded_.data());
    net_->Forward(scratch_embedded_.data(), scratch_pred_.data());

    const bool is_regression = (config_.task == HCNNTask::Regression);
    for (size_t k = 0; k < num_outputs_; ++k) {
        output[k] = scratch_pred_[k];
        if (is_regression && !target_mean_.empty())
            output[k] += static_cast<float>(target_mean_[k]);
    }
}

float CNNReadout::PredictRaw(const float* state) const
{
    assert(num_outputs_ == 1);
    float out;
    PredictRaw(state, &out);
    return out;
}

int CNNReadout::PredictClass(const float* state) const
{
    assert(trained_ && net_);
    const size_t n = num_features_;

    standardize(state, scratch_state_.data(), n);
    net_->Embed(scratch_state_.data(), static_cast<int>(n),
                scratch_embedded_.data());
    net_->Forward(scratch_embedded_.data(), scratch_pred_.data());

    return static_cast<int>(
        std::max_element(scratch_pred_.begin(),
                         scratch_pred_.begin() + num_outputs_) -
        scratch_pred_.begin());
}

// ---------------------------------------------------------------------------
//  Evaluation
// ---------------------------------------------------------------------------

double CNNReadout::R2(const float* states, const float* targets,
                      size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const size_t n = num_features_;
    std::vector<float> pred(num_outputs_);

    // Average R2 across outputs.
    double r2_sum = 0.0;
    for (size_t k = 0; k < num_outputs_; ++k) {
        double tgt_mean = 0.0;
        for (size_t s = 0; s < num_samples; ++s)
            tgt_mean += targets[s * num_outputs_ + k];
        tgt_mean /= static_cast<double>(num_samples);

        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t s = 0; s < num_samples; ++s) {
            PredictRaw(states + s * n, pred.data());
            double y  = targets[s * num_outputs_ + k];
            double yh = pred[k];
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - tgt_mean) * (y - tgt_mean);
        }
        r2_sum += (ss_tot < 1e-12) ? 0.0 : (1.0 - ss_res / ss_tot);
    }
    return r2_sum / static_cast<double>(num_outputs_);
}

double CNNReadout::Accuracy(const float* states, const float* labels,
                            size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const size_t n = num_features_;
    size_t correct = 0;

    if (num_outputs_ > 1) {
        // Multi-class: argmax vs label.
        for (size_t s = 0; s < num_samples; ++s) {
            int pred = PredictClass(states + s * n);
            if (pred == static_cast<int>(labels[s])) ++correct;
        }
    } else {
        // Binary: threshold at 0.
        for (size_t s = 0; s < num_samples; ++s) {
            float pred_val;
            PredictRaw(states + s * n, &pred_val);
            if ((pred_val > 0.0f) == (labels[s] > 0.0f)) ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(num_samples);
}

// ---------------------------------------------------------------------------
//  Serialization
// ---------------------------------------------------------------------------

void CNNReadout::flatten_weights()
{
    if (!net_) { weights_blob_.clear(); return; }
    auto fw = net_->GetWeights();
    weights_blob_.assign(fw.begin(), fw.end());
}

void CNNReadout::rebuild_from_blob()
{
    if (weights_blob_.empty() || dim_ == 0) return;

    // Reconstruct the network from stored config if needed.
    if (!net_) {
        build_architecture();
    }

    std::vector<float> fw(weights_blob_.begin(), weights_blob_.end());
    net_->SetWeights(fw);
}

void CNNReadout::SetState(std::vector<double> weights, double bias,
                          std::vector<float> feature_mean,
                          std::vector<float> feature_scale)
{
    weights_blob_ = std::move(weights);
    target_mean_.assign(1, bias);
    input_mean_ = std::move(feature_mean);
    input_scale_ = std::move(feature_scale);
    num_features_ = input_mean_.size();

    if (!weights_blob_.empty()) {
        rebuild_from_blob();
        trained_ = true;
    }
}
