#include "ESN.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

template <size_t DIM>
ESN<DIM>::ESN(const ReservoirConfig& cfg, ReadoutType readout_type)
    : reservoir_(Reservoir<DIM>::Create(cfg)),
      readout_type_(readout_type)
{
    num_inputs_      = cfg.num_inputs;
    output_fraction_ = cfg.output_fraction;

    assert(output_fraction_ > 0.0f && output_fraction_ <= 1.0f);
    size_t M = std::max<size_t>(1, static_cast<size_t>(std::round(N * output_fraction_)));
    output_stride_ = std::max<size_t>(1, N / M);
    num_output_verts_ = (N + output_stride_ - 1) / output_stride_;
    scratch_subsampled_.resize(num_output_verts_);

    // HCNN subsamples reservoir state onto a sub-hypercube via stride selection.
    // HypercubeCNN's convolution uses XOR-neighbor masks (hypercube topology),
    // so the stride MUST be a power of 2 — arbitrary strides produce a subset
    // with no coherent hypercube structure.
    if (readout_type_ == ReadoutType::HCNN &&
        (output_stride_ & (output_stride_ - 1)) != 0)
    {
        throw std::invalid_argument(
            "ESN<HCNN>: output_fraction must yield a power-of-2 stride "
            "(1, 2, 4, 8, 16, ...). Use output_fraction in "
            "{1.0, 0.5, 0.25, 0.125, 0.0625, ...}.");
    }

    if (readout_type_ == ReadoutType::HCNN)
        readout_ = CNNReadout{};
    else
        readout_ = RidgeRegression{};
}

template <size_t DIM>
void ESN<DIM>::Warmup(const float* inputs, size_t num_steps)
{
    const size_t K = num_inputs_;
    for (size_t s = 0; s < num_steps; ++s)
    {
        for (size_t ch = 0; ch < K; ++ch)
            reservoir_->InjectInput(ch, inputs[s * K + ch]);
        reservoir_->Step();
    }
}

template <size_t DIM>
void ESN<DIM>::Run(const float* inputs, size_t num_steps)
{
    const size_t K = num_inputs_;
    states_.resize((num_collected_ + num_steps) * N);
    for (size_t s = 0; s < num_steps; ++s)
    {
        for (size_t ch = 0; ch < K; ++ch)
            reservoir_->InjectInput(ch, inputs[s * K + ch]);
        reservoir_->Step();

        const float* out = reservoir_->Outputs();
        memcpy(states_.data() + (num_collected_ + s) * N, out, N * sizeof(float));
    }
    num_collected_ += num_steps;
}

template <size_t DIM>
void ESN<DIM>::ClearStates()
{
    states_.clear();
    states_.shrink_to_fit();
    num_collected_ = 0;
    features_.clear();
    features_.shrink_to_fit();
    features_computed_ = 0;
}

template <size_t DIM>
void ESN<DIM>::Reset()
{
    reservoir_->Reset();
    ClearStates();
}

template <size_t DIM>
void ESN<DIM>::ResetReservoirOnly()
{
    reservoir_->Reset();
}

template <size_t DIM>
void ESN<DIM>::SaveReservoirState(float* state_out, float* output_out) const
{
    reservoir_->SaveState(state_out, output_out);
}

template <size_t DIM>
void ESN<DIM>::RestoreReservoirState(const float* state_in, const float* output_in)
{
    reservoir_->RestoreState(state_in, output_in);
}

template <size_t DIM>
void ESN<DIM>::Train(const float* targets, size_t train_size)
{
    if (readout_type_ == ReadoutType::HCNN) {
        // CNN readout uses raw state directly -- no feature pipeline.
        // Subsample to the sub-hypercube (stride-selected vertices) before
        // handing off; CNN sees an effective-DIM hypercube of width num_output_verts_.
        auto sub = HCNNStates(0, train_size);
        std::get<CNNReadout>(readout_).Train(
            sub.data(), targets, train_size, EffectiveDIM());
        return;
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    std::get<RidgeRegression>(readout_).Train(
        features_.data(), targets, train_size, nf);
}

template <size_t DIM>
void ESN<DIM>::Train(const float* targets, size_t train_size, double lambda)
{
    assert(readout_type_ == ReadoutType::Ridge);
    EnsureFeatures();
    std::get<RidgeRegression>(readout_).Train(
        features_.data(), targets, train_size, NumFeatures(), lambda);
}

template <size_t DIM>
void ESN<DIM>::Train(const float* targets, size_t train_size,
                     const CNNReadoutConfig& config)
{
    assert(readout_type_ == ReadoutType::HCNN);
    auto sub = HCNNStates(0, train_size);
    std::get<CNNReadout>(readout_).Train(
        sub.data(), targets, train_size, EffectiveDIM(), config);
}

template <size_t DIM>
void ESN<DIM>::Train(const float* targets, size_t train_size,
                     const CNNReadoutConfig& config,
                     CNNTrainHooks& hooks)
{
    assert(readout_type_ == ReadoutType::HCNN);
    auto sub = HCNNStates(0, train_size);
    std::get<CNNReadout>(readout_).Train(
        sub.data(), targets, train_size, EffectiveDIM(), config, hooks);
}

template <size_t DIM>
void ESN<DIM>::InitOnline(const float* warmup_inputs, size_t warmup_count,
                          const CNNReadoutConfig& config)
{
    assert(readout_type_ == ReadoutType::HCNN);
    Run(warmup_inputs, warmup_count);
    auto sub = HCNNStates(0, warmup_count);
    std::get<CNNReadout>(readout_).InitOnline(
        sub.data(), warmup_count, EffectiveDIM(), config);
    ClearStates();
}

template <size_t DIM>
void ESN<DIM>::TrainLiveStep(float target_class, float lr, float weight_decay)
{
    assert(readout_type_ == ReadoutType::HCNN);
    const float* sub = SubsampleIntoScratch(reservoir_->Outputs());
    std::get<CNNReadout>(readout_).TrainOnlineStep(
        sub, static_cast<int>(target_class), lr, weight_decay);
}

template <size_t DIM>
void ESN<DIM>::CopyLiveState(float* out) const
{
    assert(readout_type_ == ReadoutType::HCNN);
    const float* src = reservoir_->Outputs();
    size_t j = 0;
    for (size_t v = 0; v < N; v += output_stride_)
        out[j++] = src[v];
}

template <size_t DIM>
void ESN<DIM>::TrainLiveBatch(const float* states, const int* targets,
                              size_t count, float lr, float weight_decay)
{
    assert(readout_type_ == ReadoutType::HCNN);
    std::get<CNNReadout>(readout_).TrainOnlineBatch(
        states, targets, count, lr, weight_decay);
}

template <size_t DIM>
void ESN<DIM>::TrainLiveStepRegression(const float* target, float lr,
                                       float weight_decay)
{
    assert(readout_type_ == ReadoutType::HCNN);
    const float* sub = SubsampleIntoScratch(reservoir_->Outputs());
    std::get<CNNReadout>(readout_).TrainOnlineStepRegression(
        sub, target, lr, weight_decay);
}

template <size_t DIM>
void ESN<DIM>::TrainLiveBatchRegression(const float* states, const float* targets,
                                        size_t count, float lr, float weight_decay)
{
    assert(readout_type_ == ReadoutType::HCNN);
    std::get<CNNReadout>(readout_).TrainOnlineBatchRegression(
        states, targets, count, lr, weight_decay);
}

template <size_t DIM>
void ESN<DIM>::ComputeTargetCentering(const float* targets, size_t num_samples)
{
    assert(readout_type_ == ReadoutType::HCNN);
    std::get<CNNReadout>(readout_).ComputeTargetCentering(targets, num_samples);
}

template <size_t DIM>
float ESN<DIM>::PredictRaw(size_t timestep) const
{
    assert(timestep < num_collected_);
    if (readout_type_ == ReadoutType::HCNN) {
        return std::get<CNNReadout>(readout_).PredictRaw(HCNNState(timestep));
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + timestep * nf;
    return std::get<RidgeRegression>(readout_).PredictRaw(f);
}

template <size_t DIM>
void ESN<DIM>::PredictRaw(size_t timestep, float* output) const
{
    assert(timestep < num_collected_);
    if (readout_type_ == ReadoutType::HCNN) {
        std::get<CNNReadout>(readout_).PredictRaw(HCNNState(timestep), output);
        return;
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + timestep * nf;
    output[0] = std::get<RidgeRegression>(readout_).PredictRaw(f);
}

template <size_t DIM>
float ESN<DIM>::PredictLiveRaw() const
{
    const float* res_out = reservoir_->Outputs();

    if (readout_type_ == ReadoutType::HCNN) {
        return std::get<CNNReadout>(readout_).PredictRaw(SubsampleIntoScratch(res_out));
    }

    return std::get<RidgeRegression>(readout_).PredictRaw(SubsampleIntoScratch(res_out));
}

template <size_t DIM>
void ESN<DIM>::PredictLiveRaw(float* output) const
{
    if (readout_type_ == ReadoutType::HCNN) {
        std::get<CNNReadout>(readout_).PredictRaw(
            SubsampleIntoScratch(reservoir_->Outputs()), output);
        return;
    }
    // Ridge: single scalar output — delegate to scalar overload.
    output[0] = PredictLiveRaw();
}

template <size_t DIM>
double ESN<DIM>::R2(const float* targets, size_t start, size_t count) const
{
    assert(start + count <= num_collected_);
    if (readout_type_ == ReadoutType::HCNN) {
        const auto& cnn = std::get<CNNReadout>(readout_);
        auto sub = HCNNStates(start, count);
        return cnn.R2(sub.data(), targets + start * cnn.NumOutputs(), count);
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + start * nf;
    return std::get<RidgeRegression>(readout_).R2(f, targets + start, count);
}

template <size_t DIM>
double ESN<DIM>::NRMSE(const float* targets, size_t start, size_t count) const
{
    assert(start + count <= num_collected_);
    if (count == 0) return 0.0;

    if (readout_type_ == ReadoutType::HCNN) {
        const auto& cnn = std::get<CNNReadout>(readout_);
        const size_t K = cnn.NumOutputs();
        const float* tgt = targets + start * K;

        // Predict all samples once, cache results. Subsample per sample into
        // the scratch buffer; cheap relative to the CNN forward pass.
        std::vector<float> preds(count * K);
        for (size_t s = 0; s < count; ++s)
            cnn.PredictRaw(HCNNState(start + s), preds.data() + s * K);

        // Average NRMSE across outputs.
        double nrmse_sum = 0.0;
        for (size_t k = 0; k < K; ++k) {
            double mean = 0.0;
            for (size_t s = 0; s < count; ++s)
                mean += tgt[s * K + k];
            mean /= static_cast<double>(count);

            double var = 0.0, mse_k = 0.0;
            for (size_t s = 0; s < count; ++s) {
                double y  = tgt[s * K + k];
                double yh = preds[s * K + k];
                var += (y - mean) * (y - mean);
                mse_k += (y - yh) * (y - yh);
            }
            if (var < 1e-12)
                nrmse_sum += std::numeric_limits<double>::infinity();
            else
                nrmse_sum += std::sqrt(mse_k / count) / std::sqrt(var / count);
        }
        return nrmse_sum / static_cast<double>(K);
    }

    const float* tgt = targets + start;
    double mean = 0.0;
    for (size_t s = 0; s < count; ++s) mean += tgt[s];
    mean /= static_cast<double>(count);

    double var = 0.0, mse = 0.0;
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + start * nf;
    const auto& ridge = std::get<RidgeRegression>(readout_);
    for (size_t s = 0; s < count; ++s) {
        double y  = tgt[s];
        double yh = ridge.PredictRaw(f + s * nf);
        var += (y - mean) * (y - mean);
        mse += (y - yh) * (y - yh);
    }
    if (var < 1e-12) return std::numeric_limits<double>::infinity();
    return std::sqrt(mse / count) / std::sqrt(var / count);
}

template <size_t DIM>
double ESN<DIM>::Accuracy(const float* labels, size_t start, size_t count) const
{
    assert(start + count <= num_collected_);
    if (readout_type_ == ReadoutType::HCNN) {
        auto sub = HCNNStates(start, count);
        return std::get<CNNReadout>(readout_).Accuracy(sub.data(), labels + start, count);
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + start * nf;
    return std::get<RidgeRegression>(readout_).Accuracy(f, labels + start, count);
}

template <size_t DIM>
size_t ESN<DIM>::NumOutputs() const
{
    if (readout_type_ == ReadoutType::HCNN)
        return std::get<CNNReadout>(readout_).NumOutputs();
    return 1;
}

template <size_t DIM>
std::vector<float> ESN<DIM>::SelectedStates() const
{
    return HCNNStates(0, num_collected_);
}

template <size_t DIM>
void ESN<DIM>::EnsureFeatures() const
{
    // HCNN bypasses the feature cache — it consumes subsampled states
    // directly via HCNNStates(). Skip to avoid wasting memory on a
    // duplicate copy of what HCNNStates() already produces on demand.
    if (readout_type_ == ReadoutType::HCNN) return;
    if (features_computed_ == num_collected_) return;
    size_t nf = NumFeatures();
    size_t old_count = features_computed_;
    size_t new_count = num_collected_ - old_count;
    features_.resize(num_collected_ * nf);
    for (size_t s = 0; s < new_count; ++s)
    {
        const float* src = states_.data() + (old_count + s) * N;
        float* dst = features_.data() + (old_count + s) * nf;
        size_t j = 0;
        for (size_t v = 0; v < N; v += output_stride_)
            dst[j++] = src[v];
    }
    features_computed_ = num_collected_;
}

template <size_t DIM>
size_t ESN<DIM>::NumFeatures() const
{
    return num_output_verts_;
}

template <size_t DIM>
ReservoirConfig ESN<DIM>::GetConfig() const
{
    ReservoirConfig cfg;
    cfg.seed            = reservoir_->GetSeed();
    cfg.alpha           = reservoir_->GetAlpha();
    cfg.spectral_radius = reservoir_->GetSpectralRadius();
    cfg.leak_rate       = reservoir_->GetLeakRate();
    cfg.input_scaling   = reservoir_->GetInputScaling();
    cfg.num_inputs      = num_inputs_;
    cfg.output_fraction = output_fraction_;
    return cfg;
}

template <size_t DIM>
typename ESN<DIM>::ReadoutState ESN<DIM>::GetReadoutState() const
{
    ReadoutState s;
    std::visit([&](const auto& r) {
        s.is_trained = r.NumFeatures() > 0;
        s.bias = static_cast<double>(r.Bias());
        s.feature_mean = r.FeatureMean();
        s.feature_scale = r.FeatureScale();
        const auto& w = r.Weights();
        s.weights.assign(w.begin(), w.end());
        if constexpr (std::is_same_v<std::decay_t<decltype(r)>, CNNReadout>) {
            s.target_mean = r.TargetMean();
        }
    }, readout_);
    return s;
}

template <size_t DIM>
void ESN<DIM>::SetReadoutState(const ReadoutState& state)
{
    if (!state.is_trained) return;
    std::visit([&](auto& r) {
        using R = std::decay_t<decltype(r)>;
        if constexpr (std::is_same_v<R, CNNReadout>) {
            r.SetState(state.weights, state.bias,
                       state.feature_mean, state.feature_scale,
                       state.target_mean);
        } else {
            r.SetState(state.weights, state.bias,
                       state.feature_mean, state.feature_scale);
        }
    }, readout_);
}

template <size_t DIM>
void ESN<DIM>::SetCNNConfig(const CNNReadoutConfig& cfg)
{
    assert(readout_type_ == ReadoutType::HCNN);
    std::get<CNNReadout>(readout_).SetConfig(cfg);
}

// ---------------------------------------------------------------
//  HCNN sub-hypercube subsampling helpers
// ---------------------------------------------------------------

template <size_t DIM>
size_t ESN<DIM>::EffectiveDIM() const
{
    // num_output_verts_ is always a power of 2 when readout_type_ == HCNN
    // (constructor rejects non-power-of-2 strides). For other readouts the
    // value is still meaningful but not necessarily used.
    size_t d = 0;
    for (size_t n = num_output_verts_; n > 1; n >>= 1)
        ++d;
    return d;
}

template <size_t DIM>
const float* ESN<DIM>::SubsampleIntoScratch(const float* src) const
{
    size_t j = 0;
    for (size_t v = 0; v < N; v += output_stride_)
        scratch_subsampled_[j++] = src[v];
    return scratch_subsampled_.data();
}

template <size_t DIM>
const float* ESN<DIM>::HCNNState(size_t timestep) const
{
    return SubsampleIntoScratch(states_.data() + timestep * N);
}

template <size_t DIM>
std::vector<float> ESN<DIM>::HCNNStates(size_t start, size_t count) const
{
    std::vector<float> buf(count * num_output_verts_);
    for (size_t s = 0; s < count; ++s)
    {
        const float* src = states_.data() + (start + s) * N;
        float* dst = buf.data() + s * num_output_verts_;
        size_t j = 0;
        for (size_t v = 0; v < N; v += output_stride_)
            dst[j++] = src[v];
    }
    return buf;
}

// Explicit template instantiations (DIM 5-16)
template class ESN<5>;
template class ESN<6>;
template class ESN<7>;
template class ESN<8>;
template class ESN<9>;
template class ESN<10>;
template class ESN<11>;
template class ESN<12>;
template class ESN<13>;
template class ESN<14>;
template class ESN<15>;
template class ESN<16>;
