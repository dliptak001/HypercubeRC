#include "ESN.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

template <size_t DIM>
ESN<DIM>::ESN(const ReservoirConfig& cfg, ReadoutType readout_type, FeatureMode feature_mode)
    : reservoir_(Reservoir<DIM>::Create(cfg)),
      readout_type_(readout_type),
      feature_mode_(feature_mode)
{
    // HCNN readout operates on raw reservoir states -- force Raw mode.
    if (readout_type_ == ReadoutType::HCNN)
        feature_mode_ = FeatureMode::Raw;

    num_inputs_ = cfg.num_inputs;
    output_fraction_ = cfg.output_fraction;
    assert(output_fraction_ > 0.0f && output_fraction_ <= 1.0f);
    size_t M = std::max<size_t>(1, static_cast<size_t>(std::round(N * output_fraction_)));
    output_stride_ = std::max<size_t>(1, N / M);
    num_output_verts_ = (N + output_stride_ - 1) / output_stride_;

    if (readout_type_ == ReadoutType::HCNN)
        readout_ = CNNReadout{};
    else if (readout_type_ == ReadoutType::Ridge)
        readout_ = RidgeRegression{};
    else
        readout_ = LinearReadout{};
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
    num_collected_ = 0;
    features_.clear();
    features_computed_ = 0;
}

template <size_t DIM>
void ESN<DIM>::Train(const float* targets, size_t train_size)
{
    if (readout_type_ == ReadoutType::HCNN) {
        // CNN readout uses raw state directly -- no feature pipeline.
        std::get<CNNReadout>(readout_).Train(
            states_.data(), targets, train_size, DIM);
        return;
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    std::visit([&](auto& r) {
        using R = std::decay_t<decltype(r)>;
        if constexpr (!std::is_same_v<R, CNNReadout>) {
            r.Train(features_.data(), targets, train_size, nf);
        }
    }, readout_);
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
                     float lr, size_t epochs,
                     float weight_decay, float lr_decay)
{
    assert(readout_type_ == ReadoutType::Linear);
    EnsureFeatures();
    std::get<LinearReadout>(readout_).Train(
        features_.data(), targets, train_size, NumFeatures(),
        lr, epochs, weight_decay, lr_decay);
}

template <size_t DIM>
void ESN<DIM>::Train(const float* targets, size_t train_size,
                     const CNNReadoutConfig& config)
{
    assert(readout_type_ == ReadoutType::HCNN);
    std::get<CNNReadout>(readout_).Train(
        states_.data(), targets, train_size, DIM, config);
}

template <size_t DIM>
void ESN<DIM>::TrainIncremental(const float* targets, size_t train_size,
                                float blend,
                                float lr, size_t epochs,
                                float weight_decay, float lr_decay)
{
    assert(readout_type_ == ReadoutType::Linear);
    EnsureFeatures();
    std::get<LinearReadout>(readout_).TrainIncremental(
        features_.data(), targets, train_size, NumFeatures(),
        blend, lr, epochs, weight_decay, lr_decay);
}

template <size_t DIM>
float ESN<DIM>::PredictRaw(size_t timestep) const
{
    assert(timestep < num_collected_);
    if (readout_type_ == ReadoutType::HCNN) {
        const float* s = states_.data() + timestep * N;
        return std::get<CNNReadout>(readout_).PredictRaw(s);
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + timestep * nf;
    return std::visit([f](const auto& r) { return r.PredictRaw(f); }, readout_);
}

template <size_t DIM>
void ESN<DIM>::PredictRaw(size_t timestep, float* output) const
{
    assert(timestep < num_collected_);
    if (readout_type_ == ReadoutType::HCNN) {
        const float* s = states_.data() + timestep * N;
        std::get<CNNReadout>(readout_).PredictRaw(s, output);
        return;
    }
    // Linear/Ridge: single scalar output.
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + timestep * nf;
    output[0] = std::visit([f](const auto& r) { return r.PredictRaw(f); }, readout_);
}

template <size_t DIM>
double ESN<DIM>::R2(const float* targets, size_t start, size_t count) const
{
    assert(start + count <= num_collected_);
    if (readout_type_ == ReadoutType::HCNN) {
        const auto& cnn = std::get<CNNReadout>(readout_);
        const float* s = states_.data() + start * N;
        return cnn.R2(s, targets + start * cnn.NumOutputs(), count);
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + start * nf;
    return std::visit([&](const auto& r) { return r.R2(f, targets + start, count); }, readout_);
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

        // Predict all samples once, cache results.
        std::vector<float> preds(count * K);
        for (size_t s = 0; s < count; ++s)
            cnn.PredictRaw(states_.data() + (start + s) * N, preds.data() + s * K);

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
    std::visit([&](const auto& r) {
        for (size_t s = 0; s < count; ++s) {
            double y  = tgt[s];
            double yh = r.PredictRaw(f + s * nf);
            var += (y - mean) * (y - mean);
            mse += (y - yh) * (y - yh);
        }
    }, readout_);
    if (var < 1e-12) return std::numeric_limits<double>::infinity();
    return std::sqrt(mse / count) / std::sqrt(var / count);
}

template <size_t DIM>
double ESN<DIM>::Accuracy(const float* labels, size_t start, size_t count) const
{
    assert(start + count <= num_collected_);
    if (readout_type_ == ReadoutType::HCNN) {
        const float* s = states_.data() + start * N;
        return std::get<CNNReadout>(readout_).Accuracy(s, labels + start, count);
    }
    EnsureFeatures();
    size_t nf = NumFeatures();
    const float* f = features_.data() + start * nf;
    return std::visit([&](const auto& r) { return r.Accuracy(f, labels + start, count); }, readout_);
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

template <size_t DIM>
void ESN<DIM>::EnsureFeatures() const
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

template <size_t DIM>
size_t ESN<DIM>::NumFeatures() const
{
    if (readout_type_ == ReadoutType::HCNN)
        return N;  // CNN operates on full raw state
    if (feature_mode_ == FeatureMode::Translated)
        return TranslationFeatureCountSelected(num_output_verts_);
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
    }, readout_);
    return s;
}

template <size_t DIM>
void ESN<DIM>::SetReadoutState(const ReadoutState& state)
{
    if (!state.is_trained) return;
    std::visit([&](auto& r) {
        using R = std::decay_t<decltype(r)>;
        if constexpr (std::is_same_v<R, RidgeRegression> ||
                      std::is_same_v<R, CNNReadout>) {
            r.SetState(state.weights, state.bias,
                       state.feature_mean, state.feature_scale);
        } else {
            std::vector<float> fw(state.weights.begin(), state.weights.end());
            r.SetState(std::move(fw), static_cast<float>(state.bias),
                       state.feature_mean, state.feature_scale);
        }
    }, readout_);
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
