#include "ESN.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

template <size_t DIM>
ESN<DIM>::ESN(const ReservoirConfig& cfg, ReadoutType readout_type, FeatureMode feature_mode)
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

template <size_t DIM>
void ESN<DIM>::Warmup(const float* inputs, size_t num_steps)
{
    for (size_t s = 0; s < num_steps; ++s)
    {
        reservoir_->InjectInput(0, inputs[s]);
        reservoir_->Step();
    }
}

template <size_t DIM>
void ESN<DIM>::Run(const float* inputs, size_t num_steps)
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
    EnsureFeatures();
    size_t nf = NumFeatures();
    std::visit([&](auto& r) {
        r.Train(features_.data(), targets, train_size, nf);
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
    EnsureFeatures();
    assert(timestep < num_collected_);
    size_t nf = NumFeatures();
    const float* f = features_.data() + timestep * nf;
    return std::visit([f](const auto& r) { return r.PredictRaw(f); }, readout_);
}

template <size_t DIM>
double ESN<DIM>::R2(const float* targets, size_t start, size_t count) const
{
    EnsureFeatures();
    assert(start + count <= num_collected_);
    size_t nf = NumFeatures();
    const float* f = features_.data() + start * nf;
    return std::visit([&](const auto& r) { return r.R2(f, targets + start, count); }, readout_);
}

template <size_t DIM>
double ESN<DIM>::NRMSE(const float* targets, size_t start, size_t count) const
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

template <size_t DIM>
double ESN<DIM>::Accuracy(const float* labels, size_t start, size_t count) const
{
    EnsureFeatures();
    assert(start + count <= num_collected_);
    size_t nf = NumFeatures();
    const float* f = features_.data() + start * nf;
    return std::visit([&](const auto& r) { return r.Accuracy(f, labels + start, count); }, readout_);
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
    if (feature_mode_ == FeatureMode::Translated)
        return TranslationFeatureCountSelected(num_output_verts_);
    return num_output_verts_;
}

// Explicit template instantiations (DIM 5-12)
template class ESN<5>;
template class ESN<6>;
template class ESN<7>;
template class ESN<8>;
template class ESN<9>;
template class ESN<10>;
template class ESN<11>;
template class ESN<12>;
