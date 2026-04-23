#include "ESN.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

template <size_t DIM>
ESN<DIM>::ESN(const ReservoirConfig& cfg)
    : reservoir_(Reservoir<DIM>::Create(cfg))
{
    num_inputs_      = cfg.num_inputs;
    output_fraction_ = cfg.output_fraction;

    assert(output_fraction_ > 0.0f && output_fraction_ <= 1.0f);
    size_t M = std::max<size_t>(1, static_cast<size_t>(std::round(N * output_fraction_)));
    output_stride_ = std::max<size_t>(1, N / M);
    num_output_verts_ = (N + output_stride_ - 1) / output_stride_;
    scratch_subsampled_.resize(num_output_verts_);

    if ((output_stride_ & (output_stride_ - 1)) != 0)
    {
        throw std::invalid_argument(
            "ESN: output_fraction must yield a power-of-2 stride "
            "(1, 2, 4, 8, 16, ...). Use output_fraction in "
            "{1.0, 0.5, 0.25, 0.125, 0.0625, ...}.");
    }
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
    auto sub = ReadoutStates(0, train_size);
    readout_.Train(sub.data(), targets, train_size, EffectiveDIM());
}

template <size_t DIM>
void ESN<DIM>::Train(const float* targets, size_t train_size,
                     const ReadoutConfig& config)
{
    auto sub = ReadoutStates(0, train_size);
    readout_.Train(sub.data(), targets, train_size, EffectiveDIM(), config);
}

template <size_t DIM>
void ESN<DIM>::Train(const float* targets, size_t train_size,
                     const ReadoutConfig& config,
                     CNNTrainHooks& hooks)
{
    auto sub = ReadoutStates(0, train_size);
    readout_.Train(sub.data(), targets, train_size, EffectiveDIM(), config, hooks);
}

template <size_t DIM>
void ESN<DIM>::InitOnline(const float* warmup_inputs, size_t warmup_count,
                          const ReadoutConfig& config)
{
    Run(warmup_inputs, warmup_count);
    auto sub = ReadoutStates(0, warmup_count);
    readout_.InitOnline(sub.data(), warmup_count, EffectiveDIM(), config);
    ClearStates();
}

template <size_t DIM>
void ESN<DIM>::TrainLiveStep(float target_class, float lr, float weight_decay)
{
    const float* sub = SubsampleIntoScratch(reservoir_->Outputs());
    readout_.TrainOnlineStep(sub, static_cast<int>(target_class), lr, weight_decay);
}

template <size_t DIM>
void ESN<DIM>::CopyLiveState(float* out) const
{
    const float* src = reservoir_->Outputs();
    size_t j = 0;
    for (size_t v = 0; v < N; v += output_stride_)
        out[j++] = src[v];
}

template <size_t DIM>
void ESN<DIM>::TrainLiveBatch(const float* states, const int* targets,
                              size_t count, float lr, float weight_decay)
{
    readout_.TrainOnlineBatch(states, targets, count, lr, weight_decay);
}

template <size_t DIM>
void ESN<DIM>::TrainLiveStepRegression(const float* target, float lr,
                                       float weight_decay)
{
    const float* sub = SubsampleIntoScratch(reservoir_->Outputs());
    readout_.TrainOnlineStepRegression(sub, target, lr, weight_decay);
}

template <size_t DIM>
void ESN<DIM>::TrainLiveBatchRegression(const float* states, const float* targets,
                                        size_t count, float lr, float weight_decay)
{
    readout_.TrainOnlineBatchRegression(states, targets, count, lr, weight_decay);
}

template <size_t DIM>
void ESN<DIM>::ComputeTargetCentering(const float* targets, size_t num_samples)
{
    readout_.ComputeTargetCentering(targets, num_samples);
}

template <size_t DIM>
float ESN<DIM>::PredictRaw(size_t timestep) const
{
    assert(timestep < num_collected_);
    return readout_.PredictRaw(HCNNState(timestep));
}

template <size_t DIM>
void ESN<DIM>::PredictRaw(size_t timestep, float* output) const
{
    assert(timestep < num_collected_);
    readout_.PredictRaw(HCNNState(timestep), output);
}

template <size_t DIM>
float ESN<DIM>::PredictLiveRaw() const
{
    return readout_.PredictRaw(SubsampleIntoScratch(reservoir_->Outputs()));
}

template <size_t DIM>
void ESN<DIM>::PredictLiveRaw(float* output) const
{
    readout_.PredictRaw(SubsampleIntoScratch(reservoir_->Outputs()), output);
}

template <size_t DIM>
double ESN<DIM>::R2(const float* targets, size_t start, size_t count) const
{
    assert(start + count <= num_collected_);
    auto sub = ReadoutStates(start, count);
    return readout_.R2(sub.data(), targets + start * readout_.NumOutputs(), count);
}

template <size_t DIM>
double ESN<DIM>::NRMSE(const float* targets, size_t start, size_t count) const
{
    assert(start + count <= num_collected_);
    if (count == 0) return 0.0;

    const size_t K = readout_.NumOutputs();
    const float* tgt = targets + start * K;

    std::vector<float> preds(count * K);
    for (size_t s = 0; s < count; ++s)
        readout_.PredictRaw(HCNNState(start + s), preds.data() + s * K);

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

template <size_t DIM>
double ESN<DIM>::Accuracy(const float* labels, size_t start, size_t count) const
{
    assert(start + count <= num_collected_);
    auto sub = ReadoutStates(start, count);
    return readout_.Accuracy(sub.data(), labels + start, count);
}

template <size_t DIM>
size_t ESN<DIM>::NumOutputs() const
{
    return readout_.NumOutputs();
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
    s.is_trained = readout_.NumFeatures() > 0;
    s.bias = static_cast<double>(readout_.Bias());
    s.feature_mean = readout_.FeatureMean();
    s.feature_scale = readout_.FeatureScale();
    const auto& w = readout_.Weights();
    s.weights.assign(w.begin(), w.end());
    s.target_mean = readout_.TargetMean();
    return s;
}

template <size_t DIM>
void ESN<DIM>::SetReadoutState(const ReadoutState& state)
{
    if (!state.is_trained) return;
    readout_.SetState(state.weights, state.bias,
                      state.feature_mean, state.feature_scale,
                      state.target_mean);
}

template <size_t DIM>
void ESN<DIM>::SetCNNConfig(const ReadoutConfig& cfg)
{
    readout_.SetConfig(cfg);
}

// ---------------------------------------------------------------
//  HCNN sub-hypercube subsampling helpers
// ---------------------------------------------------------------

template <size_t DIM>
size_t ESN<DIM>::EffectiveDIM() const
{
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
std::vector<float> ESN<DIM>::ReadoutStates(size_t start, size_t count) const
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

template <size_t DIM>
std::vector<float> ESN<DIM>::SelectedStates() const
{
    return ReadoutStates(0, num_collected_);
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
