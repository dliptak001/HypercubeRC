#include "LinearReadout.h"
#include <cmath>
#include <limits>
#include <cstdint>
#include <numeric>

void LinearReadout::Train(const float* features, const float* labels,
                           size_t num_samples, size_t num_features,
                           float learning_rate, size_t max_epochs,
                           float weight_decay, float lr_decay)
{
    num_features_ = num_features;
    weights_.assign(num_features, 0.0f);
    bias_ = 0.0f;

    // Auto learning rate: scale inversely with feature count
    if (learning_rate <= 0.0f)
        learning_rate = 1.0f / static_cast<float>(num_features);

    // Compute per-feature mean and std from training data
    feature_mean_.assign(num_features, 0.0f);
    feature_scale_.assign(num_features, 1.0f);

    for (size_t s = 0; s < num_samples; ++s)
    {
        const float* x = features + s * num_features;
        for (size_t f = 0; f < num_features; ++f)
            feature_mean_[f] += x[f];
    }
    for (size_t f = 0; f < num_features; ++f)
        feature_mean_[f] /= static_cast<float>(num_samples);

    // Compute std
    std::vector<float> var(num_features, 0.0f);
    for (size_t s = 0; s < num_samples; ++s)
    {
        const float* x = features + s * num_features;
        for (size_t f = 0; f < num_features; ++f)
        {
            float d = x[f] - feature_mean_[f];
            var[f] += d * d;
        }
    }
    for (size_t f = 0; f < num_features; ++f)
    {
        float std_f = std::sqrt(var[f] / static_cast<float>(num_samples));
        feature_scale_[f] = (std_f > 1e-8f) ? 1.0f / std_f : 1.0f;
    }

    // Standardize training features into a working buffer
    std::vector<float> std_features(num_samples * num_features);
    for (size_t s = 0; s < num_samples; ++s)
    {
        const float* x = features + s * num_features;
        float* xn = std_features.data() + s * num_features;
        for (size_t f = 0; f < num_features; ++f)
            xn[f] = (x[f] - feature_mean_[f]) * feature_scale_[f];
    }

    // Pocket: track the weight vector with lowest MSE
    std::vector<float> pocket_weights(num_features, 0.0f);
    float pocket_bias = 0.0f;
    double pocket_mse = std::numeric_limits<double>::max();

    // Shuffle index for sample presentation order
    std::vector<size_t> order(num_samples);
    std::iota(order.begin(), order.end(), 0);

    for (size_t epoch = 0; epoch < max_epochs; ++epoch)
    {
        const float lr = learning_rate / (1.0f + static_cast<float>(epoch) * lr_decay);

        // Fisher-Yates shuffle with deterministic seed per epoch
        uint64_t rng_state = epoch * 6364136223846793005ULL + 1442695040888963407ULL;
        for (size_t i = num_samples - 1; i > 0; --i)
        {
            rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
            size_t j = static_cast<size_t>((rng_state >> 33) % (i + 1));
            std::swap(order[i], order[j]);
        }

        for (size_t si = 0; si < num_samples; ++si)
        {
            size_t s = order[si];
            const float* x = std_features.data() + s * num_features;

            // Inline prediction on standardized features
            float prediction = bias_;
            for (size_t f = 0; f < num_features; ++f)
                prediction += weights_[f] * x[f];

            const float error = labels[s] - prediction;
            const float decay = 1.0f - lr * weight_decay;
            for (size_t f = 0; f < num_features; ++f)
                weights_[f] = decay * weights_[f] + lr * error * x[f];
            bias_ += lr * error;
        }

        // Pocket check: MSE on standardized training data
        double mse = 0.0;
        for (size_t s = 0; s < num_samples; ++s)
        {
            const float* x = std_features.data() + s * num_features;
            float pred = bias_;
            for (size_t f = 0; f < num_features; ++f)
                pred += weights_[f] * x[f];
            double err = labels[s] - pred;
            mse += err * err;
        }
        mse /= num_samples;

        if (mse < pocket_mse)
        {
            pocket_mse = mse;
            pocket_weights = weights_;
            pocket_bias = bias_;
        }
    }

    // Restore the best weights found
    weights_ = pocket_weights;
    bias_ = pocket_bias;
}

void LinearReadout::TrainIncremental(const float* features, const float* labels,
                                      size_t num_samples, size_t num_features,
                                      float blend,
                                      float learning_rate, size_t max_epochs,
                                      float weight_decay, float lr_decay)
{
    // Clamp blend to (0, 1]
    if (blend <= 0.0f) blend = 0.01f;
    if (blend > 1.0f) blend = 1.0f;

    // If no prior model, just do a full train
    if (weights_.empty())
    {
        Train(features, labels, num_samples, num_features,
              learning_rate, max_epochs, weight_decay, lr_decay);
        return;
    }

    // Feature count must match existing model
    if (num_features != num_features_)
        return;

    // Save existing model
    auto old_weights = weights_;
    float old_bias = bias_;
    auto old_mean = feature_mean_;
    auto old_scale = feature_scale_;

    // Train fresh on new data (resets weights internally)
    Train(features, labels, num_samples, num_features,
          learning_rate, max_epochs, weight_decay, lr_decay);

    // Blend: updated = (1 - blend) * old + blend * new
    float keep = 1.0f - blend;
    for (size_t f = 0; f < num_features; ++f)
    {
        weights_[f] = keep * old_weights[f] + blend * weights_[f];
        feature_mean_[f] = keep * old_mean[f] + blend * feature_mean_[f];
        feature_scale_[f] = keep * old_scale[f] + blend * feature_scale_[f];
    }
    bias_ = keep * old_bias + blend * bias_;
}

float LinearReadout::Predict(const float* features) const
{
    return PredictRaw(features) >= 0.0f ? 1.0f : -1.0f;
}

float LinearReadout::PredictRaw(const float* features) const
{
    float sum = bias_;
    for (size_t f = 0; f < num_features_; ++f)
        sum += weights_[f] * (features[f] - feature_mean_[f]) * feature_scale_[f];
    return sum;
}

double LinearReadout::Accuracy(const float* features, const float* labels,
                                size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    return static_cast<double>(CountCorrect(features, labels, num_samples))
           / static_cast<double>(num_samples);
}

double LinearReadout::R2(const float* features, const float* targets,
                          size_t num_samples) const
{
    if (num_samples == 0) return 0.0;

    double mean = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
        mean += targets[s];
    mean /= num_samples;

    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
    {
        double y = targets[s];
        double y_hat = PredictRaw(features + s * num_features_);
        ss_tot += (y - mean) * (y - mean);
        ss_res += (y - y_hat) * (y - y_hat);
    }

    if (ss_tot < 1e-12) return 0.0;
    return 1.0 - ss_res / ss_tot;
}

double LinearReadout::ComputeMSE(const float* features, const float* targets,
                                  size_t num_samples) const
{
    double sum = 0.0;
    for (size_t s = 0; s < num_samples; ++s)
    {
        double err = targets[s] - PredictRaw(features + s * num_features_);
        sum += err * err;
    }
    return sum / num_samples;
}

size_t LinearReadout::CountCorrect(const float* features, const float* labels,
                                    size_t num_samples) const
{
    size_t correct = 0;
    for (size_t s = 0; s < num_samples; ++s)
    {
        if (Predict(features + s * num_features_) * labels[s] > 0.0f)
            ++correct;
    }
    return correct;
}
