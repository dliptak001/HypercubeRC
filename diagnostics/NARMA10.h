#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../ESN.h"
#include "../Presets.h"

struct NARMASeq { std::vector<float> inputs; std::vector<float> targets; };

/// Generate NARMA-10 input/target sequence.
/// y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i),i=0..9) + 1.5*u(t-9)*u(t) + 0.1
/// Inputs u(t) uniform in [0, 0.5]. Targets clamped to [0, 1].
inline NARMASeq GenerateNARMA10(uint64_t input_seed, size_t total_steps)
{
    std::mt19937_64 rng(input_seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> u(total_steps);
    std::vector<float> y(total_steps, 0.0f);

    for (size_t t = 0; t < total_steps; ++t)
        u[t] = static_cast<float>(dist(rng)) * 0.25f + 0.25f;

    for (size_t t = 10; t < total_steps - 1; ++t)
    {
        float y_sum = 0.0f;
        for (size_t i = 0; i < 10; ++i) y_sum += y[t - i];
        y[t + 1] = 0.3f * y[t] + 0.05f * y[t] * y_sum
                  + 1.5f * u[t - 9] * u[t] + 0.1f;
        if (y[t + 1] > 1.0f) y[t + 1] = 1.0f;
        if (y[t + 1] < 0.0f) y[t + 1] = 0.0f;
    }
    return {u, y};
}

/// NARMA-10 nonlinear benchmark. Reports NRMSE and wall-clock timing.
template <size_t DIM>
class NARMA10
{
    static constexpr size_t N = 1ULL << DIM;

public:
    struct Result
    {
        double nrmse_hcnn;
        double hcnn_time_s;
    };

    NARMA10(const ReservoirConfig* config = nullptr,
            const ReadoutConfig& hcnn_config = BenchmarkCNNConfig())
        : config_(config), hcnn_config_(hcnn_config)
    {
    }

    static ReadoutConfig BenchmarkCNNConfig()
    {
        return presets::Baseline<DIM>().cnn;
    }

    Result Run()
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        double s_nrmse = 0.0;
        double s_time  = 0.0;

        for (uint64_t seed : Seeds())
        {
            auto [u, y] = GenerateNARMA10(seed + 99, warmup + collect);

            std::vector<float> ri(warmup + collect);
            for (size_t t = 0; t < ri.size(); ++t)
                ri[t] = u[t] * 4.0f - 1.0f;

            std::vector<float> targets(collect);
            for (size_t t = 0; t < collect; ++t)
                targets[t] = y[warmup + t];

            size_t tr = static_cast<size_t>(collect * 0.7);
            size_t te = collect - tr;

            ReservoirConfig cfg = config_ ? *config_ : ReservoirConfig{};
            cfg.seed = seed;
            cfg.output_fraction = 1.0f;

            ESN<DIM> esn(cfg);
            esn.Warmup(ri.data(), warmup);
            esn.Run(ri.data() + warmup, collect);

            auto t0 = std::chrono::steady_clock::now();
            esn.Train(targets.data(), tr, hcnn_config_);
            auto t1 = std::chrono::steady_clock::now();

            s_nrmse += esn.NRMSE(targets.data(), tr, te);
            s_time  += std::chrono::duration<double>(t1 - t0).count();
        }

        double n = static_cast<double>(Seeds().size());
        return {s_nrmse / n, s_time / n};
    }

    static uint64_t DefaultSeed()
    {
        return SurveyedSeed<DIM>();
    }

    static inline thread_local uint64_t single_seed = 0;

private:
    const ReservoirConfig* config_;
    ReadoutConfig hcnn_config_;

    static std::vector<uint64_t> Seeds()
    {
        if (single_seed) return {single_seed};
        return {DefaultSeed()};
    }
};
