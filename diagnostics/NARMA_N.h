#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../ESN.h"
#include "../Presets.h"

/// Generalized NARMA-N input/target generator.
///
/// Recurrence (order N):
///   y(t) = alpha*y(t-1) + beta*y(t-1)*sum(y(t-1..t-N))
///          + gamma*u(t-N)*u(t) + delta
/// Inputs u(t) are drawn uniform in [u_low, u_high].
///
/// Stability: NARMA-N has a positive-feedback term and can diverge for
/// higher orders. With `use_tanh = true` each y is squashed through tanh
/// (smooth, no information-destroying kink) -- this is the established
/// convention for higher-order NARMA in the RC literature. With
/// `use_tanh = false` the raw recurrence runs unbounded; canonical
/// NARMA-10 with inputs in [0, 0.5] stays bounded on its own.
///
/// Implementation is O(1) per step: an incremental running sum and deque
/// ring buffers replace per-step window recomputation.
///
/// Target alignment -- bug fixed vs. the FractalHypercubeRC original.
/// This generator was ported from FractalHypercubeRC, whose version
/// paired inputs[t] = u(t) with targets[t] = y(t+1) -- a one-step-ahead
/// shift (it allocated an extra sample "+1 for target shift"). That is
/// wrong for NARMA: y(t+1) contains the term gamma*u(t+1)*u(t+1-N), so
/// it depends on the input u(t+1). A reservoir driven only through u(t)
/// has never seen u(t+1), making that term unlearnable -- NRMSE
/// collapses toward 1.0 (predict-the-mean). NARMA is system
/// identification, not forecasting: y(t) is produced from u(t) and
/// u(t-N). This version pairs inputs[t] = u(t) with targets[t] = y(t).
/// Symptom of the bug: HypercubeRC's NARMA-10 suite scored ~0.76-0.82
/// NRMSE with the shifted target vs. ~0.08-0.19 once aligned. The same
/// misalignment is present in FractalHypercubeRC's diagnostic and is a
/// plausible contributor to that project's apparent NARMA "failure"
/// (its scores plateaued near 0.65 across every configuration).
template <typename T = float>
class NARMA_N_Generator
{
public:
    NARMA_N_Generator(size_t N = 10,
                      uint64_t seed = 193,
                      T alpha = T(0.3),
                      T beta = T(0.05),
                      T gamma = T(1.5),
                      T delta = T(0.1),
                      T u_low = T(0.0),
                      T u_high = T(0.5),
                      bool use_tanh = false)
        : N_(N), alpha_(alpha), beta_(beta), gamma_(gamma), delta_(delta),
          u_low_(u_low), u_high_(u_high), use_tanh_(use_tanh),
          rng_(seed), u_dist_(u_low_, u_high_)
    {
        if (N_ < 2)
            throw std::invalid_argument("NARMA_N_Generator: N must be >= 2");
    }

    /// Generate a NARMA-N series for a prediction task.
    /// Returns {inputs_u, targets_y} aligned at the same index:
    /// targets_y[t] = y(t) is the NARMA-N output for input u(t). y(t)
    /// depends only on u(t) and u(t-N) (plus y history), so a reservoir
    /// driven through u(t) has seen everything needed to reproduce it.
    /// (NARMA is system identification, not forecasting -- pairing u(t)
    /// with y(t+1) would make the target depend on the unseen input
    /// u(t+1) and is not learnable.)
    std::pair<std::vector<T>, std::vector<T>>
    generate_prediction_task(size_t num_steps, size_t warmup_steps = 500)
    {
        if (num_steps == 0) return {{}, {}};

        const size_t total = num_steps + warmup_steps;

        std::vector<T> u_series(total);
        std::vector<T> y_series(total, T(0));

        std::deque<T> y_hist(N_, T(0));
        std::deque<T> u_hist(N_, u_dist_(rng_));

        T running_sum_y = T(0);

        for (size_t t = 0; t < total; ++t)
        {
            T u_t = u_dist_(rng_);
            u_series[t] = u_t;

            T y_prev = y_hist.back();
            T sum_y = running_sum_y;
            T u_delayed = u_hist.front();

            T y_t = alpha_ * y_prev
                  + beta_ * y_prev * sum_y
                  + gamma_ * u_delayed * u_t
                  + delta_;

            if (use_tanh_) y_t = std::tanh(y_t);

            y_series[t] = y_t;

            running_sum_y += y_t - y_hist.front();
            y_hist.pop_front();
            y_hist.push_back(y_t);

            u_hist.pop_front();
            u_hist.push_back(u_t);
        }

        // Return the post-warmup portion. inputs[t] and targets[t] are
        // index-aligned: targets[t] = y(t) is the NARMA output for u(t).
        std::vector<T> inputs(num_steps);
        std::vector<T> targets(num_steps);

        for (size_t t = 0; t < num_steps; ++t)
        {
            inputs[t]  = u_series[warmup_steps + t];
            targets[t] = y_series[warmup_steps + t];
        }

        return {inputs, targets};
    }

private:
    size_t N_;
    T alpha_, beta_, gamma_, delta_, u_low_, u_high_;
    bool use_tanh_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<T> u_dist_;
};

/// NARMA-N nonlinear benchmark. Reports NRMSE and wall-clock timing.
/// Defaults to canonical NARMA-10; pass a larger `narma_order` (and
/// `use_tanh = true`) to stress longer memory horizons.
template <size_t DIM>
class NARMA_N
{
    static constexpr size_t N = 1ULL << DIM;

public:
    struct Result
    {
        double nrmse_hcnn;
        double hcnn_time_s;
    };

    NARMA_N(const ReservoirConfig* config = nullptr,
            const ReadoutConfig& hcnn_config = BenchmarkCNNConfig(),
            size_t narma_order = 10,
            bool use_tanh = true)
        : config_(config), hcnn_config_(hcnn_config),
          narma_order_(narma_order), use_tanh_(use_tanh)
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
            // Smaller delta keeps high-order NARMA in a usable range.
            const float delta = (narma_order_ >= 20) ? 0.01f : 0.1f;
            NARMA_N_Generator<float> gen(narma_order_, seed + 99,
                                         0.3f, 0.05f, 1.5f, delta,
                                         0.0f, 0.5f, use_tanh_);
            auto [u, y] = gen.generate_prediction_task(warmup + collect,
                                                       /*warmup_steps=*/0);

            std::vector<float> ri(warmup + collect);
            for (size_t t = 0; t < ri.size(); ++t)
                ri[t] = u[t] * 4.0f - 1.0f;

            // y[warmup + t] is the target aligned with collected step t.
            const float* targets = y.data() + warmup;

            size_t tr = static_cast<size_t>(collect * 0.7);
            size_t te = collect - tr;

            ReservoirConfig cfg = config_ ? *config_ : ReservoirConfig{};
            cfg.seed = seed;
            cfg.output_fraction = 1.0f;

            ESN<DIM> esn(cfg);
            esn.Warmup(ri.data(), warmup);
            esn.Run(ri.data() + warmup, collect);

            auto t0 = std::chrono::steady_clock::now();
            esn.Train(targets, tr, hcnn_config_);
            auto t1 = std::chrono::steady_clock::now();

            s_nrmse += esn.NRMSE(targets, tr, te);
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
    size_t narma_order_;
    bool use_tanh_;

    static std::vector<uint64_t> Seeds()
    {
        if (single_seed) return {single_seed};
        return {DefaultSeed()};
    }
};
