#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include "../ESN.h"

/// Effective rank of reservoir state space and input-correlated subspace.
/// Eigenvalue spectrum via power iteration + per-vertex R2 from 64 lagged inputs.
/// 3-seed average.
template <size_t DIM>
class StateRank
{
    static constexpr size_t N = 1ULL << DIM;

public:
    StateRank(const ReservoirConfig* config = nullptr,
              float output_fraction = 1.0f,
              uint64_t single_seed = 0)
        : config_(config), output_fraction_(output_fraction),
          single_seed_(single_seed)
    {
    }

    void RunAndPrint(size_t max_components = 30)
    {
        constexpr size_t warmup = (N < 256) ? 200 : 500;
        constexpr size_t collect = 18 * N;

        ReservoirConfig display_cfg = config_ ? *config_ : ReservoirConfig{};
        if (output_fraction_ != 1.0f)
            display_cfg.output_fraction = output_fraction_;
        PrintHeader(warmup, collect, max_components, display_cfg);

        std::vector<double> ev_sum(max_components, 0.0);
        size_t ev_count = 0;
        double s_mean_r2 = 0.0, s_min_r2 = 0.0, s_max_r2 = 0.0;
        double s_input_pct = 0.0;
        double s_high_r2_pct = 0.0;

        for (uint64_t seed : Seeds())
        {
            std::mt19937_64 rng(seed + 99);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            size_t total = warmup + collect;
            std::vector<float> inputs(total);
            for (size_t i = 0; i < total; ++i)
                inputs[i] = static_cast<float>(dist(rng));

            ReservoirConfig cfg = config_ ? *config_ : ReservoirConfig{};
            cfg.seed = seed;
            if (output_fraction_ != 1.0f)
                cfg.output_fraction = output_fraction_;
            ESN<DIM> esn(cfg);
            esn.Warmup(inputs.data(), warmup);
            esn.Run(inputs.data() + warmup, collect);

            size_t M = esn.NumOutputVerts();
            auto selected = esn.SelectedStates();
            const float* states = selected.data();

            std::vector<double> mean(M, 0.0);
            for (size_t t = 0; t < collect; ++t)
                for (size_t v = 0; v < M; ++v)
                    mean[v] += states[t * M + v];
            for (size_t v = 0; v < M; ++v)
                mean[v] /= collect;

            std::vector<double> centered(collect * M);
            for (size_t t = 0; t < collect; ++t)
                for (size_t v = 0; v < M; ++v)
                    centered[t * M + v] = states[t * M + v] - mean[v];

            auto eigenvalues = ComputeEigenvalues(centered, collect, M, max_components, seed);
            ev_count = std::max(ev_count, eigenvalues.size());
            for (size_t i = 0; i < eigenvalues.size(); ++i)
                ev_sum[i] += eigenvalues[i];

            auto [mean_r2, min_r2, max_r2, input_pct, high_r2_pct] =
                ComputeInputCorrelation(states, inputs.data() + warmup, collect, M);

            s_mean_r2 += mean_r2;
            s_min_r2 += min_r2;
            s_max_r2 += max_r2;
            s_input_pct += input_pct;
            s_high_r2_pct += high_r2_pct;
        }

        double n = static_cast<double>(Seeds().size());

        std::cout << "State covariance eigenvalues (top 10):\n";
        std::cout << "  #  | Eigenvalue | % of max | Cumulative %\n";
        std::cout << "  ---+------------+----------+-------------\n";

        double max_ev = ev_sum[0] / n;
        double total_ev = 0.0;
        for (size_t i = 0; i < ev_count; ++i)
            total_ev += ev_sum[i] / n;

        double cumulative = 0.0;
        size_t show = std::min(ev_count, static_cast<size_t>(10));
        size_t eff_rank = 0;
        for (size_t i = 0; i < show; ++i)
        {
            double ev = ev_sum[i] / n;
            cumulative += ev;
            if (ev > max_ev * 0.01) ++eff_rank;
            std::cout << "  " << std::setw(2) << (i + 1)
                      << " | " << std::scientific << std::setprecision(3) << std::setw(10) << ev
                      << " | " << std::fixed << std::setprecision(1) << std::setw(7)
                      << (ev / max_ev * 100.0) << "%"
                      << " | " << std::setw(7) << (cumulative / total_ev * 100.0) << "%\n";
        }
        for (size_t i = show; i < ev_count; ++i)
            if (ev_sum[i] / n > max_ev * 0.01) ++eff_rank;

        std::cout << "  Effective rank (>1% of max): " << eff_rank
                  << " of " << ev_count << " computed\n";

        std::cout << "\nInput-correlated variance (64 lags, 3-seed avg):\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Input-correlated: " << s_input_pct / n << "%\n";
        std::cout << std::setprecision(3);
        std::cout << "  Per-vertex R2 (mean): " << s_mean_r2 / n << "\n";
        std::cout << "  Per-vertex R2 (min/max): " << s_min_r2 / n
                  << " / " << s_max_r2 / n << "\n";
        std::cout << "  Vertices with R2 > 0.5: " << std::setprecision(1)
                  << s_high_r2_pct / n << "%\n";
    }

private:
    const ReservoirConfig* config_;
    float output_fraction_;
    uint64_t single_seed_;

    std::vector<uint64_t> Seeds() const
    {
        if (single_seed_) return {single_seed_};
        return {42, 1042, 2042};
    }

    struct InputCorr
    {
        double mean_r2, min_r2, max_r2, input_pct, high_r2_pct;
    };

    static InputCorr ComputeInputCorrelation(const float* states, const float* input_ptr,
                                              size_t collect, size_t num_verts)
    {
        constexpr size_t K = 64;
        size_t valid = collect - K;

        std::vector<float> lagged(valid * K);
        for (size_t t = 0; t < valid; ++t)
            for (size_t k = 0; k < K; ++k)
                lagged[t * K + k] = input_ptr[K + t - k];

        double total_var = 0.0, input_var = 0.0;
        size_t high_r2_count = 0;
        double min_r2 = 1.0, max_r2 = 0.0, sum_r2 = 0.0;

        for (size_t v = 0; v < num_verts; ++v)
        {
            double mean_s = 0.0;
            for (size_t t = 0; t < valid; ++t)
                mean_s += states[(K + t) * num_verts + v];
            mean_s /= valid;

            double var_s = 0.0;
            for (size_t t = 0; t < valid; ++t)
            {
                double s = states[(K + t) * num_verts + v] - mean_s;
                var_s += s * s;
            }
            var_s /= valid;
            total_var += var_s;

            double sum_corr_sq = 0.0;
            for (size_t k = 0; k < K; ++k)
            {
                double mean_i = 0.0;
                for (size_t t = 0; t < valid; ++t)
                    mean_i += lagged[t * K + k];
                mean_i /= valid;

                double cov = 0.0, var_i = 0.0;
                for (size_t t = 0; t < valid; ++t)
                {
                    double s = states[(K + t) * num_verts + v] - mean_s;
                    double i = lagged[t * K + k] - mean_i;
                    cov += s * i;
                    var_i += i * i;
                }
                if (var_i > 1e-12 && var_s > 1e-12)
                {
                    double corr = cov / std::sqrt(var_s * valid * var_i);
                    sum_corr_sq += corr * corr;
                }
            }

            double r2 = std::min(sum_corr_sq, 1.0);
            input_var += var_s * r2;
            sum_r2 += r2;
            if (r2 > max_r2) max_r2 = r2;
            if (r2 < min_r2) min_r2 = r2;
            if (r2 > 0.5) ++high_r2_count;
        }

        return {sum_r2 / num_verts, min_r2, max_r2,
                total_var > 0 ? input_var / total_var * 100.0 : 0.0,
                high_r2_count * 100.0 / num_verts};
    }

    static std::vector<double> ComputeEigenvalues(const std::vector<double>& centered,
                                                   size_t collect, size_t num_verts,
                                                   size_t max_components, uint64_t seed)
    {
        std::vector<double> eigenvalues;
        std::vector<std::vector<double>> eigenvectors;

        for (size_t comp = 0; comp < max_components && comp < num_verts; ++comp)
        {
            std::vector<double> q(num_verts);
            std::mt19937_64 rng(seed + 99999 + comp);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            double norm = 0.0;
            for (size_t v = 0; v < num_verts; ++v)
            {
                q[v] = dist(rng);
                norm += q[v] * q[v];
            }
            norm = std::sqrt(norm);
            for (size_t v = 0; v < num_verts; ++v) q[v] /= norm;

            double eigenvalue = 0.0;
            for (int iter = 0; iter < 100; ++iter)
            {
                std::vector<double> y(collect, 0.0);
                for (size_t t = 0; t < collect; ++t)
                    for (size_t v = 0; v < num_verts; ++v)
                        y[t] += centered[t * num_verts + v] * q[v];

                std::vector<double> z(num_verts, 0.0);
                for (size_t t = 0; t < collect; ++t)
                    for (size_t v = 0; v < num_verts; ++v)
                        z[v] += centered[t * num_verts + v] * y[t];

                for (size_t v = 0; v < num_verts; ++v)
                    z[v] /= collect;

                for (size_t p = 0; p < eigenvectors.size(); ++p)
                {
                    double dot = 0.0;
                    for (size_t v = 0; v < num_verts; ++v)
                        dot += z[v] * eigenvectors[p][v];
                    for (size_t v = 0; v < num_verts; ++v)
                        z[v] -= dot * eigenvectors[p][v];
                }

                norm = 0.0;
                for (size_t v = 0; v < num_verts; ++v) norm += z[v] * z[v];
                norm = std::sqrt(norm);
                eigenvalue = norm;

                if (norm > 1e-15)
                    for (size_t v = 0; v < num_verts; ++v) q[v] = z[v] / norm;
                else
                    break;
            }

            if (eigenvalue < 1e-12) break;
            eigenvalues.push_back(eigenvalue);
            eigenvectors.push_back(q);
        }

        return eigenvalues;
    }

    void PrintHeader(size_t warmup, size_t collect, size_t max_components,
                     const ReservoirConfig& cfg) const
    {
        auto seeds = Seeds();
        std::cout << "=== State Rank Analysis (raw features, " << seeds.size() << "-seed avg) ===\n";
        std::cout << "Seeds: {";
        for (size_t i = 0; i < seeds.size(); ++i)
            std::cout << (i ? "," : "") << seeds[i];
        std::cout << "} | Alpha: " << cfg.alpha
                  << " | Leak: " << cfg.leak_rate
                  << " | SR: " << cfg.spectral_radius
                  << " | Input scaling: " << cfg.input_scaling << "\n";
        std::cout << "DIM=" << DIM << "  N=" << N
                  << "  Warmup: " << warmup << " | Collect: " << collect
                  << " | Max components: " << max_components << "\n\n";
    }
};
