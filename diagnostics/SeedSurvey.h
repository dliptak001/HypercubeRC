#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstddef>
#include "MemoryCapacity.h"
#include "MackeyGlass.h"
#include "NARMA10.h"

/// @brief Diagnostic: Seed quality survey.
///
/// Runs a selected benchmark (Mackey-Glass, NARMA-10, or Memory Capacity) over
/// many random seeds at fixed hyperparameters. Reports per-seed results,
/// distribution statistics (mean, stddev, variance, min, max, median), and the
/// best seed found.
///
/// This is step one of testing the hypothesis that seed rank ordering is stable
/// across hyperparameter configurations — i.e., that a seed which ranks well at
/// one (SR, input_scaling) pair will rank well at any other pair.
template <size_t DIM>
class SeedSurvey
{
    static constexpr size_t N = 1ULL << DIM;

public:
    enum class Diagnostic { Memory_Capacity, Mackey_Glass, NARMA_10 };

    struct Result
    {
        std::map<uint64_t, double> seed_results;  // seed -> metric value
        double mean;
        double stddev;
        double variance;
        double min_val;
        double max_val;
        double median;
        uint64_t best_seed;
        double best_value;
    };

    SeedSurvey(int seed_count, float spectral_radius, float input_scaling,
               Diagnostic diagnostic, float output_fraction = 1.0f)
        : seed_count_(seed_count), spectral_radius_(spectral_radius),
          input_scaling_(input_scaling), diagnostic_(diagnostic),
          output_fraction_(output_fraction)
    {
    }

    Result Run()
    {
        // Generate reproducible random seeds
        std::mt19937_64 master_rng(12345);
        std::vector<uint64_t> seeds(seed_count_);
        for (auto& s : seeds) s = master_rng();

        const bool lower_is_better = (diagnostic_ != Diagnostic::Memory_Capacity);
        const int precision = MetricPrecision();

        PrintHeader();

        // Run all seeds in parallel — each thread sets its own thread_local single_seed
        std::vector<double> values(seed_count_);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < seed_count_; ++i)
            values[i] = RunSingleSeed(seeds[i]);

        ResetSingleSeed();

        // Build results map and track min/max sequentially
        std::map<uint64_t, double> seed_results;
        uint64_t min_seed = seeds[0], max_seed = seeds[0];
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();
        for (int i = 0; i < seed_count_; ++i)
        {
            seed_results[seeds[i]] = values[i];
            if (values[i] < min_val) { min_val = values[i]; min_seed = seeds[i]; }
            if (values[i] > max_val) { max_val = values[i]; max_seed = seeds[i]; }
        }

        // Extract values for statistics
        std::vector<double> vals;
        vals.reserve(seed_results.size());
        for (const auto& [s, v] : seed_results) vals.push_back(v);
        std::sort(vals.begin(), vals.end());

        double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
        double mean = sum / static_cast<double>(vals.size());

        double sq_sum = 0.0;
        for (double v : vals) sq_sum += (v - mean) * (v - mean);
        double variance = sq_sum / static_cast<double>(vals.size());
        double stddev = std::sqrt(variance);

        size_t n = vals.size();
        double median = (n % 2 == 0)
            ? (vals[n / 2 - 1] + vals[n / 2]) / 2.0
            : vals[n / 2];

        uint64_t best_seed  = lower_is_better ? min_seed  : max_seed;
        double   best_value = lower_is_better ? min_val   : max_val;

        // Print statistics
        std::cout << "\n--- Distribution ---\n";
        std::cout << "  Count:    " << seed_count_ << "\n";
        std::cout << "  Mean:     " << std::fixed << std::setprecision(precision + 2) << mean << "\n";
        std::cout << "  Stddev:   " << std::fixed << std::setprecision(precision + 2) << stddev << "\n";
        std::cout << "  Variance: " << std::fixed << std::setprecision(precision + 4) << variance << "\n";
        std::cout << "  Min:      " << std::fixed << std::setprecision(precision) << min_val
                  << "  (seed " << min_seed << ")\n";
        std::cout << "  Max:      " << std::fixed << std::setprecision(precision) << max_val
                  << "  (seed " << max_seed << ")\n";
        std::cout << "  Median:   " << std::fixed << std::setprecision(precision) << median << "\n";

        const char* best_label = (diagnostic_ == Diagnostic::Memory_Capacity) ? "MC" : "NRMSE";
        std::cout << "\n  Best seed: " << best_seed
                  << "  (" << best_label << " = "
                  << std::fixed << std::setprecision(precision) << best_value << ")\n";

        return {seed_results, mean, stddev, variance, min_val, max_val, median, best_seed, best_value};
    }

private:
    int seed_count_;
    float spectral_radius_;
    float input_scaling_;
    Diagnostic diagnostic_;
    float output_fraction_;

    int MetricPrecision() const
    {
        switch (diagnostic_)
        {
            case Diagnostic::Mackey_Glass:    return 5;
            case Diagnostic::NARMA_10:        return 4;
            case Diagnostic::Memory_Capacity: return 2;
        }
        return 4;
    }

    const char* DiagnosticName() const
    {
        switch (diagnostic_)
        {
            case Diagnostic::Mackey_Glass:    return "Mackey-Glass";
            case Diagnostic::NARMA_10:        return "NARMA-10";
            case Diagnostic::Memory_Capacity: return "Memory Capacity";
        }
        return "Unknown";
    }

    const char* ReadoutName() const
    {
        return (diagnostic_ == Diagnostic::Memory_Capacity) ? "Linear" : "Ridge";
    }

    void PrintHeader() const
    {
        std::cout << "=== Seed Survey: " << DiagnosticName()
                  << " (DIM=" << DIM << ", N=" << N << ") ===\n";
        std::cout << "SR: " << std::fixed << std::setprecision(2) << spectral_radius_
                  << " | Input scaling: " << std::setprecision(3) << input_scaling_
                  << " | " << seed_count_ << " seeds"
                  << " | " << ReadoutName() << " readout\n\n";
    }

    double RunSingleSeed(uint64_t seed)
    {
        ReservoirConfig cfg{};
        cfg.spectral_radius = spectral_radius_;
        cfg.input_scaling = input_scaling_;

        switch (diagnostic_)
        {
            case Diagnostic::Mackey_Glass:
            {
                MackeyGlass<DIM>::single_seed = seed;
                MackeyGlass<DIM> mg(1, ReadoutType::Ridge, &cfg, output_fraction_);
                auto r = mg.Run();
                return r.nrmse_full;
            }
            case Diagnostic::NARMA_10:
            {
                NARMA10<DIM>::single_seed = seed;
                NARMA10<DIM> narma(ReadoutType::Ridge, &cfg, output_fraction_);
                auto r = narma.Run();
                return r.nrmse_full;
            }
            case Diagnostic::Memory_Capacity:
            {
                MemoryCapacity<DIM>::single_seed = seed;
                MemoryCapacity<DIM> mc(50, &cfg, output_fraction_);
                auto r = mc.Run();
                return r.mc_total;
            }
        }
        return 0.0;
    }

    void ResetSingleSeed()
    {
        switch (diagnostic_)
        {
            case Diagnostic::Mackey_Glass:    MackeyGlass<DIM>::single_seed = 0;    break;
            case Diagnostic::NARMA_10:        NARMA10<DIM>::single_seed = 0;        break;
            case Diagnostic::Memory_Capacity: MemoryCapacity<DIM>::single_seed = 0; break;
        }
    }

    // -----------------------------------------------------------------
    // Spearman rank correlation helpers
    // -----------------------------------------------------------------

    /// Rank a seed→value map. Returns seed→rank (1 = best).
    /// Ties get average rank.
    static std::map<uint64_t, double> ComputeRanks(
        const std::map<uint64_t, double>& seed_results, bool lower_is_better)
    {
        // Sort seeds by value
        std::vector<std::pair<uint64_t, double>> sorted(seed_results.begin(), seed_results.end());
        if (lower_is_better)
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
        else
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

        // Assign ranks with average-rank tie handling
        std::map<uint64_t, double> ranks;
        size_t i = 0;
        while (i < sorted.size())
        {
            size_t j = i + 1;
            while (j < sorted.size() && sorted[j].second == sorted[i].second)
                ++j;
            // Positions i..j-1 are tied; average rank = mean of (i+1)..(j)
            double avg_rank = static_cast<double>(i + 1 + j) / 2.0;
            for (size_t k = i; k < j; ++k)
                ranks[sorted[k].first] = avg_rank;
            i = j;
        }
        return ranks;
    }

    /// Spearman rho between two rank maps (must have identical keys).
    static double ComputeSpearman(
        const std::map<uint64_t, double>& ranks_a,
        const std::map<uint64_t, double>& ranks_b)
    {
        double sum_d2 = 0.0;
        size_t n = ranks_a.size();
        for (const auto& [seed, rank_a] : ranks_a)
        {
            double d = rank_a - ranks_b.at(seed);
            sum_d2 += d * d;
        }
        return 1.0 - (6.0 * sum_d2) / (static_cast<double>(n) * (static_cast<double>(n) * static_cast<double>(n) - 1.0));
    }

public:
    /// Run the same seed population at multiple SR values and compute
    /// pairwise Spearman rank correlation.
    static void RunCorrelation(
        int seed_count,
        const std::vector<float>& sr_values,
        float input_scaling,
        Diagnostic diagnostic,
        float output_fraction = 1.0f)
    {
        const bool lower_is_better = (diagnostic != Diagnostic::Memory_Capacity);

        // Run survey at each SR
        std::vector<Result> results;
        results.reserve(sr_values.size());
        for (float sr : sr_values)
        {
            SeedSurvey<DIM> survey(seed_count, sr, input_scaling, diagnostic, output_fraction);
            results.push_back(survey.Run());
            std::cout << "\n";
        }

        // Compute ranks for each SR
        std::vector<std::map<uint64_t, double>> all_ranks;
        all_ranks.reserve(sr_values.size());
        for (const auto& r : results)
            all_ranks.push_back(ComputeRanks(r.seed_results, lower_is_better));

        // Print correlation matrix
        size_t k = sr_values.size();
        std::cout << "=== Spearman Rank Correlation Matrix ===\n";
        std::cout << "  " << seed_count << " seeds, IS=" << std::fixed << std::setprecision(2)
                  << input_scaling << "\n\n";

        // Column headers
        std::cout << "        ";
        for (float sr : sr_values)
            std::cout << "  SR=" << std::fixed << std::setprecision(2) << sr;
        std::cout << "\n";

        // Rows
        for (size_t i = 0; i < k; ++i)
        {
            std::cout << "  SR=" << std::fixed << std::setprecision(2) << sr_values[i];
            for (size_t j = 0; j < k; ++j)
            {
                double rho = (i == j) ? 1.0 : ComputeSpearman(all_ranks[i], all_ranks[j]);
                std::cout << "  " << std::fixed << std::setprecision(3) << std::setw(6) << rho;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    /// Run the same seed population at multiple input scaling values and compute
    /// pairwise Spearman rank correlation.
    static void RunCorrelationIS(
        int seed_count,
        float spectral_radius,
        const std::vector<float>& is_values,
        Diagnostic diagnostic,
        float output_fraction = 1.0f)
    {
        const bool lower_is_better = (diagnostic != Diagnostic::Memory_Capacity);

        // Run survey at each IS
        std::vector<Result> results;
        results.reserve(is_values.size());
        for (float isc : is_values)
        {
            SeedSurvey<DIM> survey(seed_count, spectral_radius, isc, diagnostic, output_fraction);
            results.push_back(survey.Run());
            std::cout << "\n";
        }

        // Compute ranks for each IS
        std::vector<std::map<uint64_t, double>> all_ranks;
        all_ranks.reserve(is_values.size());
        for (const auto& r : results)
            all_ranks.push_back(ComputeRanks(r.seed_results, lower_is_better));

        // Print correlation matrix
        size_t k = is_values.size();
        std::cout << "=== Spearman Rank Correlation Matrix ===\n";
        std::cout << "  " << seed_count << " seeds, SR=" << std::fixed << std::setprecision(2)
                  << spectral_radius << "\n\n";

        // Column headers
        std::cout << "          ";
        for (float isc : is_values)
            std::cout << "  IS=" << std::fixed << std::setprecision(3) << isc;
        std::cout << "\n";

        // Rows
        for (size_t i = 0; i < k; ++i)
        {
            std::cout << "  IS=" << std::fixed << std::setprecision(3) << is_values[i];
            for (size_t j = 0; j < k; ++j)
            {
                double rho = (i == j) ? 1.0 : ComputeSpearman(all_ranks[i], all_ranks[j]);
                std::cout << "  " << std::fixed << std::setprecision(3) << std::setw(6) << rho;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};
