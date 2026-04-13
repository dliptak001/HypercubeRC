#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../ESN.h"
#include "../readout/CNNReadout.h"
#include "../readout/HCNNPresets.h"
#include "SignalGenerators.h"

/// @brief Interactive HCNN optimizer for the Mackey-Glass benchmark.
///
/// Runs the same Mackey-Glass data pipeline as MackeyGlass<DIM> (horizon
/// prediction, 18*N samples, 70/30 train/test) and lets the caller evaluate
/// one or many CNNReadoutConfig variants against a fixed Ridge baseline.
///
/// Designed for interactive tuning: construct once for a given DIM, call
/// RidgeBaseline() to anchor comparisons, then invoke RunOne() / RunSweep()
/// with candidate configs.  Reservoir states are regenerated per trial from
/// a fixed seed list, so all trials within a run are directly comparable.
///
/// Usage:
///   OptimizeHRCCNNForMG<7> opt;           // DIM=7, single seed, defaults
///   opt.PrintHeader();
///   std::cout << "Ridge NRMSE: " << opt.RidgeBaseline() << "\n";
///
///   std::vector<std::pair<std::string, CNNReadoutConfig>> trials;
///   CNNReadoutConfig a; a.epochs = 100; trials.emplace_back("baseline", a);
///   CNNReadoutConfig b = a; b.conv_channels = 32; trials.emplace_back("ch32", b);
///   opt.RunSweep(trials);
template <size_t DIM>
class OptimizeHRCCNNForMG
{
    static constexpr size_t N       = 1ULL << DIM;
    static constexpr size_t warmup_ = (N < 256) ? 200 : 500;
    static constexpr size_t collect_ = 18 * N;

public:
    struct Result
    {
        double      nrmse;          ///< Test NRMSE averaged across seeds.
        double      elapsed_s;      ///< Wall-clock HCNN training time (averaged).
        int         num_layers;     ///< Conv+Pool pairs actually built.
        int         conv_channels;  ///< Base convolution channels.
        std::string label;
    };

    /// @param num_seeds      Reservoir seeds to average over.  Reservoir
    ///                       quality is per-DIM survey-tuned, so averaging
    ///                       is normally 1; >1 mixes in non-surveyed seeds
    ///                       and compromises the baseline match.
    /// @param num_cnn_seeds  CNN weight-init seeds to average over at the
    ///                       fixed reservoir.  >1 damps training-trajectory
    ///                       variance — small hyperparameter deltas can
    ///                       lead to meaningfully different local minima,
    ///                       so single-seed rankings can be unreliable.
    /// @param rc             Reservoir configuration (nullptr = defaults).
    /// @param horizon        Prediction horizon in steps.
    explicit OptimizeHRCCNNForMG(size_t num_seeds = 1,
                                 size_t num_cnn_seeds = 1,
                                 const ReservoirConfig* rc = nullptr,
                                 size_t horizon = 1)
        : num_seeds_(num_seeds),
          num_cnn_seeds_(num_cnn_seeds),
          horizon_(horizon),
          rc_(rc ? *rc : ReservoirConfig{})
    {
    }

    void PrintHeader() const
    {
        const size_t tr = static_cast<size_t>(collect_ * 0.7);
        std::cout << "=== OptimizeHRCCNNForMG  DIM=" << DIM
                  << "  N=" << N
                  << "  warmup=" << warmup_
                  << "  collect=" << collect_
                  << "  train=" << tr
                  << "  test=" << (collect_ - tr)
                  << "  seeds=" << num_seeds_
                  << "  cnn_seeds=" << num_cnn_seeds_
                  << "  horizon=" << horizon_
                  << " ===\n";
        std::cout << "  started: " << FormatNow() << "\n";
    }

    /// Print a completion line with the current wall-clock timestamp.
    /// Call at the end of main() so long runs have a visible end marker.
    static void PrintCompletion()
    {
        std::cout << "  completed: " << FormatNow() << "\n";
    }

    /// Compute (and cache) Ridge baseline NRMSE on the same trajectories.
    /// Uses raw features (no translation) so the comparison is apples-to-apples
    /// with HCNN, which always consumes raw state.
    double RidgeBaseline()
    {
        if (ridge_cached_ >= 0.0) return ridge_cached_;
        ridge_cached_ = RunRidge(FeatureMode::Raw);
        return ridge_cached_;
    }

    /// Compute (and cache) Ridge baseline NRMSE with the translation layer
    /// applied (2.5N features: [x | x^2 | x * x_antipodal]).  Context for
    /// HCNN — if HCNN beats Ridge+translation, the learned convolution is
    /// outperforming the hand-crafted feature basis.
    double TranslationBaseline()
    {
        if (translation_cached_ >= 0.0) return translation_cached_;
        translation_cached_ = RunRidge(FeatureMode::Translated);
        return translation_cached_;
    }

    /// Run a single HCNN trial, print its row, and return the result.
    /// Averages over num_seeds_ reservoir seeds × num_cnn_seeds_ CNN-init
    /// seeds for noise reduction.  Returns the mean NRMSE and total averaged
    /// wall time.
    Result RunOne(const CNNReadoutConfig& cnn_cfg, const std::string& label = "")
    {
        const size_t tr = static_cast<size_t>(collect_ * 0.7);
        const size_t te = collect_ - tr;

        double nrmse_acc   = 0.0;
        double elapsed_acc = 0.0;
        size_t trial_count = 0;

        for (uint64_t seed : Seeds())
        {
            auto [series, targets] = GenerateSeries();

            ReservoirConfig cfg = rc_;
            cfg.seed = seed;
            cfg.output_fraction = 1.0f;

            ESN<DIM> esn(cfg, ReadoutType::HCNN);
            esn.Warmup(series.data(), warmup_);
            esn.Run(series.data() + warmup_, collect_);

            // Average over CNN init seeds at this fixed reservoir.  The
            // reservoir trajectory is cheap to regenerate but we reuse the
            // ESN (with ClearStates) across CNN seeds to keep the comparison
            // tight — only the HCNN weight init differs.
            for (size_t k = 0; k < num_cnn_seeds_; ++k)
            {
                CNNReadoutConfig cnn_copy = cnn_cfg;
                cnn_copy.seed = static_cast<unsigned>(cnn_cfg.seed + k);

                auto t0 = std::chrono::steady_clock::now();
                esn.Train(targets.data(), tr, cnn_copy);
                auto t1 = std::chrono::steady_clock::now();

                nrmse_acc   += esn.NRMSE(targets.data(), tr, te);
                elapsed_acc += std::chrono::duration<double>(t1 - t0).count();
                ++trial_count;
            }
        }

        const double denom = static_cast<double>(trial_count);
        Result r{
            .nrmse         = nrmse_acc   / denom,
            .elapsed_s     = elapsed_acc / denom,
            .num_layers    = ResolvedLayers(cnn_cfg),
            .conv_channels = cnn_cfg.conv_channels,
            .label         = label
        };
        PrintRow(cnn_cfg, r);
        return r;
    }

    /// Run a batch of configs, printing a comparison table followed by the
    /// cached Ridge baselines (raw and translated) for context.  Each
    /// trial prints a "starting" marker before running and flushes its
    /// result row on completion, so long sweeps show progress live.
    std::vector<Result>
    RunSweep(const std::vector<std::pair<std::string, CNNReadoutConfig>>& trials)
    {
        PrintTableHeader();
        std::vector<Result> results;
        results.reserve(trials.size());
        for (size_t i = 0; i < trials.size(); ++i)
        {
            const auto& [label, cfg] = trials[i];
            std::cout << "  [" << FormatNow() << "] starting " << (i + 1)
                      << "/" << trials.size() << ": " << label << std::endl;
            results.push_back(RunOne(cfg, label));
        }
        std::cout << "  " << std::string(79, '-') << "\n";
        std::cout << "  Ridge raw         NRMSE: "
                  << std::fixed << std::setprecision(6) << RidgeBaseline() << "\n";
        std::cout << "  Ridge translated  NRMSE: "
                  << std::fixed << std::setprecision(6) << TranslationBaseline() << std::endl;
        return results;
    }

private:
    size_t          num_seeds_;
    size_t          num_cnn_seeds_;
    size_t          horizon_;
    ReservoirConfig rc_;
    double          ridge_cached_       = -1.0;
    double          translation_cached_ = -1.0;

    /// Shared Ridge runner used by both baselines; the only difference is
    /// the FeatureMode.  Same reservoir seeds, same train/test split.
    double RunRidge(FeatureMode mode)
    {
        const size_t tr = static_cast<size_t>(collect_ * 0.7);
        const size_t te = collect_ - tr;
        double acc = 0.0;

        for (uint64_t seed : Seeds())
        {
            auto [series, targets] = GenerateSeries();
            ReservoirConfig cfg = rc_;
            cfg.seed = seed;
            cfg.output_fraction = 1.0f;

            ESN<DIM> esn(cfg, ReadoutType::Ridge, mode);
            esn.Warmup(series.data(), warmup_);
            esn.Run(series.data() + warmup_, collect_);
            esn.Train(targets.data(), tr);
            acc += esn.NRMSE(targets.data(), tr, te);
        }
        return acc / static_cast<double>(num_seeds_);
    }

    /// Seed list — first entry is the MG-specific surveyed seed from the
    /// preset table, then generic seeds for stability averaging when
    /// num_seeds_ > 1.  Averaging past num_seeds_=1 mixes in non-surveyed
    /// reservoirs and will no longer match MackeyGlass.md baselines.
    std::vector<uint64_t> Seeds() const
    {
        const uint64_t kSeeds[] = {
            hcnn_presets::MackeyGlass<DIM>().reservoir.seed,
            1337ULL,
            6437149480297576047ULL,
            0xC0FFEE1234567890ULL,
            0xDEADBEEFCAFEBABEULL,
            0x0123456789ABCDEFULL,
            0xFEDCBA9876543210ULL,
            0xA5A5A5A5A5A5A5A5ULL
        };
        constexpr size_t kMax = sizeof(kSeeds) / sizeof(kSeeds[0]);
        const size_t n = std::min(num_seeds_, kMax);
        return std::vector<uint64_t>(kSeeds, kSeeds + n);
    }

    std::pair<std::vector<float>, std::vector<float>> GenerateSeries() const
    {
        auto series = GenerateMackeyGlass(warmup_ + collect_ + horizon_ + 20);
        Normalize(series);
        std::vector<float> targets(collect_);
        for (size_t t = 0; t < collect_; ++t)
            targets[t] = series[warmup_ + t + horizon_];
        return {std::move(series), std::move(targets)};
    }

    /// Mirror of CNNReadout::build_architecture() layer-count logic so the
    /// printed topology matches what HCNN actually builds.
    static int ResolvedLayers(const CNNReadoutConfig& c)
    {
        const int d = static_cast<int>(DIM);
        int layers = (c.num_layers > 0) ? c.num_layers : std::min(d - 3, 4);
        return std::max(layers, 1);
    }

    /// Human-readable local wall-clock timestamp for run headers/footers.
    static std::string FormatNow()
    {
        auto now    = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    void PrintTableHeader() const
    {
        std::cout << "\n"
                  << "  label                 | layers | ch | ep  |  bs | lr_max  |    NRMSE |  time(s)\n"
                  << "  ----------------------+--------+----+-----+-----+---------+----------+---------\n";
    }

    void PrintRow(const CNNReadoutConfig& c, const Result& r) const
    {
        std::string label = r.label;
        if (label.size() > 21) label.resize(21);

        std::cout << "  " << std::left << std::setw(21) << label << " | "
                  << std::right << std::setw(6) << r.num_layers << " | "
                  << std::setw(2) << c.conv_channels << " | "
                  << std::setw(3) << c.epochs << " | "
                  << std::setw(3) << c.batch_size << " | "
                  << std::fixed << std::setprecision(5) << std::setw(7) << c.lr_max << " | "
                  << std::setprecision(6) << std::setw(8) << r.nrmse << " | "
                  << std::setprecision(2) << std::setw(7) << r.elapsed_s << std::endl;
    }
};
