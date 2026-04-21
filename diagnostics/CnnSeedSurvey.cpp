// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 David Liptak

/// @file CnnSeedSurvey.cpp
/// @brief CNN-init seed survey driver: holds the reservoir fixed at its
///        NARMA-10 surveyed winner and sweeps HCNN weight-init seeds to
///        measure CNN-init noise and identify seeds that work well across
///        multiple DIMs.
///
/// Motivation: a reservoir-seed survey (diagnostics/SeedSurvey.cpp) finds
/// reservoirs that match the task well, but holds CNN init fixed.  The
/// CNN-init seed also matters — single-trial NRMSE has significant CV
/// across CNN seeds on the surveyed reservoir — so the Gold Standards we
/// freeze may be baking in a lucky (or unlucky) roll of the CNN init.
/// This driver runs the SAME NUM_CNN_SEEDS seeds at every target DIM
/// under the HRCCNNBaseline<DIM> architecture, then reports per-DIM
/// distributions plus per-DIM top-K leaderboard plus cross-DIM overlap
/// and ranking matrices, so seeds that are strong at multiple DIMs can
/// be identified and baked into HCNNPresets for reproducible Golds.
///
/// Task: NARMA-10.  NARMA is the stronger discriminator than Mackey-Glass
/// h=1 because it explicitly couples memory depth (the u(t-9) term) with
/// nonlinear mixing (the y(t)*sum(y(t-i)) term), whereas MG h=1 is nearly
/// a pure nonlinearity test.  A CNN-init seed that works well on NARMA
/// has to handle both axes.
///
/// Config: DIM 5..8, HRCCNNBaseline<DIM>() architecture, 100 seeds
/// [SEED_BASE..SEED_BASE+99] per DIM, TOP_K = 10 tracked for overlap.
/// Edit the constants below and rebuild.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../ESN.h"
#include "../readout/CNNReadout.h"
#include "../readout/HCNNPresets.h"
#include "SignalGenerators.h"

/// Per-trial record for this driver.  Kept file-local so the driver has
/// no dependency on OptimizeHRCCNNForMG (which would drag in Ridge code
/// we are deliberately not running here).
struct CnnSeedResult
{
    unsigned seed;      ///< cnn_cfg.seed value used for this trial.
    double   nrmse;     ///< Test NRMSE for this single training trial.
    double   elapsed_s; ///< Wall-clock HCNN training time for this trial.
};

// =====================================================================
// Configuration — change these and rebuild
// =====================================================================
static constexpr size_t   NUM_CNN_SEEDS = 20;    ///< Seeds per DIM.
static constexpr unsigned SEED_BASE     = 1;     ///< First cnn_cfg.seed value.
static constexpr size_t   TOP_K         = 10;    ///< Top-N seeds tracked per DIM.
// =====================================================================

/// One DIM's complete survey result, flattened to template-free form so
/// the driver can stack DIM 5..8 into a single vector for cross-DIM
/// reporting.
struct DimSurvey
{
    size_t                     dim;
    std::vector<CnnSeedResult> results;  ///< In call order (seeds[0], seeds[1], ...).
};

/// Run a NARMA-10 CNN-init seed survey at a fixed DIM.  Generates the
/// NARMA data once, warms and runs the reservoir once, then sweeps
/// cnn_cfg.seed over `seeds` — only the readout weight init changes
/// between trials.  Reports per-seed streaming output + a sorted summary
/// + stats, and returns the per-seed results for cross-DIM aggregation.
template <size_t DIM>
static DimSurvey SurveyOne(const std::vector<unsigned>& seeds)
{
    constexpr size_t N       = 1ULL << DIM;
    constexpr size_t warmup  = (N < 256) ? 200 : 500;
    constexpr size_t collect = 18 * N;
    const size_t tr = static_cast<size_t>(collect * 0.7);
    const size_t te = collect - tr;

    // NARMA has its own per-DIM surveyed reservoir seed.
    const uint64_t reservoir_seed = hcnn_presets::NARMA10<DIM>().reservoir.seed;

    const CNNReadoutConfig cnn_base = hcnn_presets::HRCCNNBaseline<DIM>();

    std::cout << "\n=== CnnSeedSurvey  DIM=" << DIM
              << "  N=" << N
              << "  task=NARMA-10"
              << "  reservoir_seed=" << reservoir_seed
              << "  cnn_seeds=" << seeds.size()
              << "  config=HRCCNNBaseline"
              << "  nl=" << cnn_base.num_layers
              << "  ch=" << cnn_base.conv_channels
              << "  head=FLAT"
              << "  ep=" << cnn_base.epochs
              << "  bs=" << cnn_base.batch_size
              << "  lr=" << std::fixed << std::setprecision(5) << cnn_base.lr_max
              << " ===\n" << std::flush;

    // Generate NARMA-10 once, scale inputs to [-1, 1] (same convention as
    // diagnostics/NARMA10.h::Run so results are directly comparable with
    // existing NARMA benchmark rows).
    auto [u, y] = GenerateNARMA10(reservoir_seed + 99, warmup + collect);
    std::vector<float> ri(warmup + collect);
    for (size_t t = 0; t < ri.size(); ++t) ri[t] = u[t] * 4.0f - 1.0f;
    std::vector<float> targets(collect);
    for (size_t t = 0; t < collect; ++t) targets[t] = y[warmup + t];

    ReservoirConfig rc;
    rc.seed            = reservoir_seed;
    rc.output_fraction = 1.0f;

    ESN<DIM> esn(rc, ReadoutType::HCNN);
    esn.Warmup(ri.data(), warmup);
    esn.Run(ri.data() + warmup, collect);

    DimSurvey out{DIM, {}};
    out.results.reserve(seeds.size());

    for (unsigned s : seeds) {
        CNNReadoutConfig cnn_copy = cnn_base;
        cnn_copy.seed = s;

        auto t0 = std::chrono::steady_clock::now();
        esn.Train(targets.data(), tr, cnn_copy);
        auto t1 = std::chrono::steady_clock::now();

        const double this_nrmse   = esn.NRMSE(targets.data(), tr, te);
        const double this_elapsed = std::chrono::duration<double>(t1 - t0).count();
        out.results.push_back({s, this_nrmse, this_elapsed});

        std::cout << "    seed " << std::setw(5) << s
                  << "  nrmse " << std::fixed << std::setprecision(6)
                  << this_nrmse
                  << "  time " << std::setprecision(2) << std::setw(6)
                  << this_elapsed << "s\n" << std::flush;
    }

    if (out.results.empty()) {
        std::cout << "  (no results)\n";
        return out;
    }

    // Sort a COPY ascending by NRMSE for the per-DIM summary table.
    std::vector<CnnSeedResult> sorted = out.results;
    std::sort(sorted.begin(), sorted.end(),
              [](const CnnSeedResult& a, const CnnSeedResult& b) {
                  return a.nrmse < b.nrmse;
              });

    double sum = 0.0;
    for (const auto& r : out.results) sum += r.nrmse;
    const double mean = sum / static_cast<double>(out.results.size());
    double ssq = 0.0;
    for (const auto& r : out.results) {
        const double d = r.nrmse - mean;
        ssq += d * d;
    }
    const double std_u = (out.results.size() >= 2)
        ? std::sqrt(ssq / static_cast<double>(out.results.size() - 1))
        : 0.0;
    const double sem = (out.results.size() >= 2)
        ? std_u / std::sqrt(static_cast<double>(out.results.size()))
        : 0.0;
    const double cv = (mean > 0.0) ? (std_u / mean) : 0.0;

    std::cout << "\n  --- DIM " << DIM << " sorted distribution ---\n"
              << "   rank  seed    nrmse       time(s)\n"
              << "  ------+------+----------+---------\n";
    for (size_t i = 0; i < sorted.size(); ++i) {
        std::cout << "  " << std::setw(5) << (i + 1)
                  << "  " << std::setw(5) << sorted[i].seed
                  << "  " << std::fixed << std::setprecision(6)
                  << std::setw(10) << sorted[i].nrmse
                  << "  " << std::setprecision(2) << std::setw(7)
                  << sorted[i].elapsed_s << "\n";
    }

    std::cout << "\n  DIM " << DIM << " summary:\n"
              << "    n         = " << out.results.size() << "\n"
              << "    mean      = " << std::fixed << std::setprecision(6) << mean << "\n"
              << "    std       = " << std_u << "\n"
              << "    sem       = " << sem   << "  (std / sqrt(n))\n"
              << "    cv        = " << std::setprecision(4) << (cv * 100.0) << " %\n"
              << "    min       = " << std::setprecision(6) << sorted.front().nrmse
              << "  (seed " << sorted.front().seed << ")\n"
              << "    max       = " << sorted.back().nrmse
              << "  (seed " << sorted.back().seed << ")\n"
              << "    range     = " << (sorted.back().nrmse - sorted.front().nrmse) << "\n"
              << std::endl;

    return out;
}

/// Sorted ascending by NRMSE — the top-K seeds for a DIM's results.
static std::vector<CnnSeedResult>
TopK(const std::vector<CnnSeedResult>& results, size_t k)
{
    std::vector<CnnSeedResult> sorted = results;
    std::sort(sorted.begin(), sorted.end(),
              [](const CnnSeedResult& a, const CnnSeedResult& b) {
                  return a.nrmse < b.nrmse;
              });
    if (sorted.size() > k) sorted.resize(k);
    return sorted;
}

/// Compact per-DIM top-K leaderboard: one column per DIM, K rows.
/// Quick eyeball view of "who won at this DIM".
static void PrintTopKPerDim(const std::vector<DimSurvey>& surveys, size_t k)
{
    std::cout << "\n"
              << "===========================================================\n"
              << "  Per-DIM top-" << k << " seeds (best NRMSE first)\n"
              << "===========================================================\n"
              << "  rank ";
    for (const auto& s : surveys) {
        std::cout << " | DIM " << std::setw(2) << s.dim
                  << " seed  nrmse    ";
    }
    std::cout << "\n  -----";
    for (size_t i = 0; i < surveys.size(); ++i) {
        std::cout << "-+---------------------";
    }
    std::cout << "\n";

    std::vector<std::vector<CnnSeedResult>> topk_per;
    topk_per.reserve(surveys.size());
    for (const auto& s : surveys) topk_per.push_back(TopK(s.results, k));

    for (size_t r = 0; r < k; ++r) {
        // "  " + setw(4) = 6 chars; append a space to match the 7-char
        // "  rank " / "  -----" header cells above.
        std::cout << "  " << std::setw(4) << (r + 1) << " ";
        for (const auto& col : topk_per) {
            if (r >= col.size()) {
                std::cout << " |       ---            ";
            } else {
                std::cout << " |  " << std::setw(5) << col[r].seed
                          << "  " << std::fixed << std::setprecision(6)
                          << std::setw(10) << col[r].nrmse;
            }
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}

/// Overlap analysis across the per-DIM top-K sets.  Groups seeds by how
/// many DIMs they made the top-K cut at, from "all DIMs" down to "two
/// DIMs".  Seeds that only made the cut at one DIM are intentionally
/// omitted — the whole point is to surface CROSS-DIM winners.
///
/// The per-DIM NRMSE column always shows the seed's actual measurement,
/// even for DIMs where the seed didn't make the top-K cut; those cells
/// are marked with `*` after the value so the reader can see the full
/// cross-DIM picture for any seed that appears in any bucket.
static void PrintTopKOverlap(const std::vector<DimSurvey>& surveys, size_t k)
{
    // Membership: which DIMs' top-K does each seed belong to.
    std::map<unsigned, std::set<size_t>> membership;
    // Full per-seed NRMSE across ALL DIMs (not just the top-K hits), so
    // the overlap table can show the full cross-DIM picture for any
    // seed that appears in any bucket.
    std::map<unsigned, std::map<size_t, double>> full_nrmse;

    for (const auto& s : surveys) {
        for (const auto& r : s.results) {
            full_nrmse[r.seed][s.dim] = r.nrmse;
        }
        auto top = TopK(s.results, k);
        for (const auto& r : top) {
            membership[r.seed].insert(s.dim);
        }
    }

    std::map<size_t, std::vector<unsigned>, std::greater<size_t>> buckets;
    for (const auto& [seed, dims] : membership) {
        if (dims.size() >= 2) buckets[dims.size()].push_back(seed);
    }

    std::cout << "\n"
              << "===========================================================\n"
              << "  Top-" << k << " overlap across DIMs (seeds in >= 2 DIM top-" << k << " sets)\n"
              << "  `*` after a cell = seed was NOT in that DIM's top-" << k << "\n"
              << "===========================================================\n";
    if (buckets.empty()) {
        std::cout << "  (no seed made the top-" << k << " cut at more than one DIM)\n"
                  << std::flush;
        return;
    }

    for (const auto& [count, seeds] : buckets) {
        std::cout << "\n  seeds in top-" << k << " at " << count
                  << " DIM" << (count == 1 ? "" : "s") << ":\n";
        std::cout << "    seed ";
        for (const auto& s : surveys) {
            std::cout << " |  DIM " << std::setw(2) << s.dim << "    ";
        }
        std::cout << "\n    -----";
        for (size_t i = 0; i < surveys.size(); ++i)
            std::cout << "-+------------";
        std::cout << "\n";

        // Sort within bucket by mean NRMSE across the DIMs the seed was
        // tested at (with this driver, that's every DIM — but the code
        // stays correct if someone later passes heterogeneous seed sets
        // per DIM).
        std::vector<unsigned> sorted_seeds = seeds;
        std::sort(sorted_seeds.begin(), sorted_seeds.end(),
                  [&](unsigned a, unsigned b) {
                      double sum_a = 0.0, sum_b = 0.0;
                      for (const auto& [d, v] : full_nrmse[a]) sum_a += v;
                      for (const auto& [d, v] : full_nrmse[b]) sum_b += v;
                      return (sum_a / full_nrmse[a].size()) <
                             (sum_b / full_nrmse[b].size());
                  });

        for (unsigned seed : sorted_seeds) {
            std::cout << "    " << std::setw(4) << seed;
            for (const auto& s : surveys) {
                auto it = full_nrmse[seed].find(s.dim);
                if (it == full_nrmse[seed].end()) {
                    std::cout << " |   ---      ";
                } else {
                    const bool in_topk = membership[seed].count(s.dim) > 0;
                    std::cout << " | " << std::fixed << std::setprecision(6)
                              << it->second << (in_topk ? "  " : " *");
                }
            }
            std::cout << "\n";
        }
    }
    std::cout << std::flush;
}

/// Cross-DIM report: for every seed tested, compute its rank at each DIM
/// and its mean rank across DIMs, then print seeds sorted by mean rank.
/// Lower mean rank = better "works across the board".  Rank 1 = best
/// within a given DIM's distribution.
static void PrintCrossDimSummary(const std::vector<DimSurvey>& surveys)
{
    if (surveys.empty()) return;

    std::map<unsigned, std::map<size_t, std::pair<double, size_t>>> matrix;

    for (const auto& s : surveys) {
        std::vector<size_t> idx(s.results.size());
        std::iota(idx.begin(), idx.end(), size_t{0});
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) {
                      return s.results[a].nrmse < s.results[b].nrmse;
                  });
        for (size_t r = 0; r < idx.size(); ++r) {
            const auto& rec = s.results[idx[r]];
            matrix[rec.seed][s.dim] = {rec.nrmse, r + 1};
        }
    }

    struct Row {
        unsigned seed;
        double   mean_rank;
        double   mean_nrmse;
        size_t   coverage;
    };
    std::vector<Row> rows;
    rows.reserve(matrix.size());
    for (const auto& [seed, cells] : matrix) {
        double rank_sum = 0.0;
        double nrmse_sum = 0.0;
        for (const auto& [dim, cell] : cells) {
            (void)dim;
            rank_sum  += static_cast<double>(cell.second);
            nrmse_sum += cell.first;
        }
        const size_t cov = cells.size();
        rows.push_back({seed,
                        rank_sum  / static_cast<double>(cov),
                        nrmse_sum / static_cast<double>(cov),
                        cov});
    }

    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) {
                  if (a.coverage != b.coverage) return a.coverage > b.coverage;
                  return a.mean_rank < b.mean_rank;
              });

    std::cout << "\n"
              << "===========================================================\n"
              << "  Cross-DIM CNN-seed ranking (lower mean rank = better across DIMs)\n"
              << "===========================================================\n";
    std::cout << "  seed ";
    for (const auto& s : surveys) {
        std::cout << " | rank@" << std::setw(2) << s.dim
                  << "  nrmse@" << std::setw(2) << s.dim;
    }
    std::cout << " | mean_rank | mean_nrmse\n";
    std::cout << "  -----";
    for (size_t i = 0; i < surveys.size(); ++i) {
        std::cout << "-+------------------------";
    }
    std::cout << "-+-----------+-----------\n";

    for (const auto& row : rows) {
        std::cout << "  " << std::setw(4) << row.seed;
        for (const auto& s : surveys) {
            auto it = matrix[row.seed].find(s.dim);
            if (it == matrix[row.seed].end()) {
                std::cout << " |   ---      ---     ";
            } else {
                std::cout << " | " << std::setw(5) << it->second.second
                          << "   " << std::fixed << std::setprecision(6)
                          << it->second.first;
            }
        }
        std::cout << " | " << std::fixed << std::setprecision(2)
                  << std::setw(9) << row.mean_rank
                  << " | " << std::setprecision(6)
                  << std::setw(10) << row.mean_nrmse << "\n";
    }
    std::cout << std::flush;
}

int main()
{
    std::vector<unsigned> seeds(NUM_CNN_SEEDS);
    std::iota(seeds.begin(), seeds.end(), SEED_BASE);

    std::vector<DimSurvey> all;
    /*all.reserve(4);
    all.push_back(SurveyOne<5>(seeds));
    all.push_back(SurveyOne<6>(seeds));
    all.push_back(SurveyOne<7>(seeds));
    all.push_back(SurveyOne<8>(seeds));*/

    all.reserve(1);
    all.push_back(SurveyOne<10>(seeds));

    PrintTopKPerDim(all, TOP_K);
    PrintTopKOverlap(all, TOP_K);
    PrintCrossDimSummary(all);
    return 0;
}
