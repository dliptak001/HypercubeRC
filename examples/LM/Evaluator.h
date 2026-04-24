#pragma once

/// @file Evaluator.h
/// @brief Accumulates character-level prediction metrics.
///
/// The Evaluator collects top-k accuracy, bits-per-character (BPC), and
/// per-class accuracy from a stream of (logits, true_label) pairs.  It is
/// used identically during mid-training evaluation and standalone eval,
/// eliminating the code duplication that plagues procedural designs.
///
/// Usage:
///   Evaluator eval(num_classes);
///   for each timestep:
///       model.Predict(logits);
///       eval.Record(logits, true_label);
///   auto metrics = eval.Compute();

#include <cstddef>
#include <string>
#include <vector>

#include "Vocabulary.h"

namespace lm {

/// Computed evaluation metrics.
struct Metrics
{
    double top1        = 0.0;   ///< Fraction of correct top-1 predictions.
    double top3        = 0.0;   ///< Fraction where true label is in top 3.
    double top5        = 0.0;   ///< Fraction where true label is in top 5.
    double bpc         = 0.0;   ///< Bits per character (cross-entropy in bits).
    std::size_t count  = 0;     ///< Number of samples accumulated.
};

/// Per-class accuracy entry, used for worst-class reporting.
struct ClassAccuracy
{
    int         class_index;
    double      accuracy;
    std::size_t correct;
    std::size_t total;
};

class Evaluator
{
public:
    /// @param num_classes  Number of output classes (typically Vocabulary::kSize).
    explicit Evaluator(std::size_t num_classes);

    /// Record one prediction.
    /// @param logits       Raw network output (num_classes floats).
    /// @param true_label   Ground-truth class index.
    void Record(const float* logits, int true_label);

    /// Compute aggregate metrics from all recorded predictions.
    [[nodiscard]] Metrics Compute() const;

    /// Return the N classes with the lowest accuracy (most confused).
    /// Classes with zero samples are excluded.
    [[nodiscard]] std::vector<ClassAccuracy> WorstClasses(std::size_t n) const;

    /// Clear all accumulated state.
    void Reset();

    /// Format metrics as a single log line.
    [[nodiscard]] static std::string FormatMetrics(const std::string& tag,
                                                   const Metrics& m);

    /// Format worst-class breakdown as a log line.
    [[nodiscard]] static std::string FormatWorstClasses(
        const std::string& tag,
        const std::vector<ClassAccuracy>& worst,
        const Vocabulary& vocab);

private:
    std::size_t              num_classes_;
    std::size_t              count_     = 0;
    std::size_t              correct1_  = 0;
    std::size_t              correct3_  = 0;
    std::size_t              correct5_  = 0;
    double                   log_loss_  = 0.0;
    std::vector<std::size_t> class_correct_;
    std::vector<std::size_t> class_total_;
};

}  // namespace lm
