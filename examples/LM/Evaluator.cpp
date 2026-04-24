#include "Evaluator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace lm {

Evaluator::Evaluator(std::size_t num_classes)
    : num_classes_(num_classes),
      class_correct_(num_classes, 0),
      class_total_(num_classes, 0)
{
}

void Evaluator::Record(const float* logits, int true_label)
{
    // --- Softmax log-probability for BPC ---
    float max_logit = logits[0];
    for (std::size_t k = 1; k < num_classes_; ++k)
        if (logits[k] > max_logit) max_logit = logits[k];

    double sum_exp = 0.0;
    for (std::size_t k = 0; k < num_classes_; ++k)
        sum_exp += std::exp(static_cast<double>(logits[k]) - max_logit);

    double log_prob = (logits[true_label] - max_logit) - std::log(sum_exp);
    log_loss_ -= log_prob;

    // --- Top-k accuracy via partial sort ---
    std::vector<std::size_t> indices(num_classes_);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    std::size_t k_max = std::min<std::size_t>(5, num_classes_);
    std::partial_sort(indices.begin(),
                      indices.begin() + static_cast<long>(k_max),
                      indices.end(),
                      [&](std::size_t a, std::size_t b) {
                          return logits[a] > logits[b];
                      });

    for (std::size_t k = 0; k < k_max; ++k) {
        if (static_cast<int>(indices[k]) == true_label) {
            if (k < 1) ++correct1_;
            if (k < 3) ++correct3_;
            if (k < 5) ++correct5_;
            break;
        }
    }

    // --- Per-class tracking ---
    auto label = static_cast<std::size_t>(true_label);
    if (label < num_classes_) {
        class_total_[label]++;
        if (static_cast<int>(indices[0]) == true_label)
            class_correct_[label]++;
    }

    ++count_;
}

Metrics Evaluator::Compute() const
{
    if (count_ == 0) return {};

    Metrics m;
    m.count = count_;
    m.top1  = static_cast<double>(correct1_) / count_;
    m.top3  = static_cast<double>(correct3_) / count_;
    m.top5  = static_cast<double>(correct5_) / count_;
    m.bpc   = log_loss_ / (count_ * std::log(2.0));
    return m;
}

std::vector<ClassAccuracy> Evaluator::WorstClasses(std::size_t n) const
{
    // Collect classes that have at least one sample.
    std::vector<ClassAccuracy> entries;
    for (std::size_t i = 0; i < num_classes_; ++i) {
        if (class_total_[i] == 0) continue;
        double acc = static_cast<double>(class_correct_[i]) / class_total_[i];
        entries.push_back({static_cast<int>(i), acc, class_correct_[i], class_total_[i]});
    }

    // Sort by accuracy ascending (worst first).
    std::sort(entries.begin(), entries.end(),
              [](const ClassAccuracy& a, const ClassAccuracy& b) {
                  return a.accuracy < b.accuracy;
              });

    if (entries.size() > n)
        entries.resize(n);
    return entries;
}

void Evaluator::Reset()
{
    count_ = correct1_ = correct3_ = correct5_ = 0;
    log_loss_ = 0.0;
    std::fill(class_correct_.begin(), class_correct_.end(), 0);
    std::fill(class_total_.begin(), class_total_.end(), 0);
}

std::string Evaluator::FormatMetrics(const std::string& tag, const Metrics& m)
{
    std::ostringstream os;
    os << "[" << tag << "] "
       << "top1=" << m.top1
       << " top3=" << m.top3
       << " top5=" << m.top5
       << " bpc=" << m.bpc;
    return os.str();
}

std::string Evaluator::FormatWorstClasses(
    const std::string& tag,
    const std::vector<ClassAccuracy>& worst,
    const Vocabulary& vocab)
{
    std::ostringstream os;
    os << "[" << tag << "] worst " << worst.size() << " classes:";
    for (const auto& ca : worst) {
        char ch = vocab.ClassToChar(ca.class_index);
        std::string repr;
        if      (ch == '\n') repr = "\\n";
        else if (ch == '\r') repr = "\\r";
        else if (ch == '\t') repr = "\\t";
        else if (ch == ' ')  repr = "SP";
        else { repr = "'"; repr += ch; repr += "'"; }

        os << " " << repr << "="
           << static_cast<int>(ca.accuracy * 100) << "%"
           << "(" << ca.correct << "/" << ca.total << ")";
    }
    return os.str();
}

}  // namespace lm
