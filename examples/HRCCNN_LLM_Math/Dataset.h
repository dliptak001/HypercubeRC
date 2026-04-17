#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ESN.h"

namespace hrccnn_llm_math {

/// Split a generated line into its LHS and full-expression strings.
///   line = "5 + 3 = 8#"
///   lhs  = "5 + 3"          (everything before " = ")
///   rhs  = "8"              (text between " = " and the trailing '#')
/// Returns false on malformed input (missing " = " or trailing '#').
struct LineSplit
{
    std::string lhs;    ///< LHS of the equation, no surrounding spaces.
    std::string rhs;    ///< RHS canonical number, no trailing '#'.
    std::string full;   ///< Full line verbatim, including "#".
};
bool SplitLine(const std::string& line, LineSplit& out);

/// Teacher-forced collection for one expression. Reset the reservoir, prime
/// 2x over the LHS, then Run the full expression for (L - 1) steps so that
/// the state buffer gains (L - 1) snapshots — one per training position,
/// each paired with a target == next character's class index.
///
/// Appends the class-index targets to `targets_out`. States are appended
/// into the ESN's internal buffer as usual.
template <size_t DIM>
void TeacherForceOne(ESN<DIM>& esn,
                     const LineSplit& split,
                     std::vector<float>& targets_out);

/// Reset the reservoir and prime it by running 2x over the LHS (no state
/// collection, no readout). Used by both training (before the collecting
/// Run) and inference (before the autoregressive argmax loop, which does an
/// additional final LHS pass).
template <size_t DIM>
void ResetAndPrime(ESN<DIM>& esn, const std::string& lhs);

/// Autoregressively generate an RHS string for the given LHS. Expects the
/// ESN to be HCNN with num_outputs=20 (classification head). Runs the
/// inference protocol: reset + 2x prime + final LHS pass + argmax loop
/// until '#' is emitted or `max_output_chars` is hit (the hard length cap
/// that prevents runaway generation when '#' is never predicted).
///
/// The returned string includes the trailing '#' iff the model emitted
/// it. On length-cap termination, no '#' is appended — callers can detect
/// that to classify the failure as a non-stop.
template <size_t DIM>
std::string GenerateRHS(ESN<DIM>& esn,
                        const std::string& lhs,
                        std::size_t max_output_chars = 16);

}  // namespace hrccnn_llm_math
