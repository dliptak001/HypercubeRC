#pragma once

#include <string>

namespace hrccnn_llm_math {

/// Canonical number format: at most 2 decimal places (half-away-from-zero
/// rounding), trailing zeros dropped, -0 and +0 both collapse to "0".
///
/// Examples: 5.00 -> "5", 5.50 -> "5.5", 5.25 -> "5.25", -0.00 -> "0".
///
/// Applied to both LHS operands and RHS results during training-dataset
/// construction. See docs/HRCCNN_LLM_Math.md § Number Format.
std::string Canonicalize(double v);

/// True when `v` canonicalizes to "0" — i.e., rounds to zero at 2 decimal
/// places. Used by the generator and verifier to reject `/0` divisors
/// under the canonical-form rounding policy.
inline bool IsCanonicalZero(double v) { return v > -0.005 && v < 0.005; }

}  // namespace hrccnn_llm_math
