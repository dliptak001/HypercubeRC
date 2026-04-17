#pragma once

#include <string>

namespace hrccnn_llm_math {

/// Independent shunting-yard verifier for a generated line like
/// "(1.5 + 2.5) * 3 = 12#".
///
/// Splits the line on " = ", parses the LHS via tokenize + shunting-yard,
/// evaluates the RPN, canonicalizes the result, and compares to the
/// printed RHS. Intentionally independent of the generator's evaluation
/// path so that bugs in the generator's arithmetic are caught here.
///
/// On failure, `reason` (if non-null) is populated with a short diagnostic.
bool VerifyLine(const std::string& line, std::string* reason = nullptr);

}  // namespace hrccnn_llm_math
