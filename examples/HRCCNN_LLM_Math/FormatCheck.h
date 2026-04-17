#pragma once

#include <regex>
#include <string>

namespace hrccnn_llm_math {

/// True iff `emitted` is a canonical-form RHS followed by '#':
/// optional sign, 1-9 integer digits, optional .1-2 frac digits (no trailing
/// zeros in fractions, no trailing '.'), then '#'.
inline bool IsValidFormat(const std::string& emitted)
{
    static const std::regex re(R"(^-?(0|[1-9]\d{0,8})(\.\d|\.\d[1-9])?#$)");
    return std::regex_match(emitted, re);
}

}  // namespace hrccnn_llm_math
