#include "NumberFormat.h"

#include <cmath>
#include <cstdio>

namespace hrccnn_llm_math {

std::string Canonicalize(double v)
{
    // 1. Round to 2 decimals, half-away-from-zero (std::round's behavior).
    double r = std::round(v * 100.0) / 100.0;

    // 2. Collapse -0 and 0 to "0".
    if (r == 0.0) return "0";

    const char* sign   = (r < 0.0) ? "-" : "";
    double      r_abs  = std::fabs(r);
    long long   i_part = static_cast<long long>(std::floor(r_abs));
    int         cents  = static_cast<int>(std::round((r_abs - static_cast<double>(i_part)) * 100.0));

    // Rounding can push cents to 100: normalize.
    if (cents == 100) { i_part += 1; cents = 0; }

    char buf[32];
    if (cents == 0) {
        std::snprintf(buf, sizeof(buf), "%s%lld", sign, i_part);
    } else if (cents % 10 == 0) {
        std::snprintf(buf, sizeof(buf), "%s%lld.%d", sign, i_part, cents / 10);
    } else {
        std::snprintf(buf, sizeof(buf), "%s%lld.%02d", sign, i_part, cents);
    }
    return std::string(buf);
}

}  // namespace hrccnn_llm_math
