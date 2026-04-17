#include <cstdint>
#include <iostream>
#include <random>
#include <string>

#include "Config.h"
#include "Generator.h"
#include "Verifier.h"

namespace hrccnn_llm_math {

int RunGenerate()
{
    const config::GenerateCfg& args = config::kGenerate;

    std::uint64_t seed = args.seed;
    if (!args.use_fixed_seed) {
        std::random_device rd;
        seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }

    GeneratorConfig cfg;
    cfg.rhs_filter_999 = args.rhs_filter_999;
    Generator gen(seed, cfg);

    std::size_t failed = 0;
    const std::size_t max_print_failures = 10;
    for (std::size_t i = 0; i < args.samples; ++i) {
        std::string line = gen.Sample();
        if (!args.quiet) std::cout << line << "\n";
        if (args.verify) {
            std::string reason;
            if (!VerifyLine(line, &reason)) {
                ++failed;
                if (failed <= max_print_failures) {
                    std::cerr << "VERIFY FAIL: " << line
                              << "  (" << reason << ")\n";
                }
            }
        }
    }

    std::cerr << "samples=" << args.samples
              << " seed=" << seed
              << " filter=" << (args.rhs_filter_999 ? "|RHS|<=999" : "none")
              << " verify=" << (args.verify ? "on" : "off")
              << " failed=" << failed << "\n";
    return (failed == 0) ? 0 : 3;
}

}  // namespace hrccnn_llm_math
