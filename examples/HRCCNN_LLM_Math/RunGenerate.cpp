#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

#include "Generator.h"
#include "Verifier.h"

namespace hrccnn_llm_math {

namespace {

struct GenerateArgs
{
    std::size_t   samples = 1000;
    std::uint64_t seed    = 0;
    bool          seed_set = false;
    bool          rhs_filter_999 = true;
    bool          verify  = true;
    bool          quiet   = false;  // suppress per-line printing
};

bool ParseArgs(int argc, char** argv, GenerateArgs& out)
{
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "error: " << name << " requires a value\n";
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--samples") {
            const char* v = next("--samples");
            if (!v) return false;
            out.samples = static_cast<std::size_t>(std::strtoull(v, nullptr, 10));
        } else if (a == "--seed") {
            const char* v = next("--seed");
            if (!v) return false;
            out.seed = std::strtoull(v, nullptr, 10);
            out.seed_set = true;
        } else if (a == "--no-filter") {
            out.rhs_filter_999 = false;
        } else if (a == "--no-verify") {
            out.verify = false;
        } else if (a == "--quiet") {
            out.quiet = true;
        } else {
            std::cerr << "error: unknown flag '" << a << "'\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int RunGenerate(int argc, char** argv)
{
    GenerateArgs args;
    if (!ParseArgs(argc, argv, args)) return 1;

    if (!args.seed_set) {
        std::random_device rd;
        args.seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }

    GeneratorConfig cfg;
    cfg.rhs_filter_999 = args.rhs_filter_999;
    Generator gen(args.seed, cfg);

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
              << " seed=" << args.seed
              << " filter=" << (args.rhs_filter_999 ? "|RHS|<=999" : "none")
              << " verify=" << (args.verify ? "on" : "off")
              << " failed=" << failed << "\n";
    return (failed == 0) ? 0 : 3;
}

}  // namespace hrccnn_llm_math
