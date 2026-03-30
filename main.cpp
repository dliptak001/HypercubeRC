/// @file main.cpp
/// @brief HypercubeRC benchmark suite entry point.

#include "diagnostics/BenchmarkSuite.h"

int main()
{
    // Optional: override reservoir parameters for all benchmarks.
    // ReservoirConfig cfg;
    // cfg.leak_rate = 0.3f;
    // BenchmarkSuite::RunAll(&cfg);

    BenchmarkSuite::RunAll();
    return 0;
}
