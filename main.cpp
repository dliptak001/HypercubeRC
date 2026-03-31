/// @file main.cpp
/// @brief HypercubeRC benchmark suite entry point.

#include "diagnostics/BenchmarkSuite.h"

int main()
{
    // Optional: set output fraction (default 1.0 = all vertices).
    // BenchmarkSuite::RunAll(0.5f);           // 50% of vertices as readout features

    // Optional: override reservoir parameters for all benchmarks.
    // SR and input_scaling defaults are scale-invariant — usually no need to change them.
    // ReservoirConfig cfg;
    // cfg.leak_rate = 0.3f;
    // BenchmarkSuite::RunAll(1.0f, &cfg);

    BenchmarkSuite::RunAll();
    return 0;
}
