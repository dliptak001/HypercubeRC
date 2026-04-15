/// @file main.cpp
/// @brief HypercubeRC benchmark suite entry point.
///
/// Usage: HypercubeRC [--hcnn | --hcnn-only]
///   --hcnn       Include HCNN readout in the canonical benchmark suite
///                (MC + MG + NARMA, Ridge raw / Ridge translated / HCNN).
///                HCNN uses the HRCCNN baseline config from
///                `docs/HRCCNNBaselineConfig.md`.
///   --hcnn-only  Run the HCNN-only variant suite instead: MG and NARMA
///                at DIM 5-8 with HCNN only (no MC, no Ridge comparison).
///                MG uses the Jaeger & Haas 2004 horizon h=84.

#include "diagnostics/BenchmarkSuite.h"
#include "diagnostics/HCNNBenchmarkSuite.h"
#include <cstring>

int main(int argc, char* argv[])
{
    bool run_hcnn = false;
    bool hcnn_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--hcnn") == 0)
            run_hcnn = true;
        else if (std::strcmp(argv[i], "--hcnn-only") == 0)
            hcnn_only = true;
    }

    if (hcnn_only)
        HCNNBenchmarkSuite::RunAll();
    else
        BenchmarkSuite::RunAll(1.0f, nullptr, run_hcnn);
    return 0;
}
