/// @file main.cpp
/// @brief HypercubeRC benchmark suite entry point.
///
/// Usage: HypercubeRC [--hcnn]
///   --hcnn  Include HCNN (HypercubeCNN) readout in MG and NARMA-10 benchmarks.

#include "diagnostics/BenchmarkSuite.h"
#include <cstring>

int main(int argc, char* argv[])
{
    bool run_hcnn = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--hcnn") == 0)
            run_hcnn = true;
    }

    BenchmarkSuite::RunAll(1.0f, nullptr, run_hcnn);
    return 0;
}
