/// @file main.cpp
/// @brief HypercubeRC benchmark suite entry point.

#include "diagnostics/BenchmarkSuite.h"

int main()
{
    BenchmarkSuite::RunAll(BenchmarkMode::Both);
    return 0;
}
