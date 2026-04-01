/// @file diagnostics/StandaloneESNSweep.cpp
/// @brief Grid sweep: SR x input_scaling for standalone ESN.
///
/// Configure DIM, sweep ranges, and output_fraction below, rebuild, and run.
/// 3-seed average. Ridge readout. Reports both raw and translation NRMSE
/// for MG h=1 and NARMA-10, plus MC (raw features).

#include <iomanip>
#include <iostream>
#include <vector>
#include "MemoryCapacity.h"
#include "MackeyGlass.h"
#include "NARMA10.h"

// =====================================================================
// Configuration — change these and rebuild
// =====================================================================
static constexpr size_t DIM = 7;
static constexpr float OUTPUT_FRACTION = 1.0f;  // 1.0 = all vertices, 0.5 = half

// Sweep grid — adjust per experiment
static const std::vector<float> SR_VALUES  = {0.88f, 0.89f, 0.90f, 0.91f, 0.92f};
static const std::vector<float> INP_VALUES = {0.02f, 0.04f, 0.06f, 0.08f, 0.10f};

// =====================================================================
// Main — grid sweep with formatted output
// =====================================================================
int main()
{
    size_t total = SR_VALUES.size() * INP_VALUES.size();

    std::cout << "=== Standalone ESN Sweep: DIM=" << DIM << " N=" << (1ULL << DIM)
              << " Ridge " << static_cast<int>(OUTPUT_FRACTION * 100) << "% output"
              << " (" << total << " configs, 3-seed avg) ===\n\n";

    std::cout << "    SR |  inp | MG raw   | MG trans | NAR raw  | NAR trans |    MC\n";
    std::cout << "  -----+------+----------+----------+----------+-----------+------\n" << std::flush;

    for (float sr : SR_VALUES)
    {
        for (float inp : INP_VALUES)
        {
            ReservoirConfig cfg;
            cfg.spectral_radius = sr;
            cfg.input_scaling = inp;
            cfg.output_fraction = OUTPUT_FRACTION;

            MackeyGlass<DIM> mg(1, ReadoutType::Ridge, &cfg);
            NARMA10<DIM> narma(ReadoutType::Ridge, &cfg);
            MemoryCapacity<DIM> mc(50, &cfg);

            auto mg_r = mg.Run();
            auto narma_r = narma.Run();
            double mc_total = mc.Run().mc_total;

            std::cout << std::fixed << std::setprecision(2)
                      << "  " << std::setw(4) << sr
                      << " | " << std::setw(4) << inp
                      << " | " << std::setprecision(5) << std::setw(8) << mg_r.nrmse_raw
                      << " | " << std::setw(8) << mg_r.nrmse_full
                      << " | " << std::setprecision(4) << std::setw(8) << narma_r.nrmse_raw
                      << " | " << std::setw(9) << narma_r.nrmse_full
                      << " | " << std::setprecision(2) << std::setw(5) << mc_total
                      << "\n" << std::flush;
        }
    }

    return 0;
}
