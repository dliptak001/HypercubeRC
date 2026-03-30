/// @file Tools/StandaloneESNSweep.cpp
/// @brief Grid sweep: SR x input_scaling for standalone ESN.
///
/// Configure DIM, USE_TRANSLATION, and sweep ranges below, rebuild, and run.
/// 3-seed average. Runs MG h=1, NARMA-10, and MC.

#include <iomanip>
#include <iostream>
#include <vector>
#include "../diagnostics/MemoryCapacity.h"
#include "../diagnostics/MackeyGlass.h"
#include "../diagnostics/NARMA10.h"

// =====================================================================
// Configuration — change these and rebuild
// =====================================================================
static constexpr size_t DIM = 8;
static constexpr ReadoutType READOUT = ReadoutType::Linear;

// Sweep grid — adjust per experiment
static const std::vector<float> SR_VALUES  = {0.93f, 0.94f, 0.95f, 0.96f, 0.97f};
static const std::vector<float> INP_VALUES = {0.02f, 0.04f, 0.06f, 0.10f, 0.15f};

// =====================================================================
// Main — grid sweep with formatted output
// =====================================================================
int main()
{
    size_t total = SR_VALUES.size() * INP_VALUES.size();
    const char* rn = (READOUT == ReadoutType::Ridge) ? "Ridge" : "Linear";

    std::cout << "=== Standalone ESN Sweep: DIM=" << DIM << " N=" << (1ULL << DIM)
              << " " << rn
              << " (" << total << " configs, 3-seed avg) ===\n\n";

    std::cout << "    SR |  inp |    MG h1 | NARMA-10 |    MC\n";
    std::cout << "  -----+------+----------+----------+------\n" << std::flush;

    for (float sr : SR_VALUES)
    {
        for (float inp : INP_VALUES)
        {
            ReservoirConfig cfg;
            cfg.spectral_radius = sr;
            cfg.block_scaling = {inp};

            MackeyGlass<DIM> mg(1, READOUT, &cfg);
            NARMA10<DIM> narma(READOUT, &cfg);
            MemoryCapacity<DIM> mc(50, &cfg);

            double mg_nrmse = mg.Run().nrmse_raw;
            double narma_nrmse = narma.Run().nrmse_raw;
            double mc_total = mc.Run().mc_total;

            std::cout << std::fixed << std::setprecision(2)
                      << "  " << std::setw(4) << sr
                      << " | " << std::setw(4) << inp
                      << " | " << std::setprecision(5) << std::setw(8) << mg_nrmse
                      << " | " << std::setprecision(4) << std::setw(8) << narma_nrmse
                      << " | " << std::setprecision(2) << std::setw(5) << mc_total
                      << "\n" << std::flush;
        }
    }

    return 0;
}
