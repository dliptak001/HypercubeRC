#include <iostream>

#include "Config.h"

namespace hrccnn_llm_math {

int RunGenerate();
int RunTrain();
int RunEval();
int RunInfer();

}  // namespace hrccnn_llm_math

int main()
{
    using namespace hrccnn_llm_math;
    switch (config::kMode) {
        case config::Mode::Generate: return RunGenerate();
        case config::Mode::Train:    return RunTrain();
        case config::Mode::Eval:     return RunEval();
        case config::Mode::Infer:    return RunInfer();
        default:
            std::cerr << "error: unknown config::kMode\n";
            return 1;
    }
}
