#include <iostream>

#include "Config.h"

namespace hrccnn_lm_text {

int RunTrain();
int RunEval();
int RunInfer();

}  // namespace hrccnn_lm_text

int main()
{
    using namespace hrccnn_lm_text;
    switch (config::kMode) {
        case config::Mode::Train: return RunTrain();
        case config::Mode::Eval:  return RunEval();
        case config::Mode::Infer: return RunInfer();
        default:
            std::cerr << "error: unknown config::kMode\n";
            return 1;
    }
}
