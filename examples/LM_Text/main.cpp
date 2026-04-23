#include <iostream>

#include "Config.h"

namespace lm_text {

int RunTrain();
int RunEval();
int RunInfer();

}  // namespace lm_text

int main()
{
    using namespace lm_text;
    switch (config::kMode) {
        case config::Mode::Train: return RunTrain();
        case config::Mode::Eval:  return RunEval();
        case config::Mode::Infer: return RunInfer();
        default:
            std::cerr << "error: unknown config::kMode\n";
            return 1;
    }
}
