#include <iostream>
#include <string>

namespace hrccnn_llm_math {

int RunGenerate(int argc, char** argv);
int RunTrain(int argc, char** argv);
int RunEval(int argc, char** argv);
int RunInfer(int argc, char** argv);

}  // namespace hrccnn_llm_math

namespace {

void PrintUsage(const char* prog)
{
    std::cerr <<
        "HRCCNN_LLM_Math — character-level math LM on HRCCNN\n"
        "\n"
        "Usage:\n"
        "  " << prog << " generate [--samples N] [--seed S] [--no-filter] [--no-verify] [--quiet]\n"
        "      Sample expression lines from the grammar, print them, and run\n"
        "      the independent shunting-yard verifier over each. Default N=1000.\n"
        "\n"
        "  " << prog << " train --output <model.bin> [--samples N] [--val-samples V]\n"
        "                      [--seed S] [--reservoir-seed R] [--epochs E] [--batch-size B]\n"
        "                      [--output-fraction F] [--autoreg-samples A] [--git-sha SHA]\n"
        "                      [--no-filter] [--verbose]\n"
        "      Train a DIM 12 HRCCNN classifier on generated expressions; save to <model.bin>.\n"
        "      Defaults: samples=5000, val-samples=2000, epochs=1000, batch-size=4096,\n"
        "                output-fraction=0.125, autoreg-samples=64.\n"
        "\n"
        "  " << prog << " eval --model <model.bin> [--samples N] [--seed S] [--no-char]\n"
        "      Score a saved model on a freshly-generated held-out set.\n"
        "      Reports teacher-forced char accuracy + autoregressive format /\n"
        "      exact-match accuracy.\n"
        "\n"
        "  " << prog << " infer --model <model.bin> --input \"<LHS>\" [--max-output N]\n"
        "      Load a saved model and autoregressively complete the given LHS.\n";
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) { PrintUsage(argv[0]); return 1; }
    std::string sub = argv[1];

    if (sub == "generate") return hrccnn_llm_math::RunGenerate(argc - 1, argv + 1);
    if (sub == "train")    return hrccnn_llm_math::RunTrain(argc - 1, argv + 1);
    if (sub == "eval")     return hrccnn_llm_math::RunEval(argc - 1, argv + 1);
    if (sub == "infer")    return hrccnn_llm_math::RunInfer(argc - 1, argv + 1);
    PrintUsage(argv[0]);
    return 1;
}
