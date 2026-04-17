#include <iostream>
#include <string>

namespace hrccnn_llm_math {

int RunGenerate(int argc, char** argv);

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
        "  " << prog << " train|eval|infer\n"
        "      Not yet implemented (Phase 2+).\n";
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) { PrintUsage(argv[0]); return 1; }
    std::string sub = argv[1];

    if (sub == "generate") {
        return hrccnn_llm_math::RunGenerate(argc - 1, argv + 1);
    }
    if (sub == "train" || sub == "eval" || sub == "infer") {
        std::cerr << "'" << sub << "' not yet implemented (Phase 2+).\n";
        return 2;
    }
    PrintUsage(argv[0]);
    return 1;
}
