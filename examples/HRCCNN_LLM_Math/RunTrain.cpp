#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Config.h"
#include "Dataset.h"
#include "ESN.h"
#include "FormatCheck.h"
#include "Generator.h"
#include "NumberFormat.h"
#include "Serialization.h"
#include "Vocab.h"
#include "readout/HCNNPresets.h"

namespace hrccnn_llm_math {

namespace {

constexpr std::size_t kDIM = 12;
constexpr std::size_t kMaxOutputChars = 16;

}  // namespace

int RunTrain()
{
    const config::TrainCfg& args = config::kTrain;

    if (args.output_path.empty()) {
        std::cerr << "error: config::kTrain.output_path is empty\n";
        return 1;
    }

    std::uint64_t gen_seed = args.gen_seed;
    if (!args.use_fixed_gen_seed) {
        std::random_device rd;
        gen_seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }
    std::uint64_t reservoir_seed = args.use_fixed_reservoir_seed
                                       ? args.reservoir_seed
                                       : (gen_seed ^ 0x9E3779B97F4A7C15ULL);

    ReservoirConfig rcfg;
    rcfg.seed            = reservoir_seed;
    rcfg.num_inputs      = kInputBits;
    rcfg.output_fraction = args.output_fraction;

    std::cerr << "[train] DIM=" << kDIM
              << " N=" << (1ULL << kDIM)
              << " num_inputs=" << rcfg.num_inputs
              << " output_fraction=" << rcfg.output_fraction
              << " reservoir_seed=" << rcfg.seed << "\n";
    std::cerr << "[train] samples=" << args.samples
              << " val_samples=" << args.val_samples
              << " gen_seed=" << gen_seed
              << " epochs=" << args.epochs
              << " batch_size=" << args.batch_size << "\n";

    ESN<kDIM> esn(rcfg, ReadoutType::HCNN, FeatureMode::Raw);

    GeneratorConfig gcfg;
    gcfg.rhs_filter_999 = args.rhs_filter_999;
    Generator gen(gen_seed, gcfg);

    // --- Collect training + validation states under teacher forcing. ---
    std::vector<std::string> train_lines;
    std::vector<std::string> val_lines;
    train_lines.reserve(args.samples);
    val_lines.reserve(args.val_samples);

    std::vector<float> targets;

    auto t_start = std::chrono::steady_clock::now();
    std::size_t next_progress = 500;
    for (std::size_t i = 0; i < args.samples; ++i) {
        std::string line = gen.Sample();
        train_lines.push_back(line);
        LineSplit sp;
        if (!SplitLine(line, sp)) continue;
        TeacherForceOne(esn, sp, targets);
        if (args.verbose && (i + 1 == next_progress || i + 1 == args.samples)) {
            std::cerr << "[train] collected " << (i + 1) << "/" << args.samples
                      << " train expressions\n";
            next_progress += 500;
        }
    }
    const std::size_t train_positions = esn.NumCollected();

    for (std::size_t i = 0; i < args.val_samples; ++i) {
        std::string line = gen.Sample();
        val_lines.push_back(line);
        LineSplit sp;
        if (!SplitLine(line, sp)) continue;
        TeacherForceOne(esn, sp, targets);
    }
    const std::size_t total_positions = esn.NumCollected();
    const std::size_t val_positions = total_positions - train_positions;

    auto t_collected = std::chrono::steady_clock::now();
    std::cerr << "[train] collection: train_positions=" << train_positions
              << " val_positions=" << val_positions
              << " elapsed=" << std::chrono::duration<double>(t_collected - t_start).count() << "s\n";

    // --- CNN readout config: HRCCNNBaseline + classification overrides. ---
    CNNReadoutConfig cnn_cfg = hcnn_presets::HRCCNNBaseline<kDIM>();
    cnn_cfg.task        = HCNNTask::Classification;
    cnn_cfg.num_outputs = static_cast<int>(kVocabSize);
    cnn_cfg.epochs      = args.epochs;
    cnn_cfg.batch_size  = args.batch_size;
    cnn_cfg.verbose     = args.verbose;

    std::cerr << "[train] CNN cfg: nl=" << cnn_cfg.num_layers
              << " ch=" << cnn_cfg.conv_channels
              << " head=" << (cnn_cfg.readout_type == HCNNReadoutType::FLATTEN ? "FLATTEN" : "GAP")
              << " lr=" << cnn_cfg.lr_max
              << " epochs=" << cnn_cfg.epochs
              << " bs=" << cnn_cfg.batch_size
              << " num_outputs=" << cnn_cfg.num_outputs << "\n";

    // --- Train. ---
    auto t_train_start = std::chrono::steady_clock::now();
    esn.Train(targets.data(), train_positions, cnn_cfg);
    auto t_train_end = std::chrono::steady_clock::now();
    std::cerr << "[train] training elapsed="
              << std::chrono::duration<double>(t_train_end - t_train_start).count()
              << "s\n";

    // --- Teacher-forced character accuracy on train + val segments. ---
    double acc_train = esn.Accuracy(targets.data(), 0, train_positions);
    double acc_val   = (val_positions > 0)
                       ? esn.Accuracy(targets.data(), train_positions, val_positions)
                       : 0.0;
    std::cerr << "[train] teacher-forced char-accuracy:"
              << " train=" << acc_train
              << " val=" << acc_val << "\n";

    // --- Autoregressive sanity check on a subset of validation lines. ---
    const std::size_t n_auto = std::min(args.autoreg_samples, val_lines.size());
    std::size_t exact = 0, format_ok = 0, non_stop = 0;
    for (std::size_t i = 0; i < n_auto; ++i) {
        LineSplit sp;
        if (!SplitLine(val_lines[i], sp)) continue;
        std::string emitted = GenerateRHS(esn, sp.lhs, kMaxOutputChars);
        bool stopped_on_hash = !emitted.empty() && emitted.back() == '#';
        if (!stopped_on_hash) ++non_stop;
        if (IsValidFormat(emitted)) ++format_ok;
        std::string emitted_rhs =
            stopped_on_hash ? emitted.substr(0, emitted.size() - 1) : emitted;
        if (emitted_rhs == sp.rhs && stopped_on_hash) ++exact;
    }
    if (n_auto > 0) {
        std::cerr << "[train] autoregressive (" << n_auto << " val lines):"
                  << " exact=" << exact << "/" << n_auto
                  << " format_ok=" << format_ok << "/" << n_auto
                  << " non_stop=" << non_stop << "/" << n_auto << "\n";
    }

    // --- Save model. ---
    ModelFile mf;
    mf.dim                 = static_cast<std::uint32_t>(kDIM);
    mf.meta.training_seed  = gen_seed;
    mf.meta.training_samples = static_cast<std::uint32_t>(args.samples);
    mf.meta.training_epochs  = static_cast<std::uint32_t>(args.epochs);
    mf.meta.git_sha        = args.git_sha;
    mf.reservoir_cfg       = esn.GetConfig();
    mf.cnn_cfg             = cnn_cfg;
    mf.readout             = ToSerial<kDIM>(esn.GetReadoutState());

    if (!SaveModelFile(args.output_path, mf)) {
        std::cerr << "error: failed to save model to " << args.output_path << "\n";
        return 4;
    }
    std::cerr << "[train] saved model to " << args.output_path << "\n";
    return 0;
}

}  // namespace hrccnn_llm_math
