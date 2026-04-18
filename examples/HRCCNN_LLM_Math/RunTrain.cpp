#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
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

// --- Line cache: saves gen_seed + lines only (states are re-derived by
//     re-driving the reservoir, which is fast compared to generator
//     rejection-sampling). Cache file ends up at ~1-2 MB for 15k lines. -----

constexpr char kLineCacheMagic[8] = {'H','L','M','L','I','N','E','S'};
constexpr std::uint32_t kLineCacheVersion = 2;  // v2: field-wise GeneratorConfig hash

// FNV-1a 64-bit over each field of GeneratorConfig. Field-wise (not raw
// bytes) because the struct has padding after `bool rhs_filter_999` that
// would make a byte-blob hash non-deterministic across uninitialized
// regions.  UPDATE THIS when adding fields to GeneratorConfig — otherwise
// the cache will silently miss grammar-shape changes and return stale
// lines for a different distribution.
static_assert(std::is_trivially_copyable_v<GeneratorConfig>,
              "HashGeneratorConfig assumes POD GeneratorConfig");

std::uint64_t HashGeneratorConfig(const GeneratorConfig& c)
{
    constexpr std::uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t FNV_PRIME  = 0x100000001b3ULL;
    std::uint64_t h = FNV_OFFSET;
    auto mix = [&](const void* p, std::size_t n) {
        const auto* b = static_cast<const std::uint8_t*>(p);
        for (std::size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= FNV_PRIME;
        }
    };
    const std::uint8_t filter = c.rhs_filter_999 ? 1 : 0;
    mix(&filter,                    sizeof(filter));
    mix(&c.unary_minus_probability, sizeof(c.unary_minus_probability));
    mix(&c.depth2_probability,      sizeof(c.depth2_probability));
    mix(&c.operand_max_magnitude,   sizeof(c.operand_max_magnitude));
    return h;
}

struct LineCache
{
    std::uint64_t gen_seed     = 0;
    std::uint32_t samples      = 0;
    std::uint32_t val_samples  = 0;
    std::uint64_t gen_cfg_hash = 0;
    std::vector<std::string> train_lines;
    std::vector<std::string> val_lines;
};

bool SaveLineCache(const std::string& path, const LineCache& c)
{
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;
    os.write(kLineCacheMagic, 8);
    os.write(reinterpret_cast<const char*>(&kLineCacheVersion),
             sizeof(kLineCacheVersion));
    os.write(reinterpret_cast<const char*>(&c.gen_seed),     sizeof(c.gen_seed));
    os.write(reinterpret_cast<const char*>(&c.samples),      sizeof(c.samples));
    os.write(reinterpret_cast<const char*>(&c.val_samples),  sizeof(c.val_samples));
    os.write(reinterpret_cast<const char*>(&c.gen_cfg_hash), sizeof(c.gen_cfg_hash));

    auto write_lines = [&](const std::vector<std::string>& lines) {
        std::uint64_t n = lines.size();
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& l : lines) {
            std::uint32_t len = static_cast<std::uint32_t>(l.size());
            os.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (len > 0) os.write(l.data(), len);
        }
    };
    write_lines(c.train_lines);
    write_lines(c.val_lines);
    return static_cast<bool>(os);
}

bool LoadLineCache(const std::string& path, LineCache& c)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    char magic[8] = {};
    is.read(magic, 8);
    if (!is || std::memcmp(magic, kLineCacheMagic, 8) != 0) return false;

    std::uint32_t version = 0;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!is || version != kLineCacheVersion) return false;

    is.read(reinterpret_cast<char*>(&c.gen_seed),     sizeof(c.gen_seed));
    is.read(reinterpret_cast<char*>(&c.samples),      sizeof(c.samples));
    is.read(reinterpret_cast<char*>(&c.val_samples),  sizeof(c.val_samples));
    is.read(reinterpret_cast<char*>(&c.gen_cfg_hash), sizeof(c.gen_cfg_hash));
    if (!is) return false;

    auto read_lines = [&](std::vector<std::string>& lines) -> bool {
        std::uint64_t n = 0;
        if (!is.read(reinterpret_cast<char*>(&n), sizeof(n))) return false;
        if (n > 10'000'000) return false;  // sanity cap on corrupt files
        lines.resize(static_cast<std::size_t>(n));
        for (std::uint64_t i = 0; i < n; ++i) {
            std::uint32_t len = 0;
            if (!is.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
            if (len > 1024) return false;
            lines[i].resize(len);
            if (len > 0 && !is.read(lines[i].data(), len)) return false;
        }
        return true;
    };
    if (!read_lines(c.train_lines)) return false;
    if (!read_lines(c.val_lines))   return false;
    return static_cast<bool>(is);
}

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
    const std::uint64_t gen_cfg_hash = HashGeneratorConfig(gcfg);

    // --- Collect training + validation states under teacher forcing. ---
    //     Try line cache first; on miss, regenerate from the generator and
    //     save for next time.  Reservoir states themselves are never cached
    //     (would be multi-GB); they're re-derived by re-driving the reservoir.

    std::vector<std::string> train_lines;
    std::vector<std::string> val_lines;
    std::vector<float> targets;

    const std::string line_cache_path = args.output_path + ".lines.bin";
    bool loaded_from_cache = false;

    if (args.use_line_cache) {
        LineCache lc;
        if (LoadLineCache(line_cache_path, lc) &&
            lc.gen_seed == gen_seed &&
            lc.samples == static_cast<std::uint32_t>(args.samples) &&
            lc.val_samples == static_cast<std::uint32_t>(args.val_samples) &&
            lc.gen_cfg_hash == gen_cfg_hash) {
            train_lines = std::move(lc.train_lines);
            val_lines   = std::move(lc.val_lines);
            loaded_from_cache = true;
            std::cerr << "[train] loaded line cache from " << line_cache_path
                      << " (" << train_lines.size() << " train, "
                      << val_lines.size() << " val)\n";
        }
    }

    auto t_start = std::chrono::steady_clock::now();

    if (!loaded_from_cache) {
        train_lines.clear();
        val_lines.clear();
        train_lines.reserve(args.samples);
        val_lines.reserve(args.val_samples);
        std::size_t next_progress = 500;
        for (std::size_t i = 0; i < args.samples; ++i) {
            train_lines.push_back(gen.Sample());
            if (args.verbose && (i + 1 == next_progress || i + 1 == args.samples)) {
                std::cerr << "[train] generated " << (i + 1) << "/" << args.samples
                          << " train expressions\n";
                next_progress += 500;
            }
        }
        for (std::size_t i = 0; i < args.val_samples; ++i)
            val_lines.push_back(gen.Sample());

        if (args.use_line_cache) {
            LineCache lc;
            lc.gen_seed     = gen_seed;
            lc.samples      = static_cast<std::uint32_t>(args.samples);
            lc.val_samples  = static_cast<std::uint32_t>(args.val_samples);
            lc.gen_cfg_hash = gen_cfg_hash;
            lc.train_lines  = train_lines;
            lc.val_lines    = val_lines;
            if (SaveLineCache(line_cache_path, lc))
                std::cerr << "[train] saved line cache to " << line_cache_path << "\n";
            else
                std::cerr << "warning: failed to save line cache to "
                          << line_cache_path << "\n";
        }
    }

    auto t_gen_end = std::chrono::steady_clock::now();

    // Drive the reservoir over every cached line — this populates esn.States()
    // and `targets`.  Cheap relative to generator rejection sampling.
    std::size_t next_progress = 500;
    for (std::size_t i = 0; i < train_lines.size(); ++i) {
        LineSplit sp;
        if (!SplitLine(train_lines[i], sp)) continue;
        TeacherForceOne(esn, sp, targets);
        if (args.verbose && (i + 1 == next_progress || i + 1 == train_lines.size())) {
            std::cerr << "[train] reservoir-drive " << (i + 1) << "/"
                      << train_lines.size() << " train expressions\n";
            next_progress += 500;
        }
    }
    const std::size_t train_positions = esn.NumCollected();

    for (const auto& line : val_lines) {
        LineSplit sp;
        if (!SplitLine(line, sp)) continue;
        TeacherForceOne(esn, sp, targets);
    }
    const std::size_t total_positions = esn.NumCollected();
    const std::size_t val_positions = total_positions - train_positions;

    auto t_collected = std::chrono::steady_clock::now();
    std::cerr << "[train] collection: train_positions=" << train_positions
              << " val_positions=" << val_positions
              << " gen_elapsed=" << std::chrono::duration<double>(t_gen_end - t_start).count() << "s"
              << " res_elapsed=" << std::chrono::duration<double>(t_collected - t_gen_end).count() << "s"
              << (loaded_from_cache ? " (lines from cache)" : " (lines regenerated)") << "\n";

    // --- CNN readout config: HRCCNNBaseline + classification overrides. ---
    CNNReadoutConfig cnn_cfg = hcnn_presets::HRCCNNBaseline<kDIM>();
    cnn_cfg.task          = HCNNTask::Classification;
    cnn_cfg.num_outputs   = static_cast<int>(kVocabSize);
    cnn_cfg.num_layers    = args.cnn_num_layers;
    cnn_cfg.conv_channels = args.cnn_conv_channels;
    cnn_cfg.epochs            = args.epochs;
    cnn_cfg.batch_size        = args.batch_size;
    cnn_cfg.verbose           = args.verbose;
    cnn_cfg.verbose_train_acc = args.verbose_train_acc;
    cnn_cfg.lr_decay_epochs   = args.lr_decay_epochs;

    std::cerr << "[train] CNN cfg: nl=" << cnn_cfg.num_layers
              << " ch=" << cnn_cfg.conv_channels
              << " head=" << (cnn_cfg.readout_type == HCNNReadoutType::FLATTEN ? "FLATTEN" : "GAP")
              << " lr=" << cnn_cfg.lr_max
              << " epochs=" << cnn_cfg.epochs
              << " bs=" << cnn_cfg.batch_size
              << " num_outputs=" << cnn_cfg.num_outputs << "\n";

    // --- Eval reporter (shared between mid-training hook and final report). ---
    const std::size_t n_auto        = std::min(args.autoreg_samples, val_lines.size());
    const std::size_t n_show        = args.eval_show_samples;
    auto run_eval_report = [&](const std::string& tag) {
        double acc_train = esn.Accuracy(targets.data(), 0, train_positions);
        double acc_val   = (val_positions > 0)
                           ? esn.Accuracy(targets.data(), train_positions, val_positions)
                           : 0.0;

        std::size_t exact = 0, format_ok = 0, non_stop = 0;
        std::vector<std::tuple<std::string, std::string, std::string>> samples;
        samples.reserve(n_show);
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
            if (samples.size() < n_show) {
                samples.emplace_back(sp.lhs, sp.rhs, emitted);
            }
        }

        std::cerr << "[" << tag << "] teacher-forced char-accuracy:"
                  << " train=" << acc_train
                  << " val=" << acc_val << "\n";
        if (n_auto > 0) {
            std::cerr << "[" << tag << "] autoregressive (" << n_auto
                      << " val lines):"
                      << " exact=" << exact << "/" << n_auto
                      << " format_ok=" << format_ok << "/" << n_auto
                      << " non_stop=" << non_stop << "/" << n_auto << "\n";
            for (const auto& [lhs, expected, got] : samples) {
                std::cerr << "  " << lhs << " = " << expected
                          << "   ->   " << got << "\n";
            }
        }
    };

    // --- Train. ---
    CNNTrainHooks hooks;
    hooks.eval_every_epochs = args.eval_every_epochs;
    hooks.epoch_callback = [&](int epoch_done, int total_epochs, float lr) {
        std::ostringstream tag;
        tag << "eval e=" << epoch_done << "/" << total_epochs
            << " lr=" << lr;
        run_eval_report(tag.str());

        // Checkpoint snapshot. CNNReadout::fire_hook has already called
        // flatten_weights() for us, so GetReadoutState() reflects the
        // just-completed epoch's weights.
        if (args.output_path.empty()) return;
        ModelFile ckpt;
        ckpt.dim                    = static_cast<std::uint32_t>(kDIM);
        ckpt.meta.training_seed     = gen_seed;
        ckpt.meta.training_samples  = static_cast<std::uint32_t>(args.samples);
        ckpt.meta.training_epochs   = static_cast<std::uint32_t>(epoch_done);
        ckpt.meta.git_sha           = args.git_sha;
        ckpt.reservoir_cfg          = esn.GetConfig();
        ckpt.cnn_cfg                = cnn_cfg;
        ckpt.readout                = ToSerial<kDIM>(esn.GetReadoutState());

        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), ".e%04d.bin", epoch_done);
        const std::string ckpt_path = args.output_path + suffix;
        if (SaveModelFile(ckpt_path, ckpt))
            std::cerr << "[train] checkpoint saved: " << ckpt_path << "\n";
        else
            std::cerr << "warning: failed to save checkpoint to " << ckpt_path << "\n";
    };

    auto t_train_start = std::chrono::steady_clock::now();
    esn.Train(targets.data(), train_positions, cnn_cfg, hooks);
    auto t_train_end = std::chrono::steady_clock::now();
    std::cerr << "[train] training elapsed="
              << std::chrono::duration<double>(t_train_end - t_train_start).count()
              << "s\n";

    // --- Final report. Skipped when hooks were active — the fire_hook at_final
    //     branch already produced an identical report on the last completed epoch.
    if (hooks.eval_every_epochs <= 0 || cnn_cfg.epochs <= 0)
        run_eval_report("train-final");

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
