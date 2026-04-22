#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace hcnn { class HCNN; }

enum class HCNNTask { Regression, Classification };

/// HCNN readout architecture and training parameters.
/// Must stay trivially copyable (POD) for checkpoint serialization.
struct HCNNReadoutConfig {
    int num_outputs      = 1;       ///< Classes (classification) or targets (regression).
    HCNNTask task        = HCNNTask::Regression;
    int num_layers       = 0;       ///< Conv+Pool pairs. 0 = auto: min(DIM-2, 2).
    int conv_channels    = 16;      ///< Base channels (doubles per layer).
    int epochs           = 200;
    int batch_size       = 32;
    float lr_max         = 0.005f;  ///< Cosine annealing peak. Keep <= 0.003 to avoid NaN.
    float lr_min_frac    = 0.1f;    ///< Floor = lr_max * lr_min_frac.
    int   lr_decay_epochs = 0;      ///< Cosine decay horizon. 0 = use `epochs`.
    float weight_decay   = 0.0f;
    unsigned seed        = 42;      ///< CNN weight initialization seed.
    bool verbose         = false;   ///< Print per-epoch lr to stdout.
    bool verbose_train_acc = false;  ///< Also print train accuracy/MSE each epoch.
};

/// Mid-training evaluation hooks. Separate from HCNNReadoutConfig to
/// preserve POD layout. Fires after every `eval_every_epochs` epochs
/// and after the final epoch. The readout is usable for Predict/R2/Accuracy
/// inside the callback.
struct CNNTrainHooks {
    int eval_every_epochs = 0;
    std::function<void(int epoch, int total_epochs, float lr)> epoch_callback;
    bool stop_requested = false;    ///< Set true in callback to stop early.
};

/// HypercubeCNN-based learned readout operating on raw reservoir state
/// (N = 2^DIM floats per timestep).
///
/// Data path: raw state -> standardize -> Conv+Pool stack -> Flatten ->
/// Linear -> de-center -> output.
///
/// Architecture auto-sized from DIM: min(DIM-2, 2) Conv+Pool pairs,
/// channels doubling per layer. Override via HCNNReadoutConfig::num_layers.
///
/// PIMPL: hcnn::HCNN held via unique_ptr; #include "HCNN.h" in .cpp only.
class HCNNReadout
{
public:
    HCNNReadout();
    ~HCNNReadout();
    HCNNReadout(HCNNReadout&&) noexcept;
    HCNNReadout& operator=(HCNNReadout&&) noexcept;

    HCNNReadout(const HCNNReadout&) = delete;
    HCNNReadout& operator=(const HCNNReadout&) = delete;

    // ----- Batch training -----

    /// Train on collected reservoir states (row-major, N floats per sample).
    void Train(const float* states, const float* targets,
               size_t num_samples, size_t dim,
               const HCNNReadoutConfig& config,
               CNNTrainHooks& hooks);

    void Train(const float* states, const float* targets,
               size_t num_samples, size_t dim,
               const HCNNReadoutConfig& config = {});

    // ----- Online (streaming) training -----

    /// Initialize for online training. Computes standardization from warmup
    /// states, builds architecture, sets Adam optimizer.
    void InitOnline(const float* warmup_states, size_t warmup_count,
                    size_t dim, const HCNNReadoutConfig& config);

    /// Single-sample online step (classification).
    void TrainOnlineStep(const float* state, int target_class,
                         float lr, float weight_decay = 0.0f);

    /// Mini-batch online step (classification). Parallelized via HCNN::TrainBatch.
    void TrainOnlineBatch(const float* states, const int* targets,
                          size_t count, float lr, float weight_decay = 0.0f);

    /// Single-sample online step (regression). Centered internally if target means set.
    void TrainOnlineStepRegression(const float* state, const float* target,
                                   float lr, float weight_decay = 0.0f);

    /// Mini-batch online step (regression). Centered internally if target means set.
    void TrainOnlineBatchRegression(const float* states, const float* targets,
                                    size_t count, float lr, float weight_decay = 0.0f);

    /// Compute per-output target centering. Call after InitOnline for regression.
    void ComputeTargetCentering(const float* targets, size_t num_samples);

    // ----- Prediction -----

    /// Multi-output: writes num_outputs floats. Regression: de-centered. Classification: logits.
    void PredictRaw(const float* state, float* output) const;

    /// Scalar prediction. Asserts num_outputs == 1.
    [[nodiscard]] float PredictRaw(const float* state) const;

    /// Returns predicted class index (argmax over logits).
    [[nodiscard]] int PredictClass(const float* state) const;

    // ----- Evaluation -----

    /// R-squared (averaged across outputs for multi-output regression).
    [[nodiscard]] double R2(const float* states, const float* targets,
                            size_t num_samples) const;

    /// Classification accuracy (argmax vs label for multi-class).
    [[nodiscard]] double Accuracy(const float* states, const float* labels,
                                  size_t num_samples) const;

    // ----- Accessors -----

    [[nodiscard]] size_t NumOutputs()  const { return num_outputs_; }
    [[nodiscard]] size_t NumFeatures() const { return num_features_; }
    [[nodiscard]] bool   IsTrained()   const { return trained_; }
    [[nodiscard]] const HCNNReadoutConfig& GetConfig() const { return config_; }

    // ----- Serialization -----

    [[nodiscard]] double Bias() const { return target_mean_.empty() ? 0.0 : target_mean_[0]; }
    [[nodiscard]] const std::vector<double>& TargetMean()    const { return target_mean_; }
    [[nodiscard]] const std::vector<float>&  FeatureMean()   const { return input_mean_; }
    [[nodiscard]] const std::vector<float>&  FeatureScale()  const { return input_scale_; }

    /// Flattened CNN weights (opaque blob). Lazily synced from live network.
    [[nodiscard]] const std::vector<double>& Weights() const;

    /// Restore a previously trained state. Rebuilds the CNN from config + weights.
    void SetState(std::vector<double> weights, double bias,
                  std::vector<float> feature_mean, std::vector<float> feature_scale,
                  std::vector<double> target_mean = {});

    /// Pre-set config before SetState (needed for model loading without training).
    void SetConfig(const HCNNReadoutConfig& cfg);

private:
    std::unique_ptr<hcnn::HCNN> net_;
    HCNNReadoutConfig config_;
    bool trained_ = false;
    size_t dim_ = 0;
    size_t num_features_ = 0;
    size_t num_outputs_ = 1;

    std::vector<float>  input_mean_;
    std::vector<float>  input_scale_;
    std::vector<double> target_mean_;

    mutable std::vector<double> weights_blob_;

    mutable std::vector<float> scratch_state_;
    mutable std::vector<float> scratch_embedded_;
    mutable std::vector<float> scratch_pred_;
    std::vector<float> scratch_batch_;
    std::vector<float> scratch_target_;
    std::vector<float> scratch_centered_batch_;

    void standardize(const float* in, float* out, size_t n) const;
    void compute_standardization(const float* states, size_t num_samples, size_t n);
    void build_architecture();
    void flatten_weights();
    void rebuild_from_blob();
};
