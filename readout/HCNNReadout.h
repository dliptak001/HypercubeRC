#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace hcnn { class HCNN; }

/// Task type for the HCNN readout.
enum class HCNNTask { Regression, Classification };

/// Configuration for the CNN readout's architecture and training.
struct HCNNReadoutConfig {
    int num_outputs   = 1;        ///< Number of output neurons (classes or regression targets).
    HCNNTask task     = HCNNTask::Regression; ///< Task type.
    int num_layers    = 0;        ///< Conv+Pool pairs. 0 = auto: min(DIM-3, 4). Channels double per layer.
    int conv_channels = 16;       ///< Base convolution channels (doubles per layer: 16, 32, 64, 128).
    int epochs        = 200;      ///< Training epochs.
    int batch_size    = 32;       ///< Mini-batch size.
    float lr_max      = 0.005f;   ///< Peak learning rate (cosine annealing).
    float lr_min_frac = 0.1f;     ///< lr_min = lr_max * lr_min_frac.
    int   lr_decay_epochs = 0;    ///< Cosine decay horizon. 0 = use `epochs`.
                                  ///< Set > epochs to trace only a prefix of
                                  ///< the cosine curve (keeps lr high when
                                  ///< shortening a run for wall-clock).
    float weight_decay = 0.0f;    ///< L2 weight decay.
    unsigned seed     = 42;       ///< Weight initialization seed.
    bool verbose      = false;    ///< Print per-epoch lr line to stdout.
    bool verbose_train_acc = false;///< When true, also compute + print training
                                   ///< accuracy each epoch. Costs one extra
                                   ///< forward pass over the full training set
                                   ///< per epoch — disable when using a hook
                                   ///< that already reports accuracy.
};

/// Runtime-only training hooks.  Kept out of HCNNReadoutConfig because the
/// config must stay POD for the checkpoint format (see Serialization.cpp's
/// static_assert).  The callback fires after every `eval_every_epochs`
/// completed epoch and unconditionally after the final epoch.
///
/// When the callback runs, HCNNReadout's `net_` holds the weights produced
/// by the just-completed epoch and `trained_` is already true, so the
/// client can call Predict*/Accuracy/R2 directly (typically via the
/// enclosing ESN) for a mid-training eval.
struct CNNTrainHooks {
    int eval_every_epochs = 0;  ///< 0 disables mid-training callbacks.
    std::function<void(int epoch_done_1based, int total_epochs, float lr)>
        epoch_callback;
    bool stop_requested = false; ///< Callback sets this to end training early.
};

/// @brief HypercubeCNN-based readout for reservoir computing.
///
/// HypercubeCNN-based learned convolutional readout that operates directly on
/// raw reservoir state (N = 2^DIM floats per timestep).  The CNN's learned
/// convolution kernels discover which vertex interactions predict the target —
/// no hand-crafted feature extraction, no stride selection.
///
/// **Data path:** raw reservoir state -> input standardization ->
///   HypercubeCNN (Conv->Pool stack -> FLATTEN -> Linear) -> de-center -> output.
///
/// **Architecture:** Auto-sized from DIM: min(DIM-3, 4) Conv+Pool pairs,
///   channels doubling per layer (16, 32, 64, 128).  Override via
///   HCNNReadoutConfig::num_layers.
///
/// **PIMPL:** The hcnn::HCNN object is held via unique_ptr behind a forward
/// declaration.  #include "HCNN.h" stays in the .cpp only.
class HCNNReadout
{
public:
    HCNNReadout();
    ~HCNNReadout();
    HCNNReadout(HCNNReadout&&) noexcept;
    HCNNReadout& operator=(HCNNReadout&&) noexcept;

    // Non-copyable (owns a unique_ptr<HCNN> which is non-copyable).
    HCNNReadout(const HCNNReadout&) = delete;
    HCNNReadout& operator=(const HCNNReadout&) = delete;

    /// @brief Train the CNN readout on raw reservoir states.
    /// @param states     Row-major: num_samples rows, each of N = 2^dim floats.
    /// @param targets    For regression: num_samples * num_outputs floats (row-major).
    ///                   For classification: num_samples floats (class indices as float).
    /// @param num_samples Number of training samples.
    /// @param dim        Hypercube dimension (N = 2^dim vertices per state).
    /// @param config     Architecture and training hyperparameters.
    void Train(const float* states, const float* targets,
               size_t num_samples, size_t dim,
               const HCNNReadoutConfig& config,
               CNNTrainHooks& hooks);

    void Train(const float* states, const float* targets,
               size_t num_samples, size_t dim,
               const HCNNReadoutConfig& config = {});

    /// @brief Initialize CNN for online (streaming) training.
    /// Computes standardization stats from a warmup buffer, builds the
    /// architecture, and sets the optimizer.  Call before TrainOnlineStep.
    void InitOnline(const float* warmup_states, size_t warmup_count,
                    size_t dim, const HCNNReadoutConfig& config);

    /// @brief Single-sample online gradient step (classification).
    /// State is one subsampled reservoir state (2^dim floats).
    void TrainOnlineStep(const float* state, int target_class,
                         float lr, float weight_decay = 0.0f);

    /// @brief Mini-batch online gradient step (classification).
    /// states: count rows of 2^dim floats (row-major, raw — standardized internally).
    /// targets: count int class indices.
    /// Parallelized across threads via HCNN::TrainBatch.
    void TrainOnlineBatch(const float* states, const int* targets,
                          size_t count, float lr, float weight_decay = 0.0f);

    /// @brief Single-sample online gradient step (regression).
    /// target: num_outputs floats. Internally centered if target_mean_ is set.
    void TrainOnlineStepRegression(const float* state, const float* target,
                                   float lr, float weight_decay = 0.0f);

    /// @brief Mini-batch online gradient step (regression).
    /// states: count rows of 2^dim floats (row-major, raw — standardized internally).
    /// targets: count * num_outputs contiguous floats (row-major).
    /// Internally centered if target_mean_ is set.
    void TrainOnlineBatchRegression(const float* states, const float* targets,
                                    size_t count, float lr, float weight_decay = 0.0f);

    /// @brief Compute and store per-output target centering from sample targets.
    /// Call after InitOnline for regression tasks so that online training
    /// subtracts the mean and PredictRaw adds it back (matching batch behavior).
    /// targets: num_samples * num_outputs floats (row-major).
    void ComputeTargetCentering(const float* targets, size_t num_samples);

    /// @brief Multi-output prediction: writes num_outputs floats to output.
    /// For regression: de-centered predictions.  For classification: raw logits.
    void PredictRaw(const float* state, float* output) const;

    /// @brief Scalar prediction (backward compat, num_outputs must be 1).
    [[nodiscard]] float PredictRaw(const float* state) const;

    /// @brief Classification: returns predicted class index.
    [[nodiscard]] int PredictClass(const float* state) const;

    /// @brief R-squared on raw state vectors (regression).
    /// For multi-output: returns average R2 across outputs.
    /// targets layout: num_samples * num_outputs floats (row-major).
    [[nodiscard]] double R2(const float* states, const float* targets,
                            size_t num_samples) const;

    /// @brief Classification accuracy on raw state vectors.
    /// For multi-class: compares argmax(logits) vs int(label).
    /// labels layout: num_samples floats (class indices).
    [[nodiscard]] double Accuracy(const float* states, const float* labels,
                                  size_t num_samples) const;

    /// @brief Number of output neurons.
    [[nodiscard]] size_t NumOutputs() const { return num_outputs_; }

    // --- State accessors ---

    [[nodiscard]] size_t NumFeatures() const { return num_features_; }
    [[nodiscard]] double Bias() const { return target_mean_.empty() ? 0.0 : target_mean_[0]; }
    [[nodiscard]] const std::vector<double>& TargetMean() const { return target_mean_; }

    [[nodiscard]] const std::vector<float>& FeatureMean() const { return input_mean_; }
    [[nodiscard]] const std::vector<float>& FeatureScale() const { return input_scale_; }

    /// @brief Flattened CNN weights (opaque blob for serialization).
    /// Lazily syncs from the live network on first call after training.
    [[nodiscard]] const std::vector<double>& Weights() const;

    /// @brief Restore a previously trained state.
    /// target_mean: per-output centering (regression). Empty = no centering.
    void SetState(std::vector<double> weights, double bias,
                  std::vector<float> feature_mean, std::vector<float> feature_scale,
                  std::vector<double> target_mean = {});

    /// @brief Pre-set architecture config before restoring weights via SetState.
    /// Required when loading a saved model without training — SetState's
    /// rebuild_from_blob() needs config_ to reconstruct the CNN.
    void SetConfig(const HCNNReadoutConfig& cfg);

    /// @brief Check if the readout has been trained.
    [[nodiscard]] bool IsTrained() const { return trained_; }

    /// @brief Get the config used for training (valid after Train).
    [[nodiscard]] const HCNNReadoutConfig& GetConfig() const { return config_; }

private:
    std::unique_ptr<hcnn::HCNN> net_;
    HCNNReadoutConfig config_;
    bool trained_ = false;
    size_t dim_ = 0;
    size_t num_features_ = 0;  // N = 2^dim (raw state size, for interface compat)
    size_t num_outputs_ = 1;

    // Input standardization: per-vertex mean and 1/std.
    std::vector<float> input_mean_;
    std::vector<float> input_scale_;

    // Target centering (per-output, regression only).
    std::vector<double> target_mean_;

    // Flattened weight blob for serialization.  Lazily synced from net_
    // on first Weights() call after training (avoids per-step overhead).
    mutable std::vector<double> weights_blob_;

    // Persistent scratch buffers for prediction (zero per-call allocation).
    mutable std::vector<float> scratch_state_;
    mutable std::vector<float> scratch_embedded_;
    mutable std::vector<float> scratch_pred_;
    std::vector<float> scratch_batch_;        // Persistent buffer for TrainOnlineBatch standardization.
    std::vector<float> scratch_target_;       // Persistent buffer for online regression target centering.

    void standardize(const float* in, float* out, size_t n) const;
    void compute_standardization(const float* states, size_t num_samples, size_t n);
    void build_architecture();
    void flatten_weights();
    void rebuild_from_blob();
};
