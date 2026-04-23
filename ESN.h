#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include "Reservoir.h"
#include "Readout.h"

/// @brief Echo-state network implementing the full pipeline:
///        Reservoir -> [Output Selection] -> Readout.
///
/// @tparam DIM Hypercube dimension (5-16). Vertex count is 2^DIM.
template <size_t DIM>
class ESN
{
    static constexpr size_t N = 1ULL << DIM;

public:
    explicit ESN(const ReservoirConfig& cfg);

    // ---------------------------------------------------------------
    //  Reservoir driving
    // ---------------------------------------------------------------

    void Warmup(const float* inputs, size_t num_steps);
    void Run(const float* inputs, size_t num_steps);

    void ClearStates();
    void ResetReservoirOnly();

    void SaveReservoirState(float* state_out, float* output_out) const;
    void RestoreReservoirState(const float* state_in, const float* output_in);

    // ---------------------------------------------------------------
    //  Training
    // ---------------------------------------------------------------

    void Train(const float* targets, size_t train_size);

    void Train(const float* targets, size_t train_size,
               const ReadoutConfig& config);

    void Train(const float* targets, size_t train_size,
               const ReadoutConfig& config,
               CNNTrainHooks& hooks);

    void InitOnline(const float* warmup_inputs, size_t warmup_count,
                    const ReadoutConfig& config);

    void TrainLiveStep(float target_class, float lr, float weight_decay = 0.0f);

    void CopyLiveState(float* out) const;

    void TrainLiveBatch(const float* states, const int* targets,
                        size_t count, float lr, float weight_decay = 0.0f);

    void TrainLiveStepRegression(const float* target, float lr,
                                 float weight_decay = 0.0f);

    void TrainLiveBatchRegression(const float* states, const float* targets,
                                  size_t count, float lr, float weight_decay = 0.0f);

    void ComputeTargetCentering(const float* targets, size_t num_samples);

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    [[nodiscard]] float PredictRaw(size_t timestep) const;
    void PredictRaw(size_t timestep, float* output) const;

    [[nodiscard]] float PredictLiveRaw() const;
    void PredictLiveRaw(float* output) const;

    /// @brief R-squared on collected timesteps [start, start+count).
    /// @param targets  Must span timesteps [0, start+count): for regression,
    ///                 (start+count)*num_outputs floats (row-major); for
    ///                 classification, (start+count) floats.  The method
    ///                 indexes from targets[start*num_outputs].
    [[nodiscard]] double R2(const float* targets, size_t start, size_t count) const;

    /// @param targets  Same layout contract as R2.
    [[nodiscard]] double NRMSE(const float* targets, size_t start, size_t count) const;

    /// @param labels  Must span timesteps [0, start+count): (start+count)
    ///               floats (class indices).  Indexed from labels[start].
    [[nodiscard]] double Accuracy(const float* labels, size_t start, size_t count) const;

    [[nodiscard]] size_t NumOutputs() const;

    // ---------------------------------------------------------------
    //  State access
    // ---------------------------------------------------------------

    /// @brief Extract stride-selected vertices from collected states.
    [[nodiscard]] std::vector<float> SelectedStates() const;

    // ---------------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------------
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] float OutputFraction() const { return output_fraction_; }
    [[nodiscard]] size_t NumOutputVerts() const { return num_output_verts_; }
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

    // --- Config & persistence ---

    [[nodiscard]] ReservoirConfig GetConfig() const;

    struct ReadoutState {
        std::vector<double> weights;
        double bias = 0.0;
        std::vector<float> feature_mean;
        std::vector<float> feature_scale;
        std::vector<double> target_mean;
        bool is_trained = false;
    };

    [[nodiscard]] ReadoutState GetReadoutState() const;
    void SetReadoutState(const ReadoutState& state);
    void SetCNNConfig(const ReadoutConfig& cfg);

private:
    std::unique_ptr<Reservoir<DIM>> reservoir_;
    Readout readout_;

    size_t num_inputs_ = 1;
    float output_fraction_ = 1.0f;
    size_t output_stride_ = 1;
    size_t num_output_verts_ = N;

    std::vector<float> states_;
    size_t num_collected_ = 0;

    [[nodiscard]] size_t EffectiveDIM() const;
    const float* SubsampleIntoScratch(const float* src) const;
    const float* ReadoutInput(size_t timestep) const;
    [[nodiscard]] std::vector<float> ReadoutStates(size_t start, size_t count) const;

    mutable std::vector<float> scratch_subsampled_;
};
