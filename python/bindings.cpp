#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include "../ESN.h"

namespace py = pybind11;

template <size_t DIM>
void bind_esn(py::module_& m, const char* name)
{
    using E = ESN<DIM>;
    constexpr size_t NN = 1ULL << DIM;

    py::class_<E>(m, name)
        // ── Construction ──
        .def(py::init([](uint64_t seed, float spectral_radius, float input_scaling,
                         float leak_rate, float alpha, size_t num_inputs,
                         float output_fraction) {
            ReservoirConfig cfg;
            cfg.seed             = seed;
            cfg.spectral_radius  = spectral_radius;
            cfg.input_scaling    = input_scaling;
            cfg.leak_rate        = leak_rate;
            cfg.alpha            = alpha;
            cfg.num_inputs       = num_inputs;
            cfg.output_fraction  = output_fraction;
            return std::make_unique<E>(cfg);
        }),
            py::arg("seed")             = 0ULL,
            py::arg("spectral_radius")  = 0.9f,
            py::arg("input_scaling")    = 0.02f,
            py::arg("leak_rate")        = 1.0f,
            py::arg("alpha")            = 1.0f,
            py::arg("num_inputs")       = 1ULL,
            py::arg("output_fraction")  = 1.0f)

        // ── Reservoir driving ──
        .def("warmup", [](E& self, py::array_t<float, py::array::c_style | py::array::forcecast> inputs) {
            auto buf = inputs.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumInputs();
            if (total % K != 0)
                throw std::invalid_argument("Input size must be divisible by num_inputs");
            self.Warmup(static_cast<const float*>(buf.ptr), total / K);
        }, py::arg("inputs"),
           "Drive the reservoir without recording states (wash out initial transient).")

        .def("run", [](E& self, py::array_t<float, py::array::c_style | py::array::forcecast> inputs) {
            auto buf = inputs.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumInputs();
            if (total % K != 0)
                throw std::invalid_argument("Input size must be divisible by num_inputs");
            self.Run(static_cast<const float*>(buf.ptr), total / K);
        }, py::arg("inputs"),
           "Drive the reservoir and record states for training/evaluation.")

        .def("clear_states", &E::ClearStates,
             "Clear collected states and cached features. Keeps trained readout.")

        .def("reset_reservoir_only", &E::ResetReservoirOnly,
             "Zero only the reservoir state; collected states preserved.")

        .def("save_reservoir_state", [](const E& self) {
            py::array_t<float> state(NN);
            py::array_t<float> output(NN);
            self.SaveReservoirState(state.mutable_data(), output.mutable_data());
            return py::make_tuple(state, output);
        }, "Snapshot the current reservoir state.\n"
           "Returns (state, output) tuple of (N,) float arrays.")

        .def("restore_reservoir_state", [](E& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> state,
                py::array_t<float, py::array::c_style | py::array::forcecast> output) {
            if (state.size() != static_cast<py::ssize_t>(NN) ||
                output.size() != static_cast<py::ssize_t>(NN))
                throw std::invalid_argument(
                    "state and output must each have N=" + std::to_string(NN) + " elements");
            self.RestoreReservoirState(
                static_cast<const float*>(state.request().ptr),
                static_cast<const float*>(output.request().ptr));
        }, py::arg("state"), py::arg("output"),
           "Restore a previously saved reservoir state.\n"
           "state and output must be (N,) float arrays from save_reservoir_state.")

        // ── Batch training ──
        .def("train", [](E& self,
                         py::array_t<float, py::array::c_style | py::array::forcecast> targets) {
            auto buf = targets.request();
            size_t n = static_cast<size_t>(buf.size);
            const float* ptr = static_cast<const float*>(buf.ptr);

            if (n > self.NumCollected())
                throw std::invalid_argument(
                    "train_size (" + std::to_string(n) +
                    ") exceeds num_collected (" + std::to_string(self.NumCollected()) + ")");

            self.Train(ptr, n);
        },
            py::arg("targets"),
            "Train the HCNN readout on collected states with default parameters.\n"
            "Use train_cnn() for full control over CNN config.")

        .def("train_cnn", [](E& self,
                             py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                             int num_outputs, const char* task,
                             int num_layers, int conv_channels,
                             int epochs_val, int batch_size,
                             float lr_max, float lr_min_frac,
                             int lr_decay_epochs, float weight_decay,
                             unsigned seed_val, bool verbose) {
            auto buf = targets.request();
            size_t n = static_cast<size_t>(buf.size);
            const float* ptr = static_cast<const float*>(buf.ptr);

            ReadoutConfig cfg;
            cfg.num_outputs    = num_outputs;
            cfg.task           = (std::strcmp(task, "classification") == 0)
                                     ? ReadoutTask::Classification
                                     : ReadoutTask::Regression;
            cfg.num_layers     = num_layers;
            cfg.conv_channels  = conv_channels;
            cfg.epochs         = epochs_val;
            cfg.batch_size     = batch_size;
            cfg.lr_max         = lr_max;
            cfg.lr_min_frac    = lr_min_frac;
            cfg.lr_decay_epochs = lr_decay_epochs;
            cfg.weight_decay   = weight_decay;
            cfg.seed           = seed_val;
            cfg.verbose        = verbose;

            size_t train_size = (cfg.task == ReadoutTask::Classification)
                                    ? n
                                    : n / static_cast<size_t>(cfg.num_outputs);
            if (train_size > self.NumCollected())
                throw std::invalid_argument(
                    "train_size exceeds num_collected ("
                    + std::to_string(self.NumCollected()) + ")");

            self.Train(ptr, train_size, cfg);
        },
            py::arg("targets"),
            py::arg("num_outputs")    = 1,
            py::arg("task")           = "regression",
            py::arg("num_layers")     = 0,
            py::arg("conv_channels")  = 16,
            py::arg("epochs")         = 200,
            py::arg("batch_size")     = 32,
            py::arg("lr_max")         = 0.005f,
            py::arg("lr_min_frac")    = 0.1f,
            py::arg("lr_decay_epochs") = 0,
            py::arg("weight_decay")   = 0.0f,
            py::arg("seed")           = 42u,
            py::arg("verbose")        = false,
            "Train HCNN readout on collected states.\n\n"
            "task: 'regression' or 'classification'.\n"
            "num_layers: Conv+Pool pairs (0 = auto from DIM).\n"
            "See ReadoutConfig for parameter details.")

        // ── Online (streaming) HCNN training ──
        .def("init_online", [](E& self,
                               py::array_t<float, py::array::c_style | py::array::forcecast> warmup_inputs,
                               int num_outputs, const char* task,
                               int num_layers, int conv_channels,
                               int batch_size, float lr_max,
                               unsigned seed_val) {
            auto buf = warmup_inputs.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumInputs();
            if (total % K != 0)
                throw std::invalid_argument("warmup_inputs size must be divisible by num_inputs");

            ReadoutConfig cfg;
            cfg.num_outputs   = num_outputs;
            cfg.task          = (std::strcmp(task, "classification") == 0)
                                    ? ReadoutTask::Classification
                                    : ReadoutTask::Regression;
            cfg.num_layers    = num_layers;
            cfg.conv_channels = conv_channels;
            cfg.batch_size    = batch_size;
            cfg.lr_max        = lr_max;
            cfg.seed          = seed_val;

            self.InitOnline(static_cast<const float*>(buf.ptr), total / K, cfg);
        },
            py::arg("warmup_inputs"),
            py::arg("num_outputs")   = 1,
            py::arg("task")          = "classification",
            py::arg("num_layers")    = 0,
            py::arg("conv_channels") = 16,
            py::arg("batch_size")    = 32,
            py::arg("lr_max")        = 0.005f,
            py::arg("seed")          = 42u,
            "Initialize HCNN for online (streaming) training.\n\n"
            "Runs warmup_inputs through reservoir, computes input standardization,\n"
            "builds CNN architecture. Call before train_live_step/train_live_batch.")

        .def("compute_target_centering", [](E& self,
                                            py::array_t<float, py::array::c_style | py::array::forcecast> targets) {
            auto buf = targets.request();
            size_t total = static_cast<size_t>(buf.size);
            size_t K = self.NumOutputs();
            if (total % K != 0)
                throw std::invalid_argument("targets size must be divisible by num_outputs");
            self.ComputeTargetCentering(static_cast<const float*>(buf.ptr), total / K);
        }, py::arg("targets"),
           "Compute per-output target centering for online regression.\n"
           "Call after init_online for regression tasks so that online training\n"
           "centers targets and predictions are de-centered (matching batch behavior).")

        .def("train_live_step", [](E& self, float target_class, float lr, float weight_decay) {
            self.TrainLiveStep(target_class, lr, weight_decay);
        },
            py::arg("target_class"), py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "Single-step online classification training on the live reservoir state.")

        .def("train_live_batch", [](E& self,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> states,
                                    py::array_t<int, py::array::c_style | py::array::forcecast> targets,
                                    float lr, float weight_decay) {
            auto sbuf = states.request();
            auto tbuf = targets.request();
            size_t count = static_cast<size_t>(tbuf.size);
            self.TrainLiveBatch(static_cast<const float*>(sbuf.ptr),
                                static_cast<const int*>(tbuf.ptr),
                                count, lr, weight_decay);
        },
            py::arg("states"), py::arg("targets"),
            py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "Mini-batch online classification training on pre-accumulated states.\n"
            "states: (count, num_output_verts) float array from copy_live_state.\n"
            "targets: (count,) int array of class indices.")

        .def("train_live_step_regression", [](E& self,
                                              py::array_t<float, py::array::c_style | py::array::forcecast> target,
                                              float lr, float weight_decay) {
            auto buf = target.request();
            self.TrainLiveStepRegression(static_cast<const float*>(buf.ptr), lr, weight_decay);
        },
            py::arg("target"), py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "Single-step online regression training on the live reservoir state.\n"
            "target: (num_outputs,) float array.")

        .def("train_live_batch_regression", [](E& self,
                                               py::array_t<float, py::array::c_style | py::array::forcecast> states,
                                               py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                                               float lr, float weight_decay) {
            auto sbuf = states.request();
            auto tbuf = targets.request();
            size_t K = self.NumOutputs();
            size_t count = static_cast<size_t>(tbuf.size) / K;
            self.TrainLiveBatchRegression(static_cast<const float*>(sbuf.ptr),
                                          static_cast<const float*>(tbuf.ptr),
                                          count, lr, weight_decay);
        },
            py::arg("states"), py::arg("targets"),
            py::arg("lr"), py::arg("weight_decay") = 0.0f,
            "Mini-batch online regression training on pre-accumulated states.\n"
            "states: (count, num_output_verts) float array from copy_live_state.\n"
            "targets: (count, num_outputs) float array.")

        .def("copy_live_state", [](const E& self) {
            size_t M = self.NumOutputVerts();
            py::array_t<float> arr(M);
            self.CopyLiveState(arr.mutable_data());
            return arr;
        }, "Copy the current subsampled reservoir state for external accumulation.\n"
           "Returns a (num_output_verts,) float array.")

        // ── Prediction & evaluation ──
        .def("predict_raw", [](const E& self, size_t timestep) {
            if (timestep >= self.NumCollected())
                throw std::out_of_range(
                    "timestep (" + std::to_string(timestep) +
                    ") >= num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.PredictRaw(timestep);
        }, py::arg("timestep"),
           "Return the raw continuous prediction for a collected timestep.")

        .def("predict_live_raw", [](const E& self) {
            return self.PredictLiveRaw();
        }, "Predict from the reservoir's current live state (no cached states needed).\n"
           "For autoregressive / streaming inference loops.")

        .def("predict_live_raw_multi", [](const E& self) {
            size_t K = self.NumOutputs();
            py::array_t<float> arr(K);
            self.PredictLiveRaw(arr.mutable_data());
            return arr;
        }, "Multi-output live predict: returns (num_outputs,) float array.")

        .def("r2", [](const E& self,
                      py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                      size_t start, size_t count) {
            if (start + count > self.NumCollected())
                throw std::out_of_range(
                    "start + count (" + std::to_string(start + count) +
                    ") > num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.R2(static_cast<const float*>(targets.request().ptr), start, count);
        }, py::arg("targets"), py::arg("start"), py::arg("count"),
           "Compute R-squared on a slice of collected states.")

        .def("nrmse", [](const E& self,
                         py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                         size_t start, size_t count) {
            if (start + count > self.NumCollected())
                throw std::out_of_range(
                    "start + count (" + std::to_string(start + count) +
                    ") > num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.NRMSE(static_cast<const float*>(targets.request().ptr), start, count);
        }, py::arg("targets"), py::arg("start"), py::arg("count"),
           "Compute Normalized RMSE on a slice of collected states.")

        .def("accuracy", [](const E& self,
                            py::array_t<float, py::array::c_style | py::array::forcecast> labels,
                            size_t start, size_t count) {
            if (start + count > self.NumCollected())
                throw std::out_of_range(
                    "start + count (" + std::to_string(start + count) +
                    ") > num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.Accuracy(static_cast<const float*>(labels.request().ptr), start, count);
        }, py::arg("labels"), py::arg("start"), py::arg("count"),
           "Compute classification accuracy on a slice of collected states.")

        // ── State access ──
        .def("selected_states", [](const E& self) {
            auto vec = self.SelectedStates();
            size_t M = self.NumOutputVerts();
            size_t T = self.NumCollected();
            py::array_t<float> arr({T, M});
            memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(float));
            return arr;
        }, "Return stride-selected states as a (num_collected, M) array.")

        .def("predictions", [](const E& self) {
            size_t T = self.NumCollected();
            py::array_t<float> arr(T);
            float* ptr = arr.mutable_data();
            for (size_t t = 0; t < T; ++t)
                ptr[t] = self.PredictRaw(t);
            return arr;
        }, "Return predictions for all collected timesteps as a 1D array.")

        // ── Properties ──
        .def_property_readonly("num_collected", &E::NumCollected)
        .def_property_readonly("num_outputs", &E::NumOutputs)
        .def_property_readonly("output_fraction", &E::OutputFraction)
        .def_property_readonly("num_output_verts", &E::NumOutputVerts)
        .def_property_readonly("dim", [](const E&) { return DIM; })
        .def_property_readonly("N", [](const E&) { return NN; })
        .def_property_readonly("num_inputs", &E::NumInputs)
        .def_property_readonly("seed", [](const E& self) { return self.GetConfig().seed; })
        .def_property_readonly("spectral_radius", [](const E& self) { return self.GetConfig().spectral_radius; })
        .def_property_readonly("leak_rate", [](const E& self) { return self.GetConfig().leak_rate; })
        .def_property_readonly("input_scaling", [](const E& self) { return self.GetConfig().input_scaling; })
        .def_property_readonly("alpha", [](const E& self) { return self.GetConfig().alpha; })

        // ── Persistence ──
        .def("_get_readout_state", [](const E& self) -> py::dict {
            auto s = self.GetReadoutState();
            py::dict d;
            d["is_trained"] = s.is_trained;
            d["bias"] = s.bias;
            d["weights"] = py::array_t<double>(
                {static_cast<py::ssize_t>(s.weights.size())}, s.weights.data());
            d["feature_mean"] = py::array_t<float>(
                {static_cast<py::ssize_t>(s.feature_mean.size())}, s.feature_mean.data());
            d["feature_scale"] = py::array_t<float>(
                {static_cast<py::ssize_t>(s.feature_scale.size())}, s.feature_scale.data());
            d["target_mean"] = py::array_t<double>(
                {static_cast<py::ssize_t>(s.target_mean.size())}, s.target_mean.data());
            return d;
        })
        .def("_set_readout_state", [](E& self, py::dict d) {
            typename E::ReadoutState s;
            s.is_trained = d["is_trained"].cast<bool>();
            s.bias = d["bias"].cast<double>();
            auto w = d["weights"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            s.weights.assign(w.data(), w.data() + w.size());
            auto fm = d["feature_mean"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            s.feature_mean.assign(fm.data(), fm.data() + fm.size());
            auto fs = d["feature_scale"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            s.feature_scale.assign(fs.data(), fs.data() + fs.size());
            if (d.contains("target_mean")) {
                auto tm = d["target_mean"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                s.target_mean.assign(tm.data(), tm.data() + tm.size());
            }
            self.SetReadoutState(s);
        })
        .def("set_cnn_config", [](E& self,
                                  int num_outputs, const char* task,
                                  int num_layers, int conv_channels) {
            ReadoutConfig cfg;
            cfg.num_outputs   = num_outputs;
            cfg.task          = (std::strcmp(task, "classification") == 0)
                                    ? ReadoutTask::Classification
                                    : ReadoutTask::Regression;
            cfg.num_layers    = num_layers;
            cfg.conv_channels = conv_channels;
            self.SetCNNConfig(cfg);
        },
            py::arg("num_outputs")   = 1,
            py::arg("task")          = "regression",
            py::arg("num_layers")    = 0,
            py::arg("conv_channels") = 16,
            "Pre-set CNN architecture config before restoring weights via _set_readout_state.\n"
            "Required when loading a saved HCNN model without training.")
        ;
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeRC: reservoir computing on Boolean hypercube graphs";

    bind_esn<5>(m,  "_ESN5");
    bind_esn<6>(m,  "_ESN6");
    bind_esn<7>(m,  "_ESN7");
    bind_esn<8>(m,  "_ESN8");
    bind_esn<9>(m,  "_ESN9");
    bind_esn<10>(m, "_ESN10");
    bind_esn<11>(m, "_ESN11");
    bind_esn<12>(m, "_ESN12");
    bind_esn<13>(m, "_ESN13");
    bind_esn<14>(m, "_ESN14");
    bind_esn<15>(m, "_ESN15");
    bind_esn<16>(m, "_ESN16");
}
