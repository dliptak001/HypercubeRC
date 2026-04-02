#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
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
                         float output_fraction,
                         ReadoutType readout_type, FeatureMode feature_mode) {
            ReservoirConfig cfg;
            cfg.seed             = seed;
            cfg.spectral_radius  = spectral_radius;
            cfg.input_scaling    = input_scaling;
            cfg.leak_rate        = leak_rate;
            cfg.alpha            = alpha;
            cfg.num_inputs       = num_inputs;
            cfg.output_fraction  = output_fraction;
            return std::make_unique<E>(cfg, readout_type, feature_mode);
        }),
            py::arg("seed")             = 0ULL,
            py::arg("spectral_radius")  = 0.9f,
            py::arg("input_scaling")    = 0.02f,
            py::arg("leak_rate")        = 1.0f,
            py::arg("alpha")            = 1.0f,
            py::arg("num_inputs")       = 1ULL,
            py::arg("output_fraction")  = 1.0f,
            py::arg("readout_type")     = ReadoutType::Ridge,
            py::arg("feature_mode")     = FeatureMode::Translated)

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

        // ── Training ──
        .def("train", [](E& self,
                         py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                         py::object reg,
                         py::object lr,
                         py::object epochs,
                         float weight_decay,
                         float lr_decay) {
            auto buf = targets.request();
            size_t n = static_cast<size_t>(buf.size);
            const float* ptr = static_cast<const float*>(buf.ptr);

            if (n > self.NumCollected())
                throw std::invalid_argument(
                    "train_size (" + std::to_string(n) +
                    ") exceeds num_collected (" + std::to_string(self.NumCollected()) + ")");

            if (!lr.is_none()) {
                // Linear SGD path
                if (self.GetReadoutType() != ReadoutType::Linear)
                    throw std::invalid_argument(
                        "lr/epochs parameters require ReadoutType.Linear");
                float lr_val = lr.cast<float>();
                size_t ep = epochs.is_none() ? 200 : epochs.cast<size_t>();
                self.Train(ptr, n, lr_val, ep, weight_decay, lr_decay);
            } else if (!reg.is_none()) {
                // Ridge with custom lambda
                if (self.GetReadoutType() != ReadoutType::Ridge)
                    throw std::invalid_argument(
                        "reg parameter requires ReadoutType.Ridge");
                self.Train(ptr, n, reg.cast<double>());
            } else {
                // Default parameters
                self.Train(ptr, n);
            }
        },
            py::arg("targets"),
            py::arg("reg")     = py::none(),
            py::arg("lr")          = py::none(),
            py::arg("epochs")      = py::none(),
            py::arg("weight_decay") = 1e-4f,
            py::arg("lr_decay")    = 0.01f,
            "Train the readout on collected states.\n\n"
            "Default: uses default parameters for the selected readout type.\n"
            "Ridge: pass reg for custom regularization.\n"
            "Linear: pass lr (and optionally epochs) for custom SGD.")

        .def("train_incremental", [](E& self,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> targets,
                                     float blend, float lr, size_t epochs,
                                     float weight_decay, float lr_decay) {
            if (self.GetReadoutType() != ReadoutType::Linear)
                throw std::invalid_argument(
                    "train_incremental() requires ReadoutType.Linear");
            auto buf = targets.request();
            size_t n = static_cast<size_t>(buf.size);
            if (n > self.NumCollected())
                throw std::invalid_argument(
                    "train_size (" + std::to_string(n) +
                    ") exceeds num_collected (" + std::to_string(self.NumCollected()) + ")");
            self.TrainIncremental(static_cast<const float*>(buf.ptr),
                                 n, blend, lr, epochs, weight_decay, lr_decay);
        },
            py::arg("targets"),
            py::arg("blend")        = 0.1f,
            py::arg("lr")           = 0.0f,
            py::arg("epochs")       = 200,
            py::arg("weight_decay") = 1e-4f,
            py::arg("lr_decay")     = 0.01f,
            "Incrementally update the Linear readout for streaming applications.")

        // ── Prediction & evaluation ──
        .def("predict_raw", [](const E& self, size_t timestep) {
            if (timestep >= self.NumCollected())
                throw std::out_of_range(
                    "timestep (" + std::to_string(timestep) +
                    ") >= num_collected (" + std::to_string(self.NumCollected()) + ")");
            return self.PredictRaw(timestep);
        }, py::arg("timestep"),
           "Return the raw continuous prediction for a collected timestep.")

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

        // ── State & feature access ──
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
        .def_property_readonly("num_features", &E::NumFeatures)
        .def_property_readonly("output_fraction", &E::OutputFraction)
        .def_property_readonly("output_stride", &E::OutputStride)
        .def_property_readonly("num_output_verts", &E::NumOutputVerts)
        .def_property_readonly("readout_type", &E::GetReadoutType)
        .def_property_readonly("feature_mode", &E::GetFeatureMode)
        .def_property_readonly("alpha", &E::GetAlpha)
        .def_property_readonly("dim", [](const E&) { return DIM; })
        .def_property_readonly("N", [](const E&) { return NN; })
        .def_property_readonly("num_inputs", &E::NumInputs)
        .def_property_readonly("seed", [](const E& self) { return self.GetConfig().seed; })
        .def_property_readonly("spectral_radius", [](const E& self) { return self.GetConfig().spectral_radius; })
        .def_property_readonly("leak_rate", [](const E& self) { return self.GetConfig().leak_rate; })
        .def_property_readonly("input_scaling", [](const E& self) { return self.GetConfig().input_scaling; })

        // ── Persistence (private, used by Python __getstate__/__setstate__) ──
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
            self.SetReadoutState(s);
        })
        ;
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = "HypercubeRC: reservoir computing on Boolean hypercube graphs";

    py::enum_<ReadoutType>(m, "ReadoutType")
        .value("Linear", ReadoutType::Linear,
               "Online SGD with L2 decay and pocket selection. Supports streaming.")
        .value("Ridge", ReadoutType::Ridge,
               "Closed-form Ridge regression. Deterministic, fast, optimal.");

    py::enum_<FeatureMode>(m, "FeatureMode")
        .value("Raw", FeatureMode::Raw,
               "Use selected states directly as features.")
        .value("Translated", FeatureMode::Translated,
               "Expand selected states via [x | x^2 | x*x_antipodal] (2.5x features).");

    bind_esn<5>(m,  "_ESN5");
    bind_esn<6>(m,  "_ESN6");
    bind_esn<7>(m,  "_ESN7");
    bind_esn<8>(m,  "_ESN8");
    bind_esn<9>(m,  "_ESN9");
    bind_esn<10>(m, "_ESN10");
    bind_esn<11>(m, "_ESN11");
    bind_esn<12>(m, "_ESN12");
}
