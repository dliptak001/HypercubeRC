"""Smoke tests for hypercube_rc Python bindings."""

import pickle

import numpy as np
import pytest

import hypercube_rc as hrc
from hypercube_rc import ESN


# ── Shared fixtures (trained once per module) ──

@pytest.fixture(scope="module")
def sine_signal():
    return np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)


@pytest.fixture(scope="module")
def trained_esn(sine_signal):
    esn = ESN(dim=6, seed=42)
    esn.fit(sine_signal, warmup=200)
    return esn, sine_signal


def _short_signal(n=500):
    return np.sin(np.linspace(0, 10 * np.pi, n)).astype(np.float32)


# ── Construction ──

class TestConstruction:

    @pytest.mark.parametrize("dim", range(5, 13))
    def test_all_dims(self, dim):
        esn = ESN(dim=dim, seed=1)
        assert esn.dim == dim
        assert esn.N == 2**dim
        assert esn.num_collected == 0

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be 5-16"):
            ESN(dim=4)
        with pytest.raises(ValueError, match="dim must be 5-16"):
            ESN(dim=17)

    def test_defaults(self):
        esn = ESN(dim=5)
        assert esn.num_inputs == 1
        assert esn.output_fraction == pytest.approx(1.0)
        assert esn.alpha == pytest.approx(1.0)

    def test_custom_config(self):
        esn = ESN(
            dim=6, seed=42, spectral_radius=0.95, input_scaling=0.05,
            leak_rate=0.8, alpha=1.5, output_fraction=0.5,
        )
        assert esn.alpha == pytest.approx(1.5)

    def test_repr(self):
        esn = ESN(dim=5)
        r = repr(esn)
        assert "dim=5" in r
        assert "N=32" in r


# ── Prediction (reuses shared trained ESN) ──

class TestSinePrediction:

    def test_r2(self, trained_esn):
        esn, _ = trained_esn
        r2 = esn.r2()
        assert r2 > 0.90, f"R² too low: {r2}"

    def test_nrmse(self, trained_esn):
        esn, _ = trained_esn
        nrmse = esn.nrmse()
        assert nrmse < 0.5, f"NRMSE too high: {nrmse}"

    def test_predictions_array(self, trained_esn):
        esn, _ = trained_esn
        preds = esn.predictions()
        assert preds.shape == (esn.num_collected,)
        assert preds.dtype == np.float32


# ── Multi-input ──

class TestMultiInput:

    def test_two_inputs(self):
        esn = ESN(dim=5, seed=42, num_inputs=2)
        assert esn.num_inputs == 2
        n_steps = 300
        inputs = np.random.randn(n_steps, 2).astype(np.float32) * 0.1
        esn.warmup(inputs[:100])
        esn.run(inputs[100:])
        assert esn.num_collected == 200


# ── State access ──

class TestStateAccess:

    def test_selected_states_shape(self):
        esn = ESN(dim=5, seed=1)
        signal = _short_signal()
        esn.warmup(signal[:100])
        esn.run(signal[100:])
        states = esn.selected_states()
        assert states.shape == (esn.num_collected, esn.num_output_verts)
        assert states.dtype == np.float32

    def test_clear_states(self):
        esn = ESN(dim=5, seed=1)
        signal = _short_signal()
        esn.run(signal[:200])
        assert esn.num_collected == 200
        esn.clear_states()
        assert esn.num_collected == 0


# ── Classification (own train — different code path) ──

class TestClassification:

    def test_square_wave_classification(self):
        t = np.linspace(0, 20 * np.pi, 2000)
        signal = np.sin(t).astype(np.float32)
        labels = np.where(signal >= 0, 1.0, 0.0).astype(np.float32)

        esn = ESN(dim=5, seed=42)
        esn.warmup(signal[:200])
        esn.run(signal[200:])
        esn.train_cnn(labels[200:1600], num_outputs=2, task="classification",
                       epochs=50, batch_size=32)
        acc = esn.accuracy(labels[200:], 1400, 400)
        assert acc > 0.8, f"Accuracy too low: {acc}"


# ── fit() convenience (reuses shared ESN where possible) ──

class TestFit:

    def test_fit_sine_prediction(self, trained_esn):
        esn, _ = trained_esn
        assert esn.train_size == int(esn.num_collected * 0.7)
        assert esn.test_size == esn.num_collected - esn.train_size
        r2 = esn.r2()
        assert r2 > 0.90

    def test_fit_with_train_size(self, sine_signal):
        esn = ESN(dim=5, seed=42)
        esn.fit(sine_signal, warmup=200, train_size=1400)
        assert esn.train_size == 1400

    def test_fit_with_train_frac(self, sine_signal):
        esn = ESN(dim=5, seed=42)
        esn.fit(sine_signal, warmup=200, train_frac=0.8)
        expected = int(esn.num_collected * 0.8)
        assert esn.train_size == expected

    def test_fit_returns_self(self):
        signal = _short_signal()
        esn = ESN(dim=5, seed=1)
        result = esn.fit(signal, warmup=100)
        assert result is esn

    def test_fit_nrmse_no_args(self, trained_esn):
        esn, _ = trained_esn
        nrmse = esn.nrmse()
        assert nrmse < 0.5

    def test_fit_horizon(self):
        signal = _short_signal(1000)
        esn = ESN(dim=5, seed=42)
        esn.fit(signal, warmup=200, horizon=2)
        assert esn.num_collected > 0

    def test_fit_explicit_targets(self):
        signal = _short_signal(1000)
        esn = ESN(dim=5, seed=42)
        esn.fit(signal[:-1], targets=signal[201:], warmup=200)
        assert esn.num_collected > 0

    def test_fit_multi_input_explicit_targets(self):
        n = 1000
        ch0 = np.sin(np.linspace(0, 10 * np.pi, n)).astype(np.float32)
        ch1 = np.cos(np.linspace(0, 10 * np.pi, n)).astype(np.float32)
        inputs = np.column_stack([ch0, ch1])
        targets = ch0[201:]
        esn = ESN(dim=5, seed=42, num_inputs=2)
        esn.fit(inputs[:-1], targets=targets, warmup=200)
        assert esn.num_collected > 0

    def test_fit_multi_input_requires_targets(self):
        inputs = np.ones((100, 2), dtype=np.float32)
        esn = ESN(dim=5, seed=1, num_inputs=2)
        with pytest.raises(ValueError, match="targets must be provided"):
            esn.fit(inputs, warmup=50)

    def test_fit_clears_previous_state(self):
        signal = _short_signal()
        esn = ESN(dim=5, seed=1)
        esn.fit(signal[:300], warmup=100)
        first_collected = esn.num_collected
        esn.fit(signal[:400], warmup=100)
        assert esn.num_collected != first_collected

    def test_fit_both_train_size_and_frac_raises(self):
        signal = _short_signal()
        esn = ESN(dim=5, seed=1)
        with pytest.raises(ValueError, match="not both"):
            esn.fit(signal, warmup=100, train_size=100, train_frac=0.5)

    def test_clear_states_clears_fit_targets(self):
        signal = _short_signal()
        esn = ESN(dim=5, seed=1)
        esn.fit(signal, warmup=100)
        assert esn.train_size is not None
        esn.clear_states()
        assert esn.train_size is None
        assert esn.test_size is None


# ── Eval defaults (reuses shared trained ESN) ──

class TestEvalDefaults:

    def test_r2_no_args_after_fit(self, trained_esn):
        esn, signal = trained_esn
        r2_default = esn.r2()
        r2_explicit = esn.r2(signal[201:], start=1400)
        assert abs(r2_default - r2_explicit) < 1e-6

    def test_r2_explicit_start_only(self, trained_esn):
        esn, signal = trained_esn
        r2 = esn.r2(signal[201:], start=1400)
        assert r2 > 0.90

    def test_r2_no_args_without_fit_raises(self):
        esn = ESN(dim=5, seed=1)
        signal = _short_signal()
        esn.run(signal[:200])
        esn.train(signal[:200])
        with pytest.raises(ValueError, match="No targets"):
            esn.r2()

    def test_sliced_targets_caught(self, trained_esn):
        esn, signal = trained_esn
        with pytest.raises(ValueError, match="do not slice"):
            esn.r2(signal[1401 + 200:], start=1400)

    def test_r2_all_collected(self, trained_esn):
        esn, signal = trained_esn
        r2_all = esn.r2(signal[201:])
        assert r2_all > 0.90


# ── Output fraction ──

class TestOutputFraction:

    def test_half_fraction(self):
        esn_full = ESN(dim=6, seed=1, output_fraction=1.0)
        esn_half = ESN(dim=6, seed=1, output_fraction=0.5)
        assert esn_half.num_output_verts < esn_full.num_output_verts


# ── Error handling (DIM 5, fast) ──

class TestErrorHandling:

    def test_predict_raw_out_of_range(self):
        esn = ESN(dim=5, seed=1)
        signal = _short_signal()
        esn.run(signal[:100])
        esn.train(signal[:100])
        with pytest.raises((IndexError, RuntimeError)):
            esn.predict_raw(100)

    def test_r2_out_of_range(self):
        esn = ESN(dim=5, seed=1)
        signal = _short_signal()
        esn.run(signal[:100])
        esn.train(signal[:100])
        with pytest.raises((IndexError, RuntimeError)):
            esn.r2(signal[:200], 50, 60)

    def test_train_exceeds_collected(self):
        esn = ESN(dim=5, seed=1)
        signal = _short_signal()
        esn.run(signal[:100])
        with pytest.raises(Exception, match="exceeds"):
            esn.train(signal[:200])

    def test_warmup_wrong_num_inputs(self):
        esn = ESN(dim=5, seed=1, num_inputs=3)
        signal = np.ones(10, dtype=np.float32)
        with pytest.raises(Exception, match="divisible"):
            esn.warmup(signal)


# ── Persistence (reuses shared trained ESN) ──

class TestPersistence:

    def test_pickle_roundtrip(self, trained_esn, sine_signal):
        esn, _ = trained_esn
        r2_before = esn.r2()

        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.num_collected == 0

        loaded.warmup(sine_signal[:200])
        loaded.run(sine_signal[200:-1])
        r2_after = loaded.r2(sine_signal[201:], start=1400)
        assert abs(r2_before - r2_after) < 1e-5

    def test_save_load(self, trained_esn, sine_signal, tmp_path):
        esn, _ = trained_esn
        r2_before = esn.r2()

        path = tmp_path / "model.pkl"
        esn.save(path)
        loaded = ESN.load(path)

        loaded.warmup(sine_signal[:200])
        loaded.run(sine_signal[200:-1])
        r2_after = loaded.r2(sine_signal[201:], start=1400)
        assert abs(r2_before - r2_after) < 1e-5

    def test_save_untrained(self, tmp_path):
        esn = ESN(dim=5, seed=99)
        path = tmp_path / "untrained.pkl"
        esn.save(path)
        loaded = ESN.load(path)
        assert loaded.dim == 5
        assert loaded.num_collected == 0

    def test_pickle_preserves_config(self):
        esn = ESN(dim=8, seed=123, spectral_radius=0.85,
                  input_scaling=0.05, leak_rate=0.7, alpha=1.2,
                  num_inputs=2, output_fraction=0.5)
        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.dim == 8
        assert loaded.seed == 123
        assert loaded.spectral_radius == pytest.approx(0.85)
        assert loaded.input_scaling == pytest.approx(0.05)
        assert loaded.leak_rate == pytest.approx(0.7)
        assert loaded.alpha == pytest.approx(1.2)
        assert loaded.num_inputs == 2
        assert loaded.output_fraction == pytest.approx(0.5)

    def test_collected_states_not_persisted(self, trained_esn):
        esn, _ = trained_esn
        assert esn.num_collected > 0
        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.num_collected == 0

    def test_load_wrong_type_raises(self, tmp_path):
        path = tmp_path / "not_esn.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "an ESN"}, f)
        with pytest.raises(TypeError, match="Expected ESN"):
            ESN.load(path)
