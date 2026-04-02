"""Smoke tests for hypercube_rc Python bindings."""

import pickle

import numpy as np
import pytest

import hypercube_rc as hrc
from hypercube_rc import ESN, ReadoutType, FeatureMode


def _sine_signal(n=2000):
    return np.sin(np.linspace(0, 20 * np.pi, n)).astype(np.float32)


class TestConstruction:
    """Test ESN construction for all supported dimensions."""

    @pytest.mark.parametrize("dim", range(5, 13))
    def test_all_dims(self, dim):
        esn = ESN(dim=dim, seed=1)
        assert esn.dim == dim
        assert esn.N == 2**dim
        assert esn.num_collected == 0

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be 5-12"):
            ESN(dim=4)
        with pytest.raises(ValueError, match="dim must be 5-12"):
            ESN(dim=13)

    def test_defaults(self):
        esn = ESN(dim=5)
        assert esn.readout_type == ReadoutType.Ridge
        assert esn.feature_mode == FeatureMode.Translated
        assert esn.num_inputs == 1
        assert esn.output_fraction == pytest.approx(1.0)
        assert esn.alpha == pytest.approx(1.0)

    def test_custom_config(self):
        esn = ESN(
            dim=6,
            seed=42,
            spectral_radius=0.95,
            input_scaling=0.05,
            leak_rate=0.8,
            alpha=1.5,
            output_fraction=0.5,
            readout_type=ReadoutType.Linear,
            feature_mode=FeatureMode.Raw,
        )
        assert esn.readout_type == ReadoutType.Linear
        assert esn.feature_mode == FeatureMode.Raw
        assert esn.alpha == pytest.approx(1.5)

    def test_repr(self):
        esn = ESN(dim=5)
        r = repr(esn)
        assert "dim=5" in r
        assert "N=32" in r


class TestSinePrediction:
    """Test the standard sine wave prediction pipeline."""

    def test_ridge_prediction(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.warmup(signal[:200])
        esn.run(signal[200:-1])
        targets = signal[201:]
        train_size = 1400
        test_size = esn.num_collected - train_size

        esn.train(targets[:train_size])
        r2 = esn.r2(targets, train_size, test_size)
        assert r2 > 0.99, f"Ridge R² too low: {r2}"

    def test_ridge_custom_lambda(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.warmup(signal[:200])
        esn.run(signal[200:-1])
        targets = signal[201:]
        esn.train(targets[:1400], reg=0.1)
        r2 = esn.r2(targets, 1400, esn.num_collected - 1400)
        assert r2 > 0.95

    def test_linear_prediction(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42, readout_type=ReadoutType.Linear)
        esn.warmup(signal[:200])
        esn.run(signal[200:-1])
        targets = signal[201:]
        esn.train(targets[:1400], lr=0.0, epochs=300)
        r2 = esn.r2(targets, 1400, esn.num_collected - 1400)
        assert r2 > 0.8, f"Linear R² too low: {r2}"

    def test_nrmse(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.warmup(signal[:200])
        esn.run(signal[200:-1])
        targets = signal[201:]
        esn.train(targets[:1400])
        nrmse = esn.nrmse(targets, 1400, esn.num_collected - 1400)
        assert nrmse < 0.1, f"NRMSE too high: {nrmse}"

    def test_predictions_array(self):
        signal = _sine_signal(500)
        esn = ESN(dim=5, seed=1)
        esn.warmup(signal[:100])
        esn.run(signal[100:-1])
        esn.train(signal[101:])
        preds = esn.predictions()
        assert preds.shape == (esn.num_collected,)
        assert preds.dtype == np.float32


class TestFeatureModes:
    """Test Raw vs Translated feature modes."""

    def test_raw_mode(self):
        esn = ESN(dim=5, seed=1, feature_mode=FeatureMode.Raw)
        signal = _sine_signal(500)
        esn.warmup(signal[:100])
        esn.run(signal[100:-1])
        # Raw mode: num_features == num_output_verts
        assert esn.num_features == esn.num_output_verts

    def test_translated_mode(self):
        esn = ESN(dim=5, seed=1, feature_mode=FeatureMode.Translated)
        signal = _sine_signal(500)
        esn.warmup(signal[:100])
        esn.run(signal[100:-1])
        # Translated mode: num_features == 2.5 * num_output_verts
        M = esn.num_output_verts
        assert esn.num_features == M + M + M // 2


class TestMultiInput:
    """Test multi-input ESN."""

    def test_two_inputs(self):
        esn = ESN(dim=6, seed=42, num_inputs=2)
        assert esn.num_inputs == 2
        n_steps = 300
        inputs = np.random.randn(n_steps, 2).astype(np.float32) * 0.1
        esn.warmup(inputs[:100])
        esn.run(inputs[100:])
        assert esn.num_collected == 200


class TestStateAccess:
    """Test state and feature access methods."""

    def test_selected_states_shape(self):
        esn = ESN(dim=5, seed=1)
        signal = _sine_signal(500)
        esn.warmup(signal[:100])
        esn.run(signal[100:])
        states = esn.selected_states()
        assert states.shape == (esn.num_collected, esn.num_output_verts)
        assert states.dtype == np.float32

    def test_clear_states(self):
        esn = ESN(dim=5, seed=1)
        signal = _sine_signal(500)
        esn.run(signal[:200])
        assert esn.num_collected == 200
        esn.clear_states()
        assert esn.num_collected == 0


class TestClassification:
    """Test classification accuracy."""

    def test_square_wave_classification(self):
        # Generate a square wave from sine
        t = np.linspace(0, 20 * np.pi, 2000)
        signal = np.sin(t).astype(np.float32)
        labels = np.where(signal >= 0, 1.0, -1.0).astype(np.float32)

        esn = ESN(dim=6, seed=42)
        esn.warmup(signal[:200])
        esn.run(signal[200:])
        esn.train(labels[200:1600])
        acc = esn.accuracy(labels[200:], 1400, 400)
        assert acc > 0.9, f"Accuracy too low: {acc}"


class TestFit:
    """Test the fit() convenience method."""

    def test_fit_sine_prediction(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200)
        # Default train_frac=0.7
        assert esn.train_size == int(esn.num_collected * 0.7)
        assert esn.test_size == esn.num_collected - esn.train_size
        r2 = esn.r2()  # no args — evaluates test portion
        assert r2 > 0.99, f"R² too low: {r2}"

    def test_fit_with_train_size(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200, train_size=1400)
        assert esn.train_size == 1400
        r2 = esn.r2()
        assert r2 > 0.99

    def test_fit_with_train_frac(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200, train_frac=0.8)
        expected = int(esn.num_collected * 0.8)
        assert esn.train_size == expected

    def test_fit_returns_self(self):
        signal = _sine_signal()
        esn = ESN(dim=5, seed=1)
        result = esn.fit(signal, warmup=100)
        assert result is esn

    def test_fit_nrmse_no_args(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200)
        nrmse = esn.nrmse()
        assert nrmse < 0.1

    def test_fit_horizon(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200, horizon=2)
        # horizon=2 means predicting 2 steps ahead — still works
        assert esn.num_collected > 0
        r2 = esn.r2()
        assert r2 > 0.5  # weaker bound for 2-step prediction

    def test_fit_explicit_targets(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        # Explicit targets: run(inputs[warmup:]) collects len-warmup states,
        # so targets must have len-warmup elements. Use next-step: target[i]
        # corresponds to the input one step after collected state i.
        warmup = 200
        run_inputs = signal[warmup:-1]       # 1799 steps
        targets = signal[warmup + 1:]        # 1799 targets (next-step)
        esn.fit(signal[:-1], targets=targets, warmup=warmup)
        r2 = esn.r2()
        assert r2 > 0.99

    def test_fit_multi_input_explicit_targets(self):
        np.random.seed(42)
        n = 2000
        ch0 = np.sin(np.linspace(0, 20 * np.pi, n)).astype(np.float32)
        ch1 = np.cos(np.linspace(0, 20 * np.pi, n)).astype(np.float32)
        inputs = np.column_stack([ch0, ch1])
        # run(inputs[200:]) collects 1800 states, need 1800 targets.
        # Predict next ch0 value: target[i] = ch0[200 + i + 1]
        targets = ch0[201:]                  # 1799 elements
        # Trim inputs so run collects 1799 states to match targets
        esn = ESN(dim=6, seed=42, num_inputs=2)
        esn.fit(inputs[:-1], targets=targets, warmup=200)
        r2 = esn.r2()
        assert r2 > 0.8, f"Multi-input R² too low: {r2}"

    def test_fit_multi_input_requires_targets(self):
        inputs = np.ones((100, 2), dtype=np.float32)
        esn = ESN(dim=5, seed=1, num_inputs=2)
        with pytest.raises(ValueError, match="targets must be provided"):
            esn.fit(inputs, warmup=50)

    def test_fit_clears_previous_state(self):
        signal = _sine_signal()
        esn = ESN(dim=5, seed=1)
        esn.fit(signal[:500], warmup=100)
        first_collected = esn.num_collected
        esn.fit(signal[:800], warmup=100)
        # Should have cleared and re-collected
        assert esn.num_collected != first_collected

    def test_fit_both_train_size_and_frac_raises(self):
        signal = _sine_signal()
        esn = ESN(dim=5, seed=1)
        with pytest.raises(ValueError, match="not both"):
            esn.fit(signal, warmup=100, train_size=100, train_frac=0.5)

    def test_clear_states_clears_fit_targets(self):
        signal = _sine_signal()
        esn = ESN(dim=5, seed=1)
        esn.fit(signal, warmup=100)
        assert esn.train_size is not None
        esn.clear_states()
        assert esn.train_size is None
        assert esn.test_size is None


class TestEvalDefaults:
    """Test that r2/nrmse/accuracy work with default args and catch footguns."""

    def test_r2_no_args_after_fit(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200, train_size=1400)
        # No args — evaluates test portion automatically
        r2_default = esn.r2()
        # Explicit args — same result
        r2_explicit = esn.r2(signal[201:], start=1400)
        assert abs(r2_default - r2_explicit) < 1e-6

    def test_r2_explicit_start_only(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.warmup(signal[:200])
        esn.run(signal[200:-1])
        targets = signal[201:]
        esn.train(targets[:1400])
        # start=1400, count defaults to all remaining
        r2 = esn.r2(targets, start=1400)
        assert r2 > 0.99

    def test_r2_no_args_without_fit_raises(self):
        esn = ESN(dim=5, seed=1)
        signal = _sine_signal(500)
        esn.run(signal[:200])
        esn.train(signal[:200])
        with pytest.raises(ValueError, match="No targets"):
            esn.r2()

    def test_sliced_targets_caught(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.warmup(signal[:200])
        esn.run(signal[200:-1])
        targets = signal[201:]
        esn.train(targets[:1400])
        # THE FOOTGUN: slicing targets then passing start > 0
        with pytest.raises(ValueError, match="do not slice"):
            esn.r2(targets[1400:], start=1400)  # array too short for start+count

    def test_r2_all_collected(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.warmup(signal[:200])
        esn.run(signal[200:-1])
        targets = signal[201:]
        esn.train(targets[:1400])
        # Evaluate over ALL collected states (train + test)
        r2_all = esn.r2(targets)
        assert r2_all > 0.99


class TestOutputFraction:
    """Test output_fraction parameter."""

    def test_half_fraction(self):
        esn_full = ESN(dim=6, seed=1, output_fraction=1.0)
        esn_half = ESN(dim=6, seed=1, output_fraction=0.5)
        assert esn_half.num_output_verts < esn_full.num_output_verts
        assert esn_half.num_features < esn_full.num_features


class TestErrorHandling:
    """Test that misuse raises clear Python exceptions instead of crashing."""

    def test_train_reg_on_linear_esn(self):
        esn = ESN(dim=5, seed=1, readout_type=ReadoutType.Linear)
        signal = _sine_signal(500)
        esn.run(signal[:200])
        with pytest.raises(Exception, match="Ridge"):
            esn.train(signal[:200], reg=1.0)

    def test_train_lr_on_ridge_esn(self):
        esn = ESN(dim=5, seed=1, readout_type=ReadoutType.Ridge)
        signal = _sine_signal(500)
        esn.run(signal[:200])
        with pytest.raises(Exception, match="Linear"):
            esn.train(signal[:200], lr=0.01)

    def test_train_incremental_on_ridge_esn(self):
        esn = ESN(dim=5, seed=1, readout_type=ReadoutType.Ridge)
        signal = _sine_signal(500)
        esn.run(signal[:200])
        with pytest.raises(Exception, match="Linear"):
            esn.train_incremental(signal[:200])

    def test_predict_raw_out_of_range(self):
        esn = ESN(dim=5, seed=1)
        signal = _sine_signal(500)
        esn.run(signal[:100])
        esn.train(signal[:100])
        with pytest.raises((IndexError, RuntimeError)):
            esn.predict_raw(100)  # valid range is 0-99

    def test_r2_out_of_range(self):
        esn = ESN(dim=5, seed=1)
        signal = _sine_signal(500)
        esn.run(signal[:100])
        esn.train(signal[:100])
        with pytest.raises((IndexError, RuntimeError)):
            esn.r2(signal[:200], 50, 60)  # 50+60=110 > 100

    def test_train_exceeds_collected(self):
        esn = ESN(dim=5, seed=1)
        signal = _sine_signal(500)
        esn.run(signal[:100])
        with pytest.raises(Exception, match="exceeds"):
            esn.train(signal[:200])  # 200 > 100 collected

    def test_warmup_wrong_num_inputs(self):
        esn = ESN(dim=5, seed=1, num_inputs=3)
        signal = np.ones(10, dtype=np.float32)  # 10 not divisible by 3
        with pytest.raises(Exception, match="divisible"):
            esn.warmup(signal)


class TestPersistence:
    """Test save/load and pickle support."""

    def test_pickle_roundtrip_ridge(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200)
        r2_before = esn.r2()

        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.num_collected == 0

        # Re-drive with same data
        loaded.warmup(signal[:200])
        loaded.run(signal[200:-1])
        r2_after = loaded.r2(signal[201:], start=1400)
        assert abs(r2_before - r2_after) < 1e-6

    def test_pickle_roundtrip_linear(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42, readout_type=ReadoutType.Linear)
        esn.fit(signal, warmup=200)
        r2_before = esn.r2()

        loaded = pickle.loads(pickle.dumps(esn))
        loaded.warmup(signal[:200])
        loaded.run(signal[200:-1])
        r2_after = loaded.r2(signal[201:], start=1400)
        assert abs(r2_before - r2_after) < 1e-4  # looser for SGD

    def test_save_load(self, tmp_path):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200)
        r2_before = esn.r2()

        path = tmp_path / "model.pkl"
        esn.save(path)
        loaded = ESN.load(path)

        loaded.warmup(signal[:200])
        loaded.run(signal[200:-1])
        r2_after = loaded.r2(signal[201:], start=1400)
        assert abs(r2_before - r2_after) < 1e-6

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
                  num_inputs=2, output_fraction=0.5,
                  readout_type=ReadoutType.Linear,
                  feature_mode=FeatureMode.Raw)
        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.dim == 8
        assert loaded.seed == 123
        assert loaded.spectral_radius == pytest.approx(0.85)
        assert loaded.input_scaling == pytest.approx(0.05)
        assert loaded.leak_rate == pytest.approx(0.7)
        assert loaded.alpha == pytest.approx(1.2)
        assert loaded.num_inputs == 2
        assert loaded.output_fraction == pytest.approx(0.5)
        assert loaded.readout_type == ReadoutType.Linear
        assert loaded.feature_mode == FeatureMode.Raw

    def test_collected_states_not_persisted(self):
        signal = _sine_signal()
        esn = ESN(dim=6, seed=42)
        esn.fit(signal, warmup=200)
        assert esn.num_collected > 0
        loaded = pickle.loads(pickle.dumps(esn))
        assert loaded.num_collected == 0

    def test_load_wrong_type_raises(self, tmp_path):
        path = tmp_path / "not_esn.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "an ESN"}, f)
        with pytest.raises(TypeError, match="Expected ESN"):
            ESN.load(path)
