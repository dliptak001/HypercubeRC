"""Smoke tests for hypercube_rc Python bindings."""

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
        esn.train(targets[:1400], lambda_=0.1)
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


class TestOutputFraction:
    """Test output_fraction parameter."""

    def test_half_fraction(self):
        esn_full = ESN(dim=6, seed=1, output_fraction=1.0)
        esn_half = ESN(dim=6, seed=1, output_fraction=0.5)
        assert esn_half.num_output_verts < esn_full.num_output_verts
        assert esn_half.num_features < esn_full.num_features
