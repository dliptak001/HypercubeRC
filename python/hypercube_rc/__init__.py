"""HypercubeRC: reservoir computing on Boolean hypercube graphs.

This package provides Python bindings for the HypercubeRC C++ library.
The reservoir topology is a Boolean hypercube graph with N = 2^dim neurons.

Quick start::

    import numpy as np
    import hypercube_rc as hrc

    signal = np.sin(np.linspace(0, 20 * np.pi, 2000)).astype(np.float32)
    esn = hrc.ESN(dim=7, seed=42)
    esn.warmup(signal[:200])
    esn.run(signal[200:-1])
    esn.train(signal[201:])
    print(f"R² = {esn.r2(signal[201:], 0, esn.num_collected)}")
"""

from __future__ import annotations

import numpy as np

from ._core import ReadoutType, FeatureMode
from ._core import (
    _ESN5, _ESN6, _ESN7, _ESN8, _ESN9, _ESN10, _ESN11, _ESN12,
)

__version__ = "0.1.0"
__all__ = ["ESN", "ReadoutType", "FeatureMode"]

_ESN_CLASSES = {
    5: _ESN5, 6: _ESN6, 7: _ESN7, 8: _ESN8,
    9: _ESN9, 10: _ESN10, 11: _ESN11, 12: _ESN12,
}


def _to_float32(arr):
    """Ensure array is C-contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


class ESN:
    """Echo State Network on a Boolean hypercube reservoir.

    The reservoir has N = 2^dim neurons arranged on the vertices of a
    dim-dimensional Boolean hypercube. Connectivity is computed via XOR
    addressing with zero storage overhead.

    Parameters
    ----------
    dim : int
        Hypercube dimension (5-12). Determines the number of neurons: N = 2^dim.
    seed : int
        RNG seed for weight initialization. Default: 0.
    spectral_radius : float
        Target spectral radius for the recurrent weight matrix. Default: 0.9.
        Scale-invariant across all dim values.
    input_scaling : float
        Magnitude of input weights, drawn from U(-input_scaling, +input_scaling).
        Default: 0.02. Scale-invariant across all dim values.
    leak_rate : float
        Leaky integrator coefficient. 1.0 = full replacement (default),
        < 1.0 adds temporal smoothing.
    alpha : float
        Gain applied inside tanh activation: tanh(alpha * sum). Default: 1.0.
    num_inputs : int
        Number of input channels. Default: 1.
    output_fraction : float
        Fraction of N vertices used as readout features, in (0.0, 1.0]. Default: 1.0.
        Reduce for large dim to control Ridge readout cost.
    readout_type : ReadoutType
        ReadoutType.Ridge (default) or ReadoutType.Linear.
    feature_mode : FeatureMode
        FeatureMode.Translated (default) or FeatureMode.Raw.

    Examples
    --------
    >>> import numpy as np
    >>> import hypercube_rc as hrc
    >>> signal = np.sin(np.linspace(0, 20*np.pi, 2000)).astype(np.float32)
    >>> esn = hrc.ESN(dim=6, seed=42)
    >>> esn.warmup(signal[:200])
    >>> esn.run(signal[200:-1])
    >>> esn.train(signal[201:])
    >>> esn.r2(signal[201:], 0, esn.num_collected)
    0.999...
    """

    def __init__(
        self,
        dim: int,
        *,
        seed: int = 0,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.02,
        leak_rate: float = 1.0,
        alpha: float = 1.0,
        num_inputs: int = 1,
        output_fraction: float = 1.0,
        readout_type: ReadoutType = ReadoutType.Ridge,
        feature_mode: FeatureMode = FeatureMode.Translated,
    ):
        if dim not in _ESN_CLASSES:
            raise ValueError(f"dim must be 5-12, got {dim}")
        cls = _ESN_CLASSES[dim]
        self._impl = cls(
            seed=seed,
            spectral_radius=spectral_radius,
            input_scaling=input_scaling,
            leak_rate=leak_rate,
            alpha=alpha,
            num_inputs=num_inputs,
            output_fraction=output_fraction,
            readout_type=readout_type,
            feature_mode=feature_mode,
        )

    def warmup(self, inputs: np.ndarray) -> None:
        """Drive the reservoir without recording states (wash out transient).

        Parameters
        ----------
        inputs : ndarray
            Input signal. Shape ``(num_steps,)`` for single-input or
            ``(num_steps, num_inputs)`` for multi-input. Converted to float32.
        """
        self._impl.warmup(_to_float32(inputs))

    def run(self, inputs: np.ndarray) -> None:
        """Drive the reservoir and record states for training/evaluation.

        Parameters
        ----------
        inputs : ndarray
            Input signal. Same shape convention as ``warmup()``.

        Notes
        -----
        Multiple ``run()`` calls accumulate states. Use ``clear_states()``
        to reset between independent sequences.
        """
        self._impl.run(_to_float32(inputs))

    def clear_states(self) -> None:
        """Clear collected states and cached features.

        The reservoir's live internal state is preserved. The trained
        readout is also preserved.
        """
        self._impl.clear_states()

    def train(
        self,
        targets: np.ndarray,
        *,
        lambda_: float | None = None,
        lr: float | None = None,
        epochs: int | None = None,
        weight_decay: float = 1e-4,
        lr_decay: float = 0.01,
    ) -> None:
        """Train the readout on collected states.

        With no optional arguments, uses default parameters for the selected
        readout type.

        Parameters
        ----------
        targets : ndarray
            Target values, shape ``(train_size,)``. For regression: continuous
            values. For classification: {-1.0, +1.0}.
        lambda_ : float, optional
            Ridge regularization strength (Ridge readout only). Typical: 0.01-100.
        lr : float, optional
            Learning rate (Linear readout only). Pass 0.0 for auto-selection.
        epochs : int, optional
            Number of SGD epochs (Linear readout only). Default: 200.
        weight_decay : float
            L2 regularization for SGD. Default: 1e-4.
        lr_decay : float
            Learning rate decay factor. Default: 0.01.
        """
        self._impl.train(
            _to_float32(targets),
            lambda_=lambda_,
            lr=lr,
            epochs=epochs,
            weight_decay=weight_decay,
            lr_decay=lr_decay,
        )

    def train_incremental(
        self,
        targets: np.ndarray,
        *,
        blend: float = 0.1,
        lr: float = 0.0,
        epochs: int = 200,
        weight_decay: float = 1e-4,
        lr_decay: float = 0.01,
    ) -> None:
        """Incrementally update the Linear readout for streaming.

        Trains a fresh model on the provided data, then blends it with the
        existing model: ``W = (1 - blend) * W_old + blend * W_new``.

        Parameters
        ----------
        targets : ndarray
            Target values for the new data window.
        blend : float
            Blending factor in (0, 1]. Default: 0.1.
        lr : float
            Learning rate. 0.0 = auto. Default: 0.0.
        epochs : int
            SGD epochs. Default: 200.
        weight_decay : float
            L2 regularization. Default: 1e-4.
        lr_decay : float
            LR decay factor. Default: 0.01.
        """
        self._impl.train_incremental(
            _to_float32(targets),
            blend=blend, lr=lr, epochs=epochs,
            weight_decay=weight_decay, lr_decay=lr_decay,
        )

    def predict_raw(self, timestep: int) -> float:
        """Return the raw continuous prediction for a collected timestep.

        Parameters
        ----------
        timestep : int
            Index into collected states, in [0, num_collected).

        Returns
        -------
        float
            Continuous prediction value.
        """
        return self._impl.predict_raw(timestep)

    def predictions(self) -> np.ndarray:
        """Return predictions for all collected timesteps.

        Returns
        -------
        ndarray
            1D array of shape ``(num_collected,)`` with float32 predictions.
        """
        return self._impl.predictions()

    def r2(self, targets: np.ndarray, start: int, count: int) -> float:
        """Compute R-squared on a slice of collected states.

        Parameters
        ----------
        targets : ndarray
            Target values. The slice ``targets[start:start+count]`` is used.
        start : int
            First timestep index.
        count : int
            Number of timesteps to evaluate.

        Returns
        -------
        float
            R² value. 1.0 = perfect, 0.0 = predicts the mean.
        """
        return self._impl.r2(_to_float32(targets), start, count)

    def nrmse(self, targets: np.ndarray, start: int, count: int) -> float:
        """Compute Normalized RMSE on a slice of collected states.

        Parameters
        ----------
        targets : ndarray
            Target values. The slice ``targets[start:start+count]`` is used.
        start : int
            First timestep index.
        count : int
            Number of timesteps to evaluate.

        Returns
        -------
        float
            NRMSE value. 0.0 = perfect, 1.0 = as bad as predicting the mean.
        """
        return self._impl.nrmse(_to_float32(targets), start, count)

    def accuracy(self, labels: np.ndarray, start: int, count: int) -> float:
        """Compute classification accuracy on a slice of collected states.

        Parameters
        ----------
        labels : ndarray
            Labels with values {-1.0, +1.0}.
        start : int
            First timestep index.
        count : int
            Number of timesteps to evaluate.

        Returns
        -------
        float
            Fraction correct in [0.0, 1.0].
        """
        return self._impl.accuracy(_to_float32(labels), start, count)

    def selected_states(self) -> np.ndarray:
        """Return stride-selected states.

        Returns
        -------
        ndarray
            Array of shape ``(num_collected, num_output_verts)``.
        """
        return self._impl.selected_states()

    @property
    def dim(self) -> int:
        """Hypercube dimension."""
        return self._impl.dim

    @property
    def N(self) -> int:
        """Number of neurons (2^dim)."""
        return self._impl.N

    @property
    def num_collected(self) -> int:
        """Number of timesteps recorded by ``run()``."""
        return self._impl.num_collected

    @property
    def num_features(self) -> int:
        """Number of features per timestep."""
        return self._impl.num_features

    @property
    def num_inputs(self) -> int:
        """Number of input channels."""
        return self._impl.num_inputs

    @property
    def output_fraction(self) -> float:
        """Fraction of vertices used as readout features."""
        return self._impl.output_fraction

    @property
    def output_stride(self) -> int:
        """Stride used for vertex selection."""
        return self._impl.output_stride

    @property
    def num_output_verts(self) -> int:
        """Number of selected output vertices."""
        return self._impl.num_output_verts

    @property
    def readout_type(self) -> ReadoutType:
        """Readout type (Ridge or Linear)."""
        return self._impl.readout_type

    @property
    def feature_mode(self) -> FeatureMode:
        """Feature mode (Translated or Raw)."""
        return self._impl.feature_mode

    @property
    def alpha(self) -> float:
        """Tanh gain parameter."""
        return self._impl.alpha

    def __repr__(self) -> str:
        return (
            f"ESN(dim={self.dim}, N={self.N}, "
            f"readout={self.readout_type.name}, "
            f"features={self.feature_mode.name}, "
            f"collected={self.num_collected})"
        )
