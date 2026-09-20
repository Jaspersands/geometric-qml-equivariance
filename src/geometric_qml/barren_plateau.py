"""
Gradient-variance (barren plateau) analysis.

Samples ∂_θ ⟨H⟩ at uniformly random parameters with PennyLane backprop and
reports Var[∂_θ₀⟨H⟩], the mean variance over all parameters, and log-linear
fits of the variance versus qubit number for several ansätze.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    qml = None

from .models import HeisenbergSpinChain
from .ansatz import (
    EquivariantQuantumAnsatz,
    HardwareEfficientAnsatz,
    QAOABaselineAnsatz,
)


class GradientVarianceAnalyzer:
    """
    Computes gradient variance statistics for parameterized quantum circuits
    using PennyLane's differentiable execution framework.
    """

    def __init__(self, n_qubits: int, ansatz_type: str = "equivariant", n_layers: int = 3):
        self.n_qubits = n_qubits
        self.ansatz_type = ansatz_type
        self.n_layers = n_layers

        self.spin_chain = HeisenbergSpinChain(n_qubits=n_qubits, j_coupling=1.0, anisotropy_delta=1.0)
        self.H_qml = self.spin_chain.get_pennylane_hamiltonian()

        if ansatz_type == "equivariant":
            self.ansatz = EquivariantQuantumAnsatz(n_qubits=n_qubits, n_layers=n_layers, weight_sharing=True)
        elif ansatz_type == "hea":
            self.ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_layers=n_layers)
        elif ansatz_type == "qaoa":
            self.ansatz = QAOABaselineAnsatz(n_qubits=n_qubits, n_layers=n_layers)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        """Constructs the differentiable PennyLane QNode."""
        @qml.qnode(self.dev, diff_method="backprop")
        def circuit(params):
            self.ansatz.apply_circuit(params)
            return qml.expval(self.H_qml)

        self.qnode = circuit
        self.grad_fn = qml.grad(circuit)

    def sample_gradients(self, n_samples: int = 40, seed: Optional[int] = None) -> np.ndarray:
        """
        Samples random parameter vectors and computes the analytical gradient vector for each.
        Returns array of shape (n_samples, n_params).
        """
        rng = np.random.default_rng(seed)
        n_p = self.ansatz.num_params()
        grads = np.zeros((n_samples, n_p))

        for s in range(n_samples):
            # Draw random uniform parameters in [0, 2pi)
            p_raw = rng.uniform(0.0, 2 * np.pi, size=n_p)
            p_vec = qml.numpy.array(p_raw, requires_grad=True)
            grad_vec = self.grad_fn(p_vec)
            grads[s] = np.array(grad_vec, dtype=np.float64)

        return grads

    def compute_gradient_variance(self, n_samples: int = 40, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Calculates the mean gradient, variance of the first parameter Var[partial_{theta_0} <H>],
        and the total trace variance sum_k Var[partial_{theta_k} <H>].
        """
        grads = self.sample_gradients(n_samples=n_samples, seed=seed)

        # Variance along parameter 0
        var_param_0 = float(np.var(grads[:, 0]))
        # Mean variance across all parameters
        var_mean = float(np.mean(np.var(grads, axis=0)))
        # Mean absolute gradient norm
        grad_norms = np.linalg.norm(grads, axis=1)
        mean_norm = float(np.mean(grad_norms))

        return {
            "var_param_0": var_param_0,
            "var_mean": var_mean,
            "mean_norm": mean_norm,
            "n_params": self.ansatz.num_params(),
            "n_qubits": self.n_qubits,
        }


def fit_variance_scaling(qubit_counts: Sequence[int], variances: Sequence[float]) -> Dict[str, float]:
    """
    Least-squares fit of log₂ Var = slope·N + intercept. A slope near −1 (per
    qubit) is the hallmark of a barren plateau (Var ∝ 2^{−N}); slopes near 0
    indicate trainability.
    """
    x = np.asarray(qubit_counts, dtype=float)
    y = np.log2(np.maximum(np.asarray(variances, dtype=float), 1e-300))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-300
    return {"slope_log2": float(slope), "intercept_log2": float(intercept), "r2": 1.0 - ss_res / ss_tot}


def run_barren_plateau_scaling_study(
    qubit_list: Sequence[int] = (4, 6, 8),
    n_layers: Optional[int] = None,
    n_samples: int = 30,
    seed: int = 42,
    ansatz_types: Sequence[str] = ("equivariant", "hea"),
    depth: str = "linear",
    normalize_equivariant: bool = True,
) -> Dict[str, Any]:
    """
    Gradient-variance scaling for several ansätze across qubit counts.

    ``depth='linear'`` uses L = N layers (the regime in which barren plateaus
    appear; McClean et al. 2018); ``depth='fixed'`` uses ``n_layers`` for all N.
    With ``normalize_equivariant`` the weight-shared EQNN variance is divided by
    (number of bonds)² so that it measures the gradient *per bond* — its single
    angle multiplies a sum over all bonds, otherwise the raw variance grows with N.

    Returns per-ansatz ``variances`` (of ∂θ₀), ``var_means`` (mean over all
    parameters), raw ``samples``, log-linear ``fits`` of both, and the depths used.
    N = 2 is deliberately excluded from the defaults: the EQNN cost is constant
    there (the singlet is the unique S = 0 state), so its gradient vanishes.
    """
    variances: Dict[str, List[float]] = {a: [] for a in ansatz_types}
    var_means: Dict[str, List[float]] = {a: [] for a in ansatz_types}
    samples: Dict[str, List[np.ndarray]] = {a: [] for a in ansatz_types}
    depths: List[int] = []
    for n_q in qubit_list:
        L = int(n_q) if depth == "linear" else int(n_layers if n_layers is not None else 2)
        depths.append(L)
        for a in ansatz_types:
            analyzer = GradientVarianceAnalyzer(n_qubits=n_q, ansatz_type=a, n_layers=L)
            grads = analyzer.sample_gradients(n_samples=n_samples, seed=seed)
            norm = (n_q - 1) ** 2 if (a == "equivariant" and normalize_equivariant) else 1.0
            variances[a].append(float(np.var(grads[:, 0]) / norm))
            var_means[a].append(float(np.mean(np.var(grads, axis=0)) / norm))
            samples[a].append(grads)
    fits = {a: fit_variance_scaling(qubit_list, variances[a]) for a in ansatz_types}
    fits_mean = {a: fit_variance_scaling(qubit_list, var_means[a]) for a in ansatz_types}
    out = {
        "qubit_counts": list(qubit_list),
        "depths": depths,
        "depth_mode": depth,
        "variances": variances,
        "var_means": var_means,
        "samples": samples,
        "fits": fits,
        "fits_mean": fits_mean,
        "n_layers": n_layers,
        "normalized_equivariant": normalize_equivariant,
    }
    if "equivariant" in variances:
        out["eqnn_variances"] = variances["equivariant"]
    if "hea" in variances:
        out["hea_variances"] = variances["hea"]
    return out
