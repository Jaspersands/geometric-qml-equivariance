"""
Barren Plateau and Gradient Variance Scaling Analyzer.

Calculates the statistical variance of cost function gradients:
    Var_{theta}[partial_{theta_k} <H>]
across qubit counts N and circuit depths L, comparing Equivariant QNNs vs. HEA.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

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


def run_barren_plateau_scaling_study(
    qubit_list: List[int] = [2, 4, 6, 8],
    n_layers: int = 2,
    n_samples: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes comparative gradient variance scaling study across qubit counts
    for Equivariant QNN vs. Hardware-Efficient Ansatz (HEA).
    """
    eqnn_vars = []
    hea_vars = []

    for n_q in qubit_list:
        # Equivariant QNN
        analyzer_eqnn = GradientVarianceAnalyzer(n_qubits=n_q, ansatz_type="equivariant", n_layers=n_layers)
        res_eqnn = analyzer_eqnn.compute_gradient_variance(n_samples=n_samples, seed=seed)
        eqnn_vars.append(res_eqnn["var_param_0"])

        # HEA Baseline
        analyzer_hea = GradientVarianceAnalyzer(n_qubits=n_q, ansatz_type="hea", n_layers=n_layers)
        res_hea = analyzer_hea.compute_gradient_variance(n_samples=n_samples, seed=seed)
        hea_vars.append(res_hea["var_param_0"])

    return {
        "qubit_counts": list(qubit_list),
        "eqnn_variances": eqnn_vars,
        "hea_variances": hea_vars,
        "n_layers": n_layers,
    }
