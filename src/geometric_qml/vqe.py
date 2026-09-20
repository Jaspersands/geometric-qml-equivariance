"""
Equivariant Variational Quantum Eigensolver (VQE).

Finds the ground state energy of Heisenberg spin chains using PennyLane optimizers,
leveraging exact SU(2) singlet symmetry confinement.
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

from .models import HeisenbergSpinChain, get_casimir_operator
from .ansatz import EquivariantQuantumAnsatz


class EquivariantVQE:
    """
    VQE solver restricted to the SU(2) singlet manifold.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        learning_rate: float = 0.1,
        seed: Optional[int] = None,
        lattice_type: str = "1d_chain",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = learning_rate
        self.rng = np.random.default_rng(seed)
        self.lattice_type = lattice_type

        self.spin_chain = HeisenbergSpinChain(n_qubits=n_qubits, j_coupling=1.0, anisotropy_delta=1.0, lattice_type=lattice_type)
        self.H_qml = self.spin_chain.get_pennylane_hamiltonian()
        self.ansatz = EquivariantQuantumAnsatz(
            n_qubits=n_qubits, n_layers=n_layers, weight_sharing=True, couplings=self.spin_chain.couplings
        )

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, diff_method="backprop")
        def cost_fn(params):
            self.ansatz.apply_circuit(params)
            return qml.expval(self.H_qml)

        @qml.qnode(self.dev, diff_method="backprop")
        def state_fn(params):
            self.ansatz.apply_circuit(params)
            return qml.state()

        self.cost_fn = cost_fn
        self.state_fn = state_fn

    def train(self, steps: int = 40, initial_params: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Runs gradient descent VQE optimization."""
        opt = qml.AdamOptimizer(stepsize=self.lr)
        p_raw = self.rng.uniform(0.1, 1.0, size=self.ansatz.num_params()) if initial_params is None else np.array(initial_params, dtype=float)
        p = qml.numpy.array(p_raw, requires_grad=True)

        exact_e, _ = self.spin_chain.exact_ground_state()
        energy_history = []

        for step in range(steps):
            p, energy = opt.step_and_cost(self.cost_fn, p)
            energy_history.append(float(energy))

        final_energy = float(self.cost_fn(p))
        psi = np.asarray(self.state_fn(p))
        s2 = float(np.real(np.vdot(psi, get_casimir_operator(self.n_qubits) @ psi)))
        return {
            "final_energy": final_energy,
            "exact_ground_energy": exact_e,
            "energy_error": abs(final_energy - exact_e),
            "energy_history": energy_history,
            "optimal_params": np.array(p),
            "singlet_purity": 1.0 - s2 / max(1.0, 0.25 * self.n_qubits * (0.5 * self.n_qubits + 1)),
            "total_spin_expectation": s2,
        }
