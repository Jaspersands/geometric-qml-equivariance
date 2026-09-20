"""
Google Sycamore transpilation of exchange-generated equivariant circuits.

The exchange generator decomposes exactly as

    exp(−iθ (XX + YY + ZZ)) = exp(−iθ ZZ) · exp(−iθ (XX + YY))
                            = [CNOT · Rz(2θ) · CNOT] · FSim(2θ, 0)

because the three terms commute. ``cirq.FSimGate(θ', φ)`` implements
exp(−iθ'/2 (XX + YY)) exp(−iφ |11⟩⟨11|), so θ' = 2θ. The gate is expressed
with ``FSimGate`` or ``PhasedFSimGate`` (the tunable-coupler parameterisation)
and qubits are routed on a Sycamore-style grid with a snake so that every
chain bond is a physical neighbour.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cirq

    HAS_CIRQ = True
except ImportError:  # pragma: no cover
    HAS_CIRQ = False
    cirq = None


def decompose_heisenberg_interaction_to_fsim(
    q0: "cirq.Qid",
    q1: "cirq.Qid",
    theta: float,
    use_phased_fsim: bool = False,
) -> List["cirq.Operation"]:
    """exp(−iθ (XX+YY+ZZ)) on (q0, q1) as native FSim + CNOT/Rz."""
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")
    if use_phased_fsim:
        xy = cirq.PhasedFSimGate(theta=2.0 * theta, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0)
    else:
        xy = cirq.FSimGate(theta=2.0 * theta, phi=0.0)
    return [xy(q0, q1), cirq.CNOT(q0, q1), cirq.rz(2.0 * theta)(q1), cirq.CNOT(q0, q1)]


class SycamoreEquivariantTranspiler:
    """
    Places ``n_qubits`` on a grid of width ``grid_width`` in a snake pattern
    (row 0 left→right, row 1 right→left, …) so consecutive indices are adjacent.
    """

    def __init__(self, n_qubits: int, grid_origin: Tuple[int, int] = (4, 4), grid_width: int = 4, use_phased_fsim: bool = False):
        if not HAS_CIRQ:
            raise RuntimeError("Cirq required.")
        self.n_qubits = int(n_qubits)
        self.origin = grid_origin
        self.grid_width = int(grid_width)
        self.use_phased_fsim = bool(use_phased_fsim)
        self.grid_qubits = self._place_qubits_on_grid()

    def _place_qubits_on_grid(self) -> List["cirq.GridQubit"]:
        r0, c0 = self.origin
        qubits = []
        for i in range(self.n_qubits):
            row, k = divmod(i, self.grid_width)
            col = k if row % 2 == 0 else self.grid_width - 1 - k
            qubits.append(cirq.GridQubit(r0 + row, c0 + col))
        return qubits

    def check_adjacency(self, bonds: Sequence[Tuple[int, int]]) -> bool:
        return all(self.grid_qubits[i].is_adjacent(self.grid_qubits[j]) for i, j in bonds)

    def transpile_circuit(
        self,
        n_layers: int,
        params: np.ndarray,
        couplings: Optional[List[Tuple[int, int]]] = None,
    ) -> "cirq.Circuit":
        """Singlet preparation + ``n_layers`` exchange layers with one shared angle per layer."""
        bonds = couplings or [(i, i + 1) for i in range(self.n_qubits - 1)]
        if not self.check_adjacency(bonds):
            bad = [(i, j) for i, j in bonds if not self.grid_qubits[i].is_adjacent(self.grid_qubits[j])]
            raise ValueError(f"bonds {bad} are not adjacent on the grid; choose a different coupling map")
        params = np.asarray(params, dtype=float)
        circuit = cirq.Circuit()
        for i in range(0, self.n_qubits - 1, 2):
            q_a, q_b = self.grid_qubits[i], self.grid_qubits[i + 1]
            circuit.append([cirq.X(q_b), cirq.H(q_a), cirq.CNOT(q_a, q_b), cirq.Z(q_a)])
        for l in range(n_layers):
            theta_l = params[l % len(params)]
            for i, j in bonds:
                circuit.append(decompose_heisenberg_interaction_to_fsim(self.grid_qubits[i], self.grid_qubits[j], theta_l, self.use_phased_fsim))
        return circuit

    def gate_counts(self, circuit: "cirq.Circuit") -> dict:
        counts = {"fsim": 0, "cnot": 0, "single_qubit": 0, "moments": len(circuit)}
        for op in circuit.all_operations():
            if isinstance(op.gate, (cirq.FSimGate, cirq.PhasedFSimGate)):
                counts["fsim"] += 1
            elif len(op.qubits) == 2:
                counts["cnot"] += 1
            else:
                counts["single_qubit"] += 1
        return counts

    def simulate_with_sycamore_noise(self, circuit: "cirq.Circuit", depolarizing_prob: float = 0.005) -> np.ndarray:
        noisy = cirq.Circuit()
        for moment in circuit:
            noisy.append(moment)
            for q in moment.qubits:
                noisy.append(cirq.depolarize(depolarizing_prob).on(q))
        return cirq.DensityMatrixSimulator().simulate(noisy).final_density_matrix
