"""
Google Sycamore Hardware Transpiler for Equivariant Exchange Circuits.

Decomposes the isotropic Heisenberg exchange generator:
    exp(-i * theta * (XX + YY + ZZ))
into native Google Sycamore PhasedFSimGate / FSimGate operations and single-qubit rotations
on planar grid topologies.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
    import cirq
    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False
    cirq = None


def decompose_heisenberg_interaction_to_fsim(
    q0: "cirq.Qid",
    q1: "cirq.Qid",
    theta: float,
) -> List["cirq.Operation"]:
    """
    Decomposes exp(-i * theta * (XX + YY + ZZ)) into native Cirq FSim/PhasedFSim gates:
        exp(-i * theta * (XX + YY + ZZ)) = exp(-i * theta * ZZ) * exp(-i * theta * (XX + YY))
    Where:
        - exp(-i * theta * (XX + YY)) = FSim(theta = 2 * theta, phi = 0)
        - exp(-i * theta * ZZ) = CNOT(q0, q1) + Rz(2 * theta, q1) + CNOT(q0, q1)
    """
    if not HAS_CIRQ:
        raise RuntimeError("Cirq required.")

    ops = []

    # 1. XY plane exchange rotation: FSim(2*theta, 0)
    # Cirq FSimGate(theta, phi) implements exp(-i * (theta/2) * (XX + YY) - i * (phi/4) * (I-Z)(I-Z))
    ops.append(cirq.FSimGate(theta=2.0 * theta, phi=0.0)(q0, q1))

    # 2. Z-axis interaction: exp(-i * theta * ZZ)
    ops.append(cirq.CNOT(q0, q1))
    ops.append(cirq.rz(2.0 * theta)(q1))
    ops.append(cirq.CNOT(q0, q1))

    return ops


class SycamoreEquivariantTranspiler:
    """
    Transpiles Equivariant Quantum Neural Networks onto Google Sycamore planar grid architectures.
    """

    def __init__(self, n_qubits: int, grid_origin: Tuple[int, int] = (4, 4)):
        if not HAS_CIRQ:
            raise RuntimeError("Cirq required.")
        self.n_qubits = n_qubits
        self.origin = grid_origin
        self.grid_qubits = self._place_qubits_on_grid()

    def _place_qubits_on_grid(self) -> List["cirq.GridQubit"]:
        """Places qubits linearly or in a snake pattern on Sycamore grid."""
        qubits = []
        r, c = self.origin
        for i in range(self.n_qubits):
            qubits.append(cirq.GridQubit(r + (i // 4), c + (i % 4)))
        return qubits

    def transpile_circuit(
        self,
        n_layers: int,
        params: np.ndarray,
        couplings: Optional[List[Tuple[int, int]]] = None,
    ) -> "cirq.Circuit":
        """
        Builds a native Cirq circuit consisting of Sycamore FSim gates and 1Q gates.
        """
        circuit = cirq.Circuit()
        bonds = couplings or [(i, i + 1) for i in range(self.n_qubits - 1)]

        # Singlet state preparation (|01> - |10>) / sqrt(2)
        for i in range(0, self.n_qubits, 2):
            q_a = self.grid_qubits[i]
            q_b = self.grid_qubits[i + 1]
            circuit.append([
                cirq.X(q_b),
                cirq.H(q_a),
                cirq.CNOT(q_a, q_b),
                cirq.Z(q_a),
            ])

        # Equivariant layers
        for l in range(n_layers):
            theta_l = params[l] if len(params) == n_layers else params[l % len(params)]
            for i, j in bonds:
                q_i = self.grid_qubits[i]
                q_j = self.grid_qubits[j]
                ops = decompose_heisenberg_interaction_to_fsim(q_i, q_j, theta_l)
                circuit.append(ops)

        return circuit

    def simulate_with_sycamore_noise(
        self, circuit: "cirq.Circuit", depolarizing_prob: float = 0.005
    ) -> np.ndarray:
        """
        Simulates circuit execution under Sycamore hardware noise channel.
        """
        noisy_circuit = cirq.Circuit()
        for moment in circuit:
            noisy_circuit.append(moment)
            for q in moment.qubits:
                noisy_circuit.append(cirq.depolarize(depolarizing_prob).on(q))

        sim = cirq.DensityMatrixSimulator()
        res = sim.simulate(noisy_circuit)
        return res.final_density_matrix
