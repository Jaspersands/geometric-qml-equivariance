"""
Parameterised circuit families:

1. ``EquivariantQuantumAnsatz`` — SU(2)-equivariant exchange layers
   exp(−iθ (X_iX_j + Y_iY_j + Z_iZ_j)) that commute with the total spin.
2. ``HardwareEfficientAnsatz`` — RY/RZ rotations with CNOT ladders (no symmetry).
3. ``QAOABaselineAnsatz`` — alternating ZZ-cost and X-mixer layers.

Each exposes ``apply_circuit(params)`` (PennyLane), ``generators()`` (the
Hermitian generators of every gate, used by the DLA analyzer) and
``unitary(params)`` (dense NumPy matrix, no PennyLane needed).
"""

from __future__ import annotations

from functools import reduce
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg

from .models import PAULIS, pauli_string_matrix, create_singlet_product_state

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:  # pragma: no cover
    HAS_PENNYLANE = False
    qml = None


def _exchange_generator(i: int, j: int, n: int) -> np.ndarray:
    return sum(pauli_string_matrix({i: p, j: p}, n) for p in "XYZ")


def _cnot_matrix(control: int, target: int, n: int) -> np.ndarray:
    P0 = 0.5 * (pauli_string_matrix({}, n) + pauli_string_matrix({control: "Z"}, n))
    P1 = 0.5 * (pauli_string_matrix({}, n) - pauli_string_matrix({control: "Z"}, n))
    return P0 + P1 @ pauli_string_matrix({target: "X"}, n)


def _cnot_generator(control: int, target: int, n: int) -> np.ndarray:
    """CNOT = exp(iπ/4 (1−Z_c)(1−X_t)) up to phase; generator (1−Z_c)(1−X_t)."""
    I = pauli_string_matrix({}, n)
    return (I - pauli_string_matrix({control: "Z"}, n)) @ (I - pauli_string_matrix({target: "X"}, n))


class EquivariantQuantumAnsatz:
    """
    Exchange-generated EQNN. With ``weight_sharing`` one angle per layer is
    shared by all bonds (permutation-equivariant on translation-invariant
    lattices); otherwise every bond has its own angle.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        couplings: Optional[List[Tuple[int, int]]] = None,
        weight_sharing: bool = True,
    ):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.weight_sharing = bool(weight_sharing)
        self.couplings = list(couplings) if couplings is not None else [(i, i + 1) for i in range(n_qubits - 1)]

    def num_params(self) -> int:
        return self.n_layers if self.weight_sharing else self.n_layers * len(self.couplings)

    def _angles(self, params) -> List[List[float]]:
        params = np.asarray(params)
        if self.weight_sharing:
            return [[params[l]] * len(self.couplings) for l in range(self.n_layers)]
        return [list(params[l * len(self.couplings):(l + 1) * len(self.couplings)]) for l in range(self.n_layers)]

    def apply_circuit(self, params) -> None:
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")
        create_singlet_product_state(self.n_qubits)
        for layer in self._angles(params):
            for (i, j), theta in zip(self.couplings, layer):
                for p in ("XX", "YY", "ZZ"):
                    qml.PauliRot(2 * theta, p, wires=[i, j])

    def generators(self) -> List[np.ndarray]:
        return [_exchange_generator(i, j, self.n_qubits) for i, j in self.couplings]

    def unitary(self, params) -> np.ndarray:
        """Dense unitary of the parameterised layers (excludes state preparation)."""
        gens = self.generators()
        U = np.eye(2**self.n_qubits, dtype=np.complex128)
        for layer in self._angles(params):
            for g, theta in zip(gens, layer):
                U = scipy.linalg.expm(-1j * float(theta) * g) @ U
        return U


class HardwareEfficientAnsatz:
    """RY·RZ on every qubit, linear CNOT chain, repeated; final rotation layer."""

    def __init__(self, n_qubits: int, n_layers: int = 3):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)

    def num_params(self) -> int:
        return (self.n_layers + 1) * self.n_qubits * 2

    def apply_circuit(self, params) -> None:
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")
        p_idx = 0
        for _ in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(params[p_idx], wires=i)
                qml.RZ(params[p_idx + 1], wires=i)
                p_idx += 2
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        for i in range(self.n_qubits):
            qml.RY(params[p_idx], wires=i)
            qml.RZ(params[p_idx + 1], wires=i)
            p_idx += 2

    def generators(self) -> List[np.ndarray]:
        n = self.n_qubits
        gens = [pauli_string_matrix({i: "Y"}, n) for i in range(n)] + [pauli_string_matrix({i: "Z"}, n) for i in range(n)]
        gens += [_cnot_generator(i, i + 1, n) for i in range(n - 1)]
        return gens

    def unitary(self, params) -> np.ndarray:
        n = self.n_qubits
        params = np.asarray(params, dtype=float)
        U = np.eye(2**n, dtype=np.complex128)
        p_idx = 0

        def rot(i, theta_y, theta_z):
            ry = scipy.linalg.expm(-0.5j * theta_y * PAULIS["Y"])
            rz = scipy.linalg.expm(-0.5j * theta_z * PAULIS["Z"])
            m = rz @ ry
            return reduce(np.kron, [m if k == i else PAULIS["I"] for k in range(n)])

        for _ in range(self.n_layers):
            for i in range(n):
                U = rot(i, params[p_idx], params[p_idx + 1]) @ U
                p_idx += 2
            for i in range(n - 1):
                U = _cnot_matrix(i, i + 1, n) @ U
        for i in range(n):
            U = rot(i, params[p_idx], params[p_idx + 1]) @ U
            p_idx += 2
        return U


class QAOABaselineAnsatz:
    """|+⟩^N then alternating exp(−iγ Σ ZZ) and exp(−iβ Σ X)."""

    def __init__(self, n_qubits: int, n_layers: int = 3, couplings: Optional[List[Tuple[int, int]]] = None):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.couplings = list(couplings) if couplings is not None else [(i, i + 1) for i in range(n_qubits - 1)]

    def num_params(self) -> int:
        return 2 * self.n_layers

    def apply_circuit(self, params) -> None:
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        for l in range(self.n_layers):
            gamma, beta = params[2 * l], params[2 * l + 1]
            for i, j in self.couplings:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma, wires=j)
                qml.CNOT(wires=[i, j])
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)

    def generators(self) -> List[np.ndarray]:
        n = self.n_qubits
        cost = sum(pauli_string_matrix({i: "Z", j: "Z"}, n) for i, j in self.couplings)
        mixer = sum(pauli_string_matrix({i: "X"}, n) for i in range(n))
        return [cost, mixer]

    def unitary(self, params) -> np.ndarray:
        cost, mixer = self.generators()
        U = np.eye(2**self.n_qubits, dtype=np.complex128)
        for l in range(self.n_layers):
            U = scipy.linalg.expm(-1j * params[2 * l + 1] * mixer) @ scipy.linalg.expm(-1j * params[2 * l] * cost) @ U
        return U
