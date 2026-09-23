"""
Permutation-equivariant Quantum Graph Neural Network.

Each layer evolves under the graph Heisenberg Hamiltonian
    H_A = Σ_{i<j} A_ij (X_iX_j + Y_iY_j + Z_iZ_j)
for a shared angle θ (a single exponential, so the layer is exactly
equivariant under every graph automorphism), followed by degree-weighted
Z rotations exp(−iφ deg_i Z_i). Node relabelling by an automorphism π of A
therefore satisfies  P_π U(A) P_π† = U(A).

Weight sharing is what makes the model equivariant, so it has few parameters
by construction: 2 per layer (θ, φ), or 3 with ``second_neighbours=True``,
which adds an exchange term on graph-distance-2 pairs with its own angle.

Compilation
-----------
A single exponential of a sum of non-commuting terms is not a hardware gate.
With ``trotter_steps=r`` every layer is compiled into a symmetric (second-order)
Trotter product of native two-qubit exchange gates
exp(−iα σ_i·σ_j) = IsingXX(2α)·IsingYY(2α)·IsingZZ(2α). Each of those commutes
with the total spin, so SU(2) equivariance stays exact; permutation
equivariance becomes approximate because the product has a fixed bond order,
with an error that falls as O(θ³/r²). ``equivariance_error`` measures it.
"""

from __future__ import annotations

from functools import reduce
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

from .models import pauli_string_matrix

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:  # pragma: no cover
    HAS_PENNYLANE = False
    qml = None


def permutation_matrix(perm: Sequence[int], n_qubits: int) -> np.ndarray:
    """Unitary P with P|b⟩ = |b'⟩, b'[perm[i]] = b[i] (qubit 0 = most significant bit)."""
    dim = 2**n_qubits
    P = np.zeros((dim, dim))
    for b in range(dim):
        bits = [(b >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)]
        new_bits = [0] * n_qubits
        for i, bit in enumerate(bits):
            new_bits[perm[i]] = bit
        b2 = sum(bit << (n_qubits - 1 - i) for i, bit in enumerate(new_bits))
        P[b2, b] = 1.0
    return P


class QuantumGraphNeuralNetwork:
    """Graph-convolutional quantum layers on the adjacency matrix ``A``."""

    def __init__(self, adjacency_matrix: np.ndarray, n_layers: int = 3, trotter_steps: Optional[int] = None,
                 second_neighbours: bool = False):
        self.adj = np.array(adjacency_matrix, dtype=np.float64)
        if self.adj.shape[0] != self.adj.shape[1] or not np.allclose(self.adj, self.adj.T):
            raise ValueError("adjacency_matrix must be square and symmetric")
        self.n_nodes = self.adj.shape[0]
        self.n_layers = int(n_layers)
        self.degrees = self.adj.sum(axis=1)
        self.trotter_steps = None if trotter_steps is None else int(trotter_steps)
        self.second_neighbours = bool(second_neighbours)
        A = (self.adj != 0).astype(int)
        two = ((A @ A) > 0) & (A == 0)
        np.fill_diagonal(two, False)
        self.adj2 = two.astype(float)                      # graph-distance-2 pairs (automorphism invariant)

    @property
    def params_per_layer(self) -> int:
        return 3 if self.second_neighbours else 2

    def num_params(self) -> int:
        return self.params_per_layer * self.n_layers

    def _weighted_bonds(self, adj: np.ndarray) -> List[Tuple[int, int, float]]:
        n = self.n_nodes
        return [(i, j, float(adj[i, j])) for i in range(n) for j in range(i + 1, n) if abs(adj[i, j]) > 1e-12]

    def _exchange_sum(self, adj: np.ndarray) -> np.ndarray:
        n = self.n_nodes
        H = np.zeros((2**n, 2**n), dtype=np.complex128)
        for i, j, w in self._weighted_bonds(adj):
            for p in "XYZ":
                H += w * pauli_string_matrix({i: p, j: p}, n)
        return H

    def graph_hamiltonian(self) -> np.ndarray:
        return self._exchange_sum(self.adj)

    def _layer_terms(self, params, l: int):
        """[(angle, adjacency)] exchange terms of layer l, and the node angle φ."""
        k = self.params_per_layer
        theta, phi = float(params[k * l]), float(params[k * l + 1])
        terms = [(theta, self.adj)]
        if self.second_neighbours:
            terms.append((float(params[k * l + 2]), self.adj2))
        return terms, phi

    def _trotter_sequence(self, terms) -> List[Tuple[int, int, float]]:
        """Second-order product: bonds forward with half angles, then backward."""
        r = self.trotter_steps
        bonds = [(i, j, ang * w / r) for ang, adj in terms for i, j, w in self._weighted_bonds(adj)]
        half = [(i, j, 0.5 * a) for i, j, a in bonds]
        return (half + half[::-1]) * r

    def _pennylane_hamiltonian(self, adj: np.ndarray):
        coeffs, ops = [], []
        for i, j, w in self._weighted_bonds(adj):
            for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
                coeffs.append(w)
                ops.append(P(i) @ P(j))
        return qml.Hamiltonian(coeffs, ops)

    def _pennylane_graph_hamiltonian(self):
        return self._pennylane_hamiltonian(self.adj)

    def apply_circuit(self, params) -> None:
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")
        for i in range(self.n_nodes):
            qml.Hadamard(wires=i)
        for l in range(self.n_layers):
            terms, phi = self._layer_terms(params, l)
            if self.trotter_steps is None:
                for ang, adj in terms:                   # exact e^{−iθ H} (simulator-only operator)
                    qml.exp(self._pennylane_hamiltonian(adj), -1j * ang)
            else:
                for i, j, a in self._trotter_sequence(terms):   # native exchange gates
                    qml.IsingXX(2 * a, wires=[i, j])
                    qml.IsingYY(2 * a, wires=[i, j])
                    qml.IsingZZ(2 * a, wires=[i, j])
            for i in range(self.n_nodes):
                qml.RZ(2.0 * phi * float(self.degrees[i]), wires=i)

    def unitary(self, params) -> np.ndarray:
        n = self.n_nodes
        hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        U = reduce(np.kron, [hadamard] * n).astype(np.complex128)   # |+⟩^n preparation
        Zsum = sum(float(self.degrees[i]) * pauli_string_matrix({i: "Z"}, n) for i in range(n))
        H_cache = {}
        for l in range(self.n_layers):
            terms, phi = self._layer_terms(params, l)
            if self.trotter_steps is None:
                H = sum(ang * H_cache.setdefault(id(adj), self._exchange_sum(adj)) for ang, adj in terms)
                U = scipy.linalg.expm(-1j * H) @ U
            else:
                for i, j, a in self._trotter_sequence(terms):
                    G = sum(pauli_string_matrix({i: p, j: p}, n) for p in "XYZ")
                    U = scipy.linalg.expm(-1j * a * G) @ U
            U = scipy.linalg.expm(-1j * phi * Zsum) @ U
        return U

    def equivariance_error(self, params, perm: Sequence[int]) -> float:
        """‖P_π U P_π† − U‖₂ (0 for the exact model and any automorphism π)."""
        U = self.unitary(params)
        P = permutation_matrix(perm, self.n_nodes)
        return float(np.linalg.norm(P @ U @ P.conj().T - U, 2))

    def su2_equivariance_error(self, params) -> float:
        """Largest ‖[U_layers, S^α]‖ over the total-spin components, excluding the node rotations' Z-only part."""
        from .models import get_su2_generators

        n = self.n_nodes
        V = np.eye(2**n, dtype=np.complex128)
        for l in range(self.n_layers):
            terms, _ = self._layer_terms(params, l)
            if self.trotter_steps is None:
                V = scipy.linalg.expm(-1j * sum(a * self._exchange_sum(adj) for a, adj in terms)) @ V
            else:
                for i, j, a in self._trotter_sequence(terms):
                    V = scipy.linalg.expm(-1j * a * sum(pauli_string_matrix({i: p, j: p}, n) for p in "XYZ")) @ V
        return float(max(np.linalg.norm(V @ S - S @ V, 2) for S in get_su2_generators(n)))

    def is_permutation_equivariant(self, params, perm: Sequence[int], tol: float = 1e-8) -> bool:
        """True iff P_π U P_π† = U for the node permutation ``perm``."""
        U = self.unitary(params)
        P = permutation_matrix(perm, self.n_nodes)
        return bool(np.linalg.norm(P @ U @ P.conj().T - U) < tol)
