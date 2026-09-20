"""
Permutation-equivariant Quantum Graph Neural Network.

Each layer evolves under the graph Heisenberg Hamiltonian
    H_A = Σ_{i<j} A_ij (X_iX_j + Y_iY_j + Z_iZ_j)
for a shared angle θ (a single exponential, so the layer is exactly
equivariant under every graph automorphism), followed by degree-weighted
Z rotations exp(−iφ deg_i Z_i). Node relabelling by an automorphism π of A
therefore satisfies  P_π U(A) P_π† = U(A).
"""

from __future__ import annotations

from functools import reduce
from typing import List, Sequence

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

    def __init__(self, adjacency_matrix: np.ndarray, n_layers: int = 3):
        self.adj = np.array(adjacency_matrix, dtype=np.float64)
        if self.adj.shape[0] != self.adj.shape[1] or not np.allclose(self.adj, self.adj.T):
            raise ValueError("adjacency_matrix must be square and symmetric")
        self.n_nodes = self.adj.shape[0]
        self.n_layers = int(n_layers)
        self.degrees = self.adj.sum(axis=1)

    def num_params(self) -> int:
        return 2 * self.n_layers

    def graph_hamiltonian(self) -> np.ndarray:
        n = self.n_nodes
        H = np.zeros((2**n, 2**n), dtype=np.complex128)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.adj[i, j]) > 1e-12:
                    for p in "XYZ":
                        H += self.adj[i, j] * pauli_string_matrix({i: p, j: p}, n)
        return H

    def _pennylane_graph_hamiltonian(self):
        coeffs, ops = [], []
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if abs(self.adj[i, j]) > 1e-12:
                    for P in (qml.PauliX, qml.PauliY, qml.PauliZ):
                        coeffs.append(float(self.adj[i, j]))
                        ops.append(P(i) @ P(j))
        return qml.Hamiltonian(coeffs, ops)

    def apply_circuit(self, params) -> None:
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")
        for i in range(self.n_nodes):
            qml.Hadamard(wires=i)
        H = self._pennylane_graph_hamiltonian()
        for l in range(self.n_layers):
            theta, phi = params[2 * l], params[2 * l + 1]
            qml.exp(H, -1j * theta)                      # exact e^{−iθ H_A}
            for i in range(self.n_nodes):
                qml.RZ(2.0 * phi * float(self.degrees[i]), wires=i)

    def unitary(self, params) -> np.ndarray:
        n = self.n_nodes
        H = self.graph_hamiltonian()
        hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        U = reduce(np.kron, [hadamard] * n).astype(np.complex128)   # |+⟩^n preparation
        for l in range(self.n_layers):
            theta, phi = float(params[2 * l]), float(params[2 * l + 1])
            U = scipy.linalg.expm(-1j * theta * H) @ U
            Zsum = sum(float(self.degrees[i]) * pauli_string_matrix({i: "Z"}, n) for i in range(n))
            U = scipy.linalg.expm(-1j * phi * Zsum) @ U
        return U

    def is_permutation_equivariant(self, params, perm: Sequence[int], tol: float = 1e-8) -> bool:
        """True iff P_π U P_π† = U for the node permutation ``perm``."""
        U = self.unitary(params)
        P = permutation_matrix(perm, self.n_nodes)
        return bool(np.linalg.norm(P @ U @ P.conj().T - U) < tol)
