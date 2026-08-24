"""
Permutation-Equivariant Quantum Graph Neural Network (QGNN).

Implements graph-convolutional quantum layers that are strictly invariant/equivariant
under node permutation automorphisms:
    [U_{QGNN}(A), P_pi] = 0 for pi in Aut(G).
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    qml = None


class QuantumGraphNeuralNetwork:
    """
    Permutation-equivariant quantum neural network tailored to graph data.
    """

    def __init__(self, adjacency_matrix: np.ndarray, n_layers: int = 3):
        self.adj = np.array(adjacency_matrix, dtype=np.float64)
        self.n_nodes = self.adj.shape[0]
        self.n_layers = n_layers

    def num_params(self) -> int:
        # 2 shared parameters per layer: (edge_weight_theta, node_weight_phi)
        return 2 * self.n_layers

    def apply_circuit(self, params: np.ndarray):
        """Builds the permutation-equivariant QGNN circuit."""
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")

        # 1. Equal superposition over all nodes
        for i in range(self.n_nodes):
            qml.Hadamard(wires=i)

        # 2. Graph convolutional layers
        for l in range(self.n_layers):
            theta_edge = params[2 * l]
            phi_node = params[2 * l + 1]

            # Edge message passing: exp(-i * theta * A_{ij} * (XX + YY + ZZ))
            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    weight = self.adj[i, j]
                    if abs(weight) > 1e-6:
                        angle = 2.0 * theta_edge * weight
                        qml.PauliRot(angle, "XX", wires=[i, j])
                        qml.PauliRot(angle, "YY", wires=[i, j])
                        qml.PauliRot(angle, "ZZ", wires=[i, j])

            # Node aggregation: exp(-i * phi * sum_i Z_i)
            for i in range(self.n_nodes):
                deg = float(np.sum(self.adj[i, :]))
                qml.RZ(2.0 * phi_node * max(1.0, deg), wires=i)
