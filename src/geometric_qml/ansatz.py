"""
Quantum Neural Network Ansatz Architectures:
1. Equivariant Quantum Neural Network (EQNN) preserving SU(2) symmetry.
2. Hardware-Efficient Ansatz (HEA) baseline.
3. QAOA / Alternating Operator Ansatz baseline.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    qml = None

from .models import create_singlet_product_state


class EquivariantQuantumAnsatz:
    """
    Equivariant Quantum Neural Network (EQNN) preserving exact SU(2) spin-rotation symmetry:
        [U(theta), S^alpha] = 0 for alpha in {x, y, z}.

    Constructed from exchange generators:
        G_{ij}(theta) = exp(-i * theta * (X_i X_j + Y_i Y_j + Z_i Z_j))
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 3,
        couplings: Optional[List[Tuple[int, int]]] = None,
        weight_sharing: bool = True, # If True, shares parameter theta_l across all bonds in layer l
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weight_sharing = weight_sharing

        if couplings is None:
            # Default 1D nearest neighbor chain
            self.couplings = [(i, i + 1) for i in range(n_qubits - 1)]
        else:
            self.couplings = couplings

    def num_params(self) -> int:
        if self.weight_sharing:
            return self.n_layers
        return self.n_layers * len(self.couplings)

    def apply_circuit(self, params: np.ndarray):
        """Builds the EQNN circuit in PennyLane."""
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")

        # 1. Initialize inside the invariant singlet subspace
        create_singlet_product_state(self.n_qubits)

        # 2. Apply Equivariant Layers
        p_idx = 0
        for l in range(self.n_layers):
            if self.weight_sharing:
                theta_l = params[l]
                for i, j in self.couplings:
                    # exp(-i * theta * (XX + YY + ZZ))
                    # Decomposed into isotropic Pauli evolutions
                    qml.PauliRot(2 * theta_l, "XX", wires=[i, j])
                    qml.PauliRot(2 * theta_l, "YY", wires=[i, j])
                    qml.PauliRot(2 * theta_l, "ZZ", wires=[i, j])
            else:
                for i, j in self.couplings:
                    theta_ij = params[p_idx]
                    p_idx += 1
                    qml.PauliRot(2 * theta_ij, "XX", wires=[i, j])
                    qml.PauliRot(2 * theta_ij, "YY", wires=[i, j])
                    qml.PauliRot(2 * theta_ij, "ZZ", wires=[i, j])


class HardwareEfficientAnsatz:
    """
    Standard unconstrained Hardware-Efficient Ansatz (HEA):
    Alternating single-qubit rotations (RY, RZ) and linear CNOT entanglers.
    Exhibits exponentially vanishing gradient variance Var[grad] ~ O(1/2^N).
    """

    def __init__(self, n_qubits: int, n_layers: int = 3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def num_params(self) -> int:
        # 2 rotation parameters per qubit per layer + 2 rotation parameters per qubit final
        return (self.n_layers + 1) * self.n_qubits * 2

    def apply_circuit(self, params: np.ndarray):
        """Builds standard HEA circuit in PennyLane."""
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")

        # Initial state |0...0>
        p_idx = 0
        for l in range(self.n_layers):
            # 1Q Rotations
            for i in range(self.n_qubits):
                qml.RY(params[p_idx], wires=i)
                qml.RZ(params[p_idx + 1], wires=i)
                p_idx += 2

            # Entangling linear CNOT chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Final rotation layer
        for i in range(self.n_qubits):
            qml.RY(params[p_idx], wires=i)
            qml.RZ(params[p_idx + 1], wires=i)
            p_idx += 2


class QAOABaselineAnsatz:
    """
    QAOA-style alternating cost and mixer ansatz.
    """

    def __init__(self, n_qubits: int, n_layers: int = 3, couplings: Optional[List[Tuple[int, int]]] = None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.couplings = couplings or [(i, i + 1) for i in range(n_qubits - 1)]

    def num_params(self) -> int:
        return 2 * self.n_layers

    def apply_circuit(self, params: np.ndarray):
        """Applies QAOA circuit: alternating ZZ cost evolution and X mixer."""
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")

        # Equal superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)

        for l in range(self.n_layers):
            gamma = params[2 * l]
            beta = params[2 * l + 1]

            # Cost unitary: exp(-i * gamma * ZZ)
            for i, j in self.couplings:
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma, wires=j)
                qml.CNOT(wires=[i, j])

            # Mixer unitary: exp(-i * beta * X)
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)
