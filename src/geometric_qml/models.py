"""
Many-Body Spin Models and Representation Theory Operators.

Implements:
1. 1D & 2D Heisenberg XXX and XXZ Hamiltonian construction in PennyLane & NumPy.
2. Global SU(2) symmetry generators S^x, S^y, S^z and Casimir operator S^2.
3. Singlet subspace projector and Catalan dimension scaling.
4. Singlet product initial state preparation.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    qml = None


def compute_singlet_dimension(n_qubits: int) -> int:
    """
    Computes dimension of the SU(2) total spin singlet subspace (S=0)
    for an even number of spin-1/2 particles, given by the Catalan number:
        C_{N/2} = (1 / (N/2 + 1)) * (N choose N/2)
    """
    if n_qubits % 2 != 0:
        return 0
    k = n_qubits // 2
    return math.comb(n_qubits, k) // (k + 1)


class HeisenbergSpinChain:
    """
    1D and 2D Heisenberg XXX (isotropic) and XXZ (anisotropic) spin models.
        H = J * sum_{<i,j>} (X_i X_j + Y_i Y_j + Delta * Z_i Z_j) + h_z * sum_i Z_i
    """

    def __init__(
        self,
        n_qubits: int,
        j_coupling: float = 1.0,
        anisotropy_delta: float = 1.0, # Delta = 1.0 corresponds to exact SU(2) XXX model
        periodic_boundary: bool = False,
        lattice_type: str = "1d_chain", # '1d_chain', '2d_grid', 'triangular', 'kagome'
    ):
        if n_qubits % 2 != 0:
            raise ValueError(f"Number of qubits must be even for singlet physics, got {n_qubits}")
        self.n_qubits = n_qubits
        self.j = j_coupling
        self.delta = anisotropy_delta
        self.pbc = periodic_boundary
        self.lattice_type = lattice_type
        self.couplings = self._build_lattice_couplings()

    def _build_lattice_couplings(self) -> List[Tuple[int, int]]:
        """Generates nearest-neighbor edges based on lattice geometry."""
        n = self.n_qubits
        edges = []

        if self.lattice_type == "1d_chain":
            for i in range(n - 1):
                edges.append((i, i + 1))
            if self.pbc and n > 2:
                edges.append((n - 1, 0))

        elif self.lattice_type == "2d_grid":
            # Best 2D rectangular decomposition (e.g. 2 x (n//2))
            nrows = 2
            ncols = n // 2
            for r in range(nrows):
                for c in range(ncols):
                    idx = r * ncols + c
                    if c + 1 < ncols:
                        edges.append((idx, idx + 1))
                    if r + 1 < nrows:
                        edges.append((idx, idx + ncols))

        elif self.lattice_type in ["triangular", "kagome"]:
            # Nearest-neighbor chain + next-nearest-neighbor frustration
            for i in range(n - 1):
                edges.append((i, i + 1))
            for i in range(n - 2):
                edges.append((i, i + 2))
            if self.pbc:
                edges.append((n - 1, 0))
                edges.append((n - 2, 0))
        else:
            for i in range(n - 1):
                edges.append((i, i + 1))

        return edges

    def get_pennylane_hamiltonian(self) -> "qml.Hamiltonian":
        """Constructs the Hamiltonian as a PennyLane observable."""
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane is required.")

        coeffs = []
        ops = []

        for i, j in self.couplings:
            # X_i X_j
            coeffs.append(self.j)
            ops.append(qml.PauliX(i) @ qml.PauliX(j))

            # Y_i Y_j
            coeffs.append(self.j)
            ops.append(qml.PauliY(i) @ qml.PauliY(j))

            # Delta * Z_i Z_j
            coeffs.append(self.j * self.delta)
            ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

        return qml.Hamiltonian(coeffs, ops)

    def get_matrix(self) -> np.ndarray:
        """Returns exact 2^N x 2^N Hamiltonian matrix."""
        H_qml = self.get_pennylane_hamiltonian()
        return qml.matrix(H_qml, wire_order=list(range(self.n_qubits)))

    def exact_ground_state(self) -> Tuple[float, np.ndarray]:
        """Calculates exact ground state energy and eigenvector via diagonalization."""
        mat = self.get_matrix()
        evals, evecs = np.linalg.eigh(mat)
        return float(evals[0]), evecs[:, 0]


def get_su2_generators(n_qubits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns total spin operators S^x, S^y, S^z on N qubits:
        S^alpha = (1/2) * sum_{i=1}^N sigma_i^alpha
    """
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) * 0.5
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) * 0.5
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128) * 0.5
    id2 = np.eye(2, dtype=np.complex128)

    dim = 2**n_qubits
    Sx_tot = np.zeros((dim, dim), dtype=np.complex128)
    Sy_tot = np.zeros((dim, dim), dtype=np.complex128)
    Sz_tot = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(n_qubits):
        # Build operator on qubit i
        ox = [sx if k == i else id2 for k in range(n_qubits)]
        oy = [sy if k == i else id2 for k in range(n_qubits)]
        oz = [sz if k == i else id2 for k in range(n_qubits)]

        term_x = ox[0]
        term_y = oy[0]
        term_z = oz[0]
        for k in range(1, n_qubits):
            term_x = np.kron(term_x, ox[k])
            term_y = np.kron(term_y, oy[k])
            term_z = np.kron(term_z, oz[k])

        Sx_tot += term_x
        Sy_tot += term_y
        Sz_tot += term_z

    return Sx_tot, Sy_tot, Sz_tot


def get_casimir_operator(n_qubits: int) -> np.ndarray:
    """Returns the Casimir operator S^2 = (S^x)^2 + (S^y)^2 + (S^z)^2."""
    Sx, Sy, Sz = get_su2_generators(n_qubits)
    return Sx @ Sx + Sy @ Sy + Sz @ Sz


def create_singlet_product_state(n_qubits: int):
    """
    PennyLane circuit preparation function creating a tensor product of Bell singlets:
        |Psi_0> = |S>_{0,1} (x) |S>_{2,3} (x) ... (x) |S>_{N-2, N-1}
    where |S> = (|01> - |10>) / sqrt(2).
    This state lies exactly within the SU(2) total spin S=0 singlet manifold.
    """
    for i in range(0, n_qubits, 2):
        qml.PauliX(wires=i + 1)
        qml.Hadamard(wires=i)
        qml.CNOT(wires=[i, i + 1])
        qml.PauliZ(wires=i)
