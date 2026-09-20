"""
Many-body spin models and SU(2) representation-theory helpers.

* Heisenberg XXX / XXZ Hamiltonians on several lattice geometries, buildable
  with or without PennyLane.
* Total-spin generators S^α, the Casimir S², total-spin sector decomposition.
* Singlet product state (Bell-singlet pairs), which lies in the S = 0 sector.
"""

from __future__ import annotations

import math
from functools import reduce
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:  # pragma: no cover
    HAS_PENNYLANE = False
    qml = None

_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)
PAULIS = {"I": _I2, "X": _SX, "Y": _SY, "Z": _SZ}

LATTICES = ("1d_chain", "ring", "2d_grid", "triangular", "kagome")


def pauli_string_matrix(spec: Dict[int, str], n_qubits: int) -> np.ndarray:
    """Dense 2ⁿ×2ⁿ matrix of a Pauli string given as {qubit: 'X'|'Y'|'Z'}."""
    ops = [PAULIS[spec.get(k, "I")] for k in range(n_qubits)]
    return reduce(np.kron, ops)


def compute_singlet_dimension(n_qubits: int) -> int:
    """Dimension of the S = 0 sector of N spin-½: the Catalan number C_{N/2} (0 for odd N)."""
    if n_qubits % 2 != 0:
        return 0
    k = n_qubits // 2
    return math.comb(n_qubits, k) // (k + 1)


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #
def lattice_edges(
    n_qubits: int,
    lattice_type: str = "1d_chain",
    periodic: bool = False,
    shape: Optional[Tuple[int, int]] = None,
) -> List[Tuple[int, int]]:
    """
    Nearest-neighbour bonds for a lattice on ``n_qubits`` sites.

    * ``1d_chain``  — open chain (closed if ``periodic``)
    * ``ring``      — closed chain
    * ``2d_grid``   — rectangular ``shape=(rows, cols)`` (default ``(2, n/2)``), optional periodic wrap
    * ``triangular``— triangular ladder: chain bonds plus next-nearest ``(i, i+2)`` diagonals
    * ``kagome``    — corner-sharing triangles: sites grouped in threes, three bonds per triangle
                      and one link between consecutive triangles (``n`` must be a multiple of 3)
    """
    n = int(n_qubits)
    if lattice_type not in LATTICES:
        raise ValueError(f"lattice_type must be one of {LATTICES}, got {lattice_type!r}")
    edges: List[Tuple[int, int]] = []
    if lattice_type in ("1d_chain", "ring"):
        edges = [(i, i + 1) for i in range(n - 1)]
        if (periodic or lattice_type == "ring") and n > 2:
            edges.append((n - 1, 0))
    elif lattice_type == "2d_grid":
        rows, cols = shape if shape is not None else (2, n // 2)
        if rows * cols != n:
            raise ValueError(f"shape {rows}x{cols} does not match n_qubits={n}")
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    edges.append((idx, idx + 1))
                elif periodic and cols > 2:
                    edges.append((idx, r * cols))
                if r + 1 < rows:
                    edges.append((idx, idx + cols))
                elif periodic and rows > 2:
                    edges.append((idx, c))
    elif lattice_type == "triangular":
        edges = [(i, i + 1) for i in range(n - 1)] + [(i, i + 2) for i in range(n - 2)]
        if periodic and n > 3:
            edges += [(n - 1, 0), (n - 2, 0), (n - 1, 1)]
    elif lattice_type == "kagome":
        if n % 3 != 0:
            raise ValueError("kagome lattice needs n_qubits to be a multiple of 3")
        n_tri = n // 3
        for t in range(n_tri):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [(a, b), (b, c), (a, c)]
            if t + 1 < n_tri:
                edges.append((c, 3 * (t + 1)))
        if periodic and n_tri > 1:
            edges.append((n - 1, 0))
    return edges


# --------------------------------------------------------------------------- #
# Heisenberg model
# --------------------------------------------------------------------------- #
class HeisenbergSpinChain:
    """
    H = J Σ_<ij> (X_i X_j + Y_i Y_j + Δ Z_i Z_j) on a chosen lattice.
    Δ = 1 is the SU(2)-symmetric XXX model; Δ ≠ 1 keeps only the U(1) symmetry.
    """

    def __init__(
        self,
        n_qubits: int,
        j_coupling: float = 1.0,
        anisotropy_delta: float = 1.0,
        periodic_boundary: bool = False,
        lattice_type: str = "1d_chain",
        shape: Optional[Tuple[int, int]] = None,
    ):
        if n_qubits % 2 != 0:
            raise ValueError(f"Number of qubits must be even for singlet physics, got {n_qubits}")
        self.n_qubits = int(n_qubits)
        self.j = float(j_coupling)
        self.delta = float(anisotropy_delta)
        self.pbc = bool(periodic_boundary)
        self.lattice_type = lattice_type
        self.shape = shape
        self.couplings = lattice_edges(self.n_qubits, lattice_type, periodic=self.pbc, shape=shape)

    def _build_lattice_couplings(self) -> List[Tuple[int, int]]:  # v0.2 name
        return list(self.couplings)

    def pauli_terms(self) -> List[Tuple[float, Dict[int, str]]]:
        """[(coefficient, {qubit: Pauli}), ...]"""
        terms = []
        for i, j in self.couplings:
            terms.append((self.j, {i: "X", j: "X"}))
            terms.append((self.j, {i: "Y", j: "Y"}))
            terms.append((self.j * self.delta, {i: "Z", j: "Z"}))
        return terms

    def get_pennylane_hamiltonian(self):
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane is required.")
        coeffs, ops = [], []
        for c, spec in self.pauli_terms():
            (i, pi), (j, pj) = sorted(spec.items())
            op_i = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[pi](i)
            op_j = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[pj](j)
            coeffs.append(c)
            ops.append(op_i @ op_j)
        return qml.Hamiltonian(coeffs, ops)

    def get_matrix(self, backend: str = "numpy") -> np.ndarray:
        """Dense 2ⁿ×2ⁿ Hamiltonian. ``backend='numpy'`` needs no PennyLane."""
        if backend == "pennylane":
            return qml.matrix(self.get_pennylane_hamiltonian(), wire_order=list(range(self.n_qubits)))
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim), dtype=np.complex128)
        for c, spec in self.pauli_terms():
            H += c * pauli_string_matrix(spec, self.n_qubits)
        return H

    def exact_ground_state(self) -> Tuple[float, np.ndarray]:
        evals, evecs = np.linalg.eigh(self.get_matrix())
        return float(evals[0]), evecs[:, 0]

    def spectrum(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.get_matrix())


# --------------------------------------------------------------------------- #
# SU(2) machinery
# --------------------------------------------------------------------------- #
def get_su2_generators(n_qubits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total spin S^α = ½ Σ_i σ_i^α."""
    dim = 2**n_qubits
    S = [np.zeros((dim, dim), dtype=np.complex128) for _ in range(3)]
    for i in range(n_qubits):
        for a, p in enumerate("XYZ"):
            S[a] += 0.5 * pauli_string_matrix({i: p}, n_qubits)
    return S[0], S[1], S[2]


def get_casimir_operator(n_qubits: int) -> np.ndarray:
    Sx, Sy, Sz = get_su2_generators(n_qubits)
    return Sx @ Sx + Sy @ Sy + Sz @ Sz


def total_spin_sectors(n_qubits: int) -> Dict[float, int]:
    """
    Multiplicity of each total-spin irrep: {S: number of copies}. Eigenvalues of
    S² are S(S+1); each irrep contributes 2S+1 degenerate eigenvalues.
    """
    evals = np.linalg.eigvalsh(get_casimir_operator(n_qubits))
    sectors: Dict[float, int] = {}
    for s2 in np.round(evals, 6):
        S = round((-1 + math.sqrt(1 + 4 * s2)) / 2, 6)
        sectors[S] = sectors.get(S, 0) + 1
    return {float(S): int(round(count / (2 * S + 1))) for S, count in sectors.items()}


def create_singlet_product_state_vector(n_qubits: int) -> np.ndarray:
    """|S⟩⊗|S⟩⊗… as a dense state vector, |S⟩ = (|01⟩ − |10⟩)/√2."""
    singlet = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2.0)
    return reduce(np.kron, [singlet] * (n_qubits // 2))


def create_singlet_product_state(n_qubits: int):
    """PennyLane gate sequence preparing |S⟩^{⊗N/2} on wires (0,1),(2,3),…"""
    for i in range(0, n_qubits, 2):
        qml.PauliX(wires=i + 1)
        qml.Hadamard(wires=i)
        qml.CNOT(wires=[i, i + 1])
        qml.PauliZ(wires=i)
