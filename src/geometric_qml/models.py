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


def pauli_string_sparse(spec: Dict[int, str], n_qubits: int):
    """The same Pauli string as a scipy.sparse CSR matrix (one non-zero per column)."""
    import scipy.sparse as sp

    n = int(n_qubits)
    b = np.arange(2**n)
    flip = 0
    phase = np.ones(2**n, dtype=np.complex128)
    for q, p in spec.items():
        bit = (b >> (n - 1 - q)) & 1
        if p in ("X", "Y"):
            flip |= 1 << (n - 1 - q)
        if p == "Z":
            phase *= np.where(bit == 1, -1.0, 1.0)
        elif p == "Y":                                   # Y|0⟩ = i|1⟩, Y|1⟩ = −i|0⟩
            phase *= np.where(bit == 1, -1j, 1j)
    return sp.csr_matrix((phase, (b ^ flip, b)), shape=(2**n, 2**n))


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
    * ``kagome``    — the kagome lattice on ``shape=(Lx, Ly)`` unit cells of three sites
                      (default: the 2×2-cell, 12-site cluster; ``periodic`` gives the torus,
                      on which every site has coordination 4 and belongs to exactly two
                      triangles). See ``kagome_edges``.
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
        if shape is None:
            if n != 12:
                raise ValueError("kagome needs shape=(Lx, Ly) with n_qubits = 3·Lx·Ly; the default is the 12-site 2×2 cluster")
            shape = (2, 2)
        lx, ly = shape
        if 3 * lx * ly != n:
            raise ValueError(f"kagome shape {lx}x{ly} has {3 * lx * ly} sites, not {n}")
        edges = kagome_edges(lx, ly, periodic=periodic)
    return edges


def kagome_edges(lx: int, ly: int, periodic: bool = False) -> List[Tuple[int, int]]:
    """
    Kagome bonds on an ``lx × ly`` grid of unit cells (sites A, B, C = 3c, 3c+1, 3c+2
    for cell c = x + lx·y). With lattice vectors a₁ = (2, 0), a₂ = (1, √3) and
    sublattice offsets A = 0, B = (1, 0), C = (½, √3/2), the bonds are the
    up-triangle A–B–C inside every cell and the down triangle
    {B(x, y), A(x+1, y), C(x+1, y−1)} between cells. ``periodic`` wraps both
    directions (needs lx, ly ≥ 2 for a simple graph).
    """
    def site(x: int, y: int, s: int) -> Optional[int]:
        if periodic:
            x, y = x % lx, y % ly
        elif not (0 <= x < lx and 0 <= y < ly):
            return None
        return 3 * (x + lx * y) + s

    bonds = set()
    for y in range(ly):
        for x in range(lx):
            a, b, c = site(x, y, 0), site(x, y, 1), site(x, y, 2)
            bonds |= {tuple(sorted(e)) for e in ((a, b), (b, c), (a, c))}
            # down triangle to the right / below
            for u, v in ((site(x, y, 1), site(x + 1, y, 0)),
                         (site(x + 1, y, 0), site(x + 1, y - 1, 2)),
                         (site(x, y, 1), site(x + 1, y - 1, 2))):
                if u is not None and v is not None and u != v:
                    bonds.add(tuple(sorted((u, v))))
    return sorted(bonds)


def kagome_positions(lx: int, ly: int) -> np.ndarray:
    """Planar coordinates of the kagome sites (for plotting)."""
    out = []
    for y in range(ly):
        for x in range(lx):
            r = np.array([2.0 * x + y, np.sqrt(3.0) * y])
            out += [r, r + [1.0, 0.0], r + [0.5, np.sqrt(3.0) / 2]]
    return np.array(out)


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

    def get_sparse_matrix(self):
        """The Hamiltonian as a scipy.sparse CSR matrix (built term by term, no dense 2ⁿ objects)."""
        H = None
        for c, spec in self.pauli_terms():
            term = c * pauli_string_sparse(spec, self.n_qubits)
            H = term if H is None else H + term
        return H.tocsr()

    def get_matrix(self, backend: str = "numpy") -> np.ndarray:
        """Dense 2ⁿ×2ⁿ Hamiltonian. ``backend='numpy'`` needs no PennyLane."""
        if backend == "pennylane":
            return qml.matrix(self.get_pennylane_hamiltonian(), wire_order=list(range(self.n_qubits)))
        return self.get_sparse_matrix().toarray()

    def exact_ground_state(self) -> Tuple[float, np.ndarray]:
        """Ground energy and state; Lanczos beyond 10 qubits."""
        if self.n_qubits > 10:
            import scipy.sparse.linalg as spla

            w, v = spla.eigsh(self.get_sparse_matrix(), k=1, which="SA", tol=1e-10)
            return float(w[0]), v[:, 0]
        evals, evecs = np.linalg.eigh(self.get_matrix())
        return float(evals[0]), evecs[:, 0]

    def spectrum(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.get_matrix())


# --------------------------------------------------------------------------- #
# SU(2) machinery
# --------------------------------------------------------------------------- #
def get_su2_generators(n_qubits: int, sparse: bool = False):
    """Total spin S^α = ½ Σ_i σ_i^α (dense, or scipy.sparse with ``sparse=True``)."""
    S = []
    for p in "XYZ":
        M = sum(0.5 * pauli_string_sparse({i: p}, n_qubits) for i in range(n_qubits)).tocsr()
        S.append(M if sparse else M.toarray())
    return S[0], S[1], S[2]


def get_casimir_operator(n_qubits: int) -> np.ndarray:
    Sx, Sy, Sz = get_su2_generators(n_qubits)
    return Sx @ Sx + Sy @ Sy + Sz @ Sz


def total_spin_sectors(n_qubits: int) -> Dict[float, int]:
    """
    Multiplicity of each total-spin irrep: {S: number of copies},
    m_S = C(N, N/2 − S) − C(N, N/2 − S − 1).
    """
    from .dla import spin_multiplicities

    return spin_multiplicities(n_qubits)


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
