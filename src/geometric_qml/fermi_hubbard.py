"""
Fermi–Hubbard chain with its SU(2)_spin × SU(2)_charge (η-pairing) symmetries.

    H = −t Σ_{⟨ij⟩σ} (c†_{iσ} c_{jσ} + h.c.) + U Σ_i n_{i↑} n_{i↓}        (standard)
    H = −t Σ (…) + U Σ_i (n_{i↑} − ½)(n_{i↓} − ½)                          (particle-hole symmetric)

Qubits are Jordan–Wigner mapped with ordering (site 0 ↑, site 0 ↓, site 1 ↑, …).
The spin SU(2) generators commute with both forms; the charge (η-pairing) SU(2)
generators commute with the particle-hole symmetric form on bipartite lattices.

All operators are built as ``scipy.sparse`` matrices (a Jordan–Wigner string is
a signed permutation times a projector, so each has exactly 2^{2L−1} non-zeros);
dense arrays are produced only on request for small systems. Ground energies
use Lanczos (``eigsh``) inside a fixed (N, S_z) sector, so L = 8 (16 qubits)
takes seconds rather than the 70 GB per operator a dense build would need.
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .models import PAULIS

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:  # pragma: no cover
    HAS_PENNYLANE = False
    qml = None

_SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0⟩⟨1|: annihilates an occupied mode


def jordan_wigner_annihilation(p: int, n_qubits: int, sparse: bool = False):
    """c_p = Z_0 ⋯ Z_{p−1} σ⁻_p (dense by default; ``sparse=True`` for a CSR matrix)."""
    if sparse:
        n = int(n_qubits)
        b = np.arange(2**n)
        occ = (b >> (n - 1 - p)) & 1                          # qubit value 1 = occupied
        parity = np.zeros_like(b)
        for q in range(p):
            parity ^= (b >> (n - 1 - q)) & 1
        src = b[occ == 1]
        vals = np.where(parity[occ == 1] == 1, -1.0, 1.0)
        return sp.csr_matrix((vals.astype(np.complex128), (src ^ (1 << (n - 1 - p)), src)), shape=(2**n, 2**n))
    ops = [PAULIS["Z"]] * p + [_SIGMA_MINUS] + [PAULIS["I"]] * (n_qubits - p - 1)
    return reduce(np.kron, ops)


class FermiHubbardChain:
    """
    Parameters
    ----------
    n_sites : lattice sites L (2L qubits)
    hopping_t, on_site_u : t and U
    particle_hole_symmetric : use U(n↑−½)(n↓−½) instead of U n↑n↓
    periodic : close the chain into a ring (must be bipartite, i.e. even L, for η symmetry)
    """

    def __init__(
        self,
        n_sites: int = 2,
        hopping_t: float = 1.0,
        on_site_u: float = 4.0,
        particle_hole_symmetric: bool = False,
        periodic: bool = False,
    ):
        self.n_sites = int(n_sites)
        self.n_qubits = 2 * self.n_sites
        self.t = float(hopping_t)
        self.u = float(on_site_u)
        self.symmetric = bool(particle_hole_symmetric)
        self.periodic = bool(periodic)
        self._c = [jordan_wigner_annihilation(p, self.n_qubits, sparse=True) for p in range(self.n_qubits)]

    # ----------------------------------------------------------------- #
    def up(self, i: int) -> int:
        return 2 * i

    def down(self, i: int) -> int:
        return 2 * i + 1

    def bonds(self) -> List[Tuple[int, int]]:
        b = [(i, i + 1) for i in range(self.n_sites - 1)]
        if self.periodic and self.n_sites > 2:
            b.append((self.n_sites - 1, 0))
        return b

    def _n(self, p: int) -> sp.csr_matrix:
        return (self._c[p].conj().T @ self._c[p]).tocsr()

    @staticmethod
    def _out(M, sparse: bool, max_dense_qubits: int, n_qubits: int):
        if sparse:
            return M.tocsr()
        if n_qubits > max_dense_qubits:
            raise MemoryError(f"a dense {2**n_qubits}×{2**n_qubits} matrix is too large; pass sparse=True")
        return M.toarray()

    def get_sparse_matrix(self) -> sp.csr_matrix:
        dim = 2**self.n_qubits
        H = sp.csr_matrix((dim, dim), dtype=np.complex128)
        for i, j in self.bonds():
            for spin in (self.up, self.down):
                hop = self._c[spin(i)].conj().T @ self._c[spin(j)]
                H = H - self.t * (hop + hop.conj().T)
        I = sp.identity(dim, dtype=np.complex128, format="csr")
        for i in range(self.n_sites):
            nu, nd = self._n(self.up(i)), self._n(self.down(i))
            if self.symmetric:
                H = H + self.u * (nu - 0.5 * I) @ (nd - 0.5 * I)
            else:
                H = H + self.u * nu @ nd
        return H.tocsr()

    def get_matrix(self, sparse: bool = False, max_dense_qubits: int = 14):
        """The Hamiltonian; dense unless ``sparse=True`` (dense refuses beyond ``max_dense_qubits``)."""
        return self._out(self.get_sparse_matrix(), sparse, max_dense_qubits, self.n_qubits)

    def get_pennylane_hamiltonian(self):
        """Same operator as ``get_matrix`` expressed as a PennyLane Hamiltonian."""
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")
        coeffs, ops = [], []
        for i, j in self.bonds():
            for spin in (self.up, self.down):
                a, b = spin(i), spin(j)
                lo, hi = min(a, b), max(a, b)
                zs = [qml.PauliZ(k) for k in range(lo + 1, hi)]
                for P in (qml.PauliX, qml.PauliY):
                    op = P(lo)
                    for z in zs:
                        op = op @ z
                    op = op @ P(hi)
                    coeffs.append(-0.5 * self.t)
                    ops.append(op)
        for i in range(self.n_sites):
            u_q, d_q = self.up(i), self.down(i)
            # n↑ n↓ = ¼(1 − Z↑ − Z↓ + Z↑Z↓);  (n↑−½)(n↓−½) = ¼ Z↑Z↓
            coeffs.append(0.25 * self.u); ops.append(qml.PauliZ(u_q) @ qml.PauliZ(d_q))
            if not self.symmetric:
                coeffs += [-0.25 * self.u, -0.25 * self.u, 0.25 * self.u]
                ops += [qml.PauliZ(u_q), qml.PauliZ(d_q), qml.Identity(u_q)]
        return qml.Hamiltonian(coeffs, ops)

    # ----------------------------------------------------------------- #
    def number_operator(self, sparse: bool = False):
        N = sum(self._n(p) for p in range(self.n_qubits))
        return self._out(N, sparse, 14, self.n_qubits)

    def spin_generators(self, sparse: bool = False):
        """S^z = ½Σ(n↑−n↓), S⁺ = Σ c†↑ c↓ → (S^x, S^y, S^z)."""
        Sz = 0.5 * sum(self._n(self.up(i)) - self._n(self.down(i)) for i in range(self.n_sites))
        Sp = sum(self._c[self.up(i)].conj().T @ self._c[self.down(i)] for i in range(self.n_sites))
        Sm = Sp.conj().T
        return tuple(self._out(M, sparse, 14, self.n_qubits) for M in (0.5 * (Sp + Sm), (Sp - Sm) / (2j), Sz))

    def charge_generators(self, sparse: bool = False):
        """η⁺ = Σ(−1)^i c†_{i↑} c†_{i↓}, η^z = ½Σ(n_i − 1) → (η^x, η^y, η^z)."""
        dim = 2**self.n_qubits
        eta_p = sum(((-1) ** i) * self._c[self.up(i)].conj().T @ self._c[self.down(i)].conj().T for i in range(self.n_sites))
        eta_m = eta_p.conj().T
        eta_z = 0.5 * (self.number_operator(sparse=True) - self.n_sites * sp.identity(dim, format="csr"))
        return tuple(self._out(M, sparse, 14, self.n_qubits) for M in (0.5 * (eta_p + eta_m), (eta_p - eta_m) / (2j), eta_z))

    def sector_indices(self, n_particles: int, sz: Optional[float] = None) -> np.ndarray:
        """Computational-basis indices with the given particle number (and S_z)."""
        n = self.n_qubits
        b = np.arange(2**n)
        bits = (b[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
        keep = bits.sum(axis=1) == int(n_particles)
        if sz is not None:
            n_up, n_dn = bits[:, 0::2].sum(axis=1), bits[:, 1::2].sum(axis=1)
            keep &= np.isclose(0.5 * (n_up - n_dn), sz)
        return b[keep]

    def ground_energy(self, n_particles: Optional[int] = None, sz: Optional[float] = None) -> float:
        """
        Lowest eigenvalue, optionally within a fixed particle-number (and S_z) sector.
        Dense diagonalisation for sectors up to 2 000 states, Lanczos beyond.
        """
        H = self.get_sparse_matrix()
        if n_particles is not None:
            idx = self.sector_indices(n_particles, sz)
            H = H[idx][:, idx]
        if H.shape[0] <= 2000:
            return float(np.linalg.eigvalsh(H.toarray())[0])
        return float(spla.eigsh(H, k=1, which="SA", tol=1e-10)[0][0])
