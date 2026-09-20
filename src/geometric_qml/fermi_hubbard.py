"""
Fermi–Hubbard chain with its SU(2)_spin × SU(2)_charge (η-pairing) symmetries.

    H = −t Σ_{⟨ij⟩σ} (c†_{iσ} c_{jσ} + h.c.) + U Σ_i n_{i↑} n_{i↓}        (standard)
    H = −t Σ (…) + U Σ_i (n_{i↑} − ½)(n_{i↓} − ½)                          (particle-hole symmetric)

Qubits are Jordan–Wigner mapped with ordering (site 0 ↑, site 0 ↓, site 1 ↑, …).
The spin SU(2) generators commute with both forms; the charge (η-pairing) SU(2)
generators commute with the particle-hole symmetric form on bipartite lattices.
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import PAULIS

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:  # pragma: no cover
    HAS_PENNYLANE = False
    qml = None

_SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=np.complex128)  # |0⟩⟨1|: annihilates an occupied mode


def jordan_wigner_annihilation(p: int, n_qubits: int) -> np.ndarray:
    """c_p = Z_0 ⋯ Z_{p−1} σ⁻_p."""
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
        self._c = [jordan_wigner_annihilation(p, self.n_qubits) for p in range(self.n_qubits)]

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

    def _n(self, p: int) -> np.ndarray:
        return self._c[p].conj().T @ self._c[p]

    def get_matrix(self) -> np.ndarray:
        dim = 2**self.n_qubits
        H = np.zeros((dim, dim), dtype=np.complex128)
        for i, j in self.bonds():
            for spin in (self.up, self.down):
                hop = self._c[spin(i)].conj().T @ self._c[spin(j)]
                H -= self.t * (hop + hop.conj().T)
        I = np.eye(dim, dtype=np.complex128)
        for i in range(self.n_sites):
            nu, nd = self._n(self.up(i)), self._n(self.down(i))
            if self.symmetric:
                H += self.u * (nu - 0.5 * I) @ (nd - 0.5 * I)
            else:
                H += self.u * nu @ nd
        return H

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
    def number_operator(self) -> np.ndarray:
        return sum(self._n(p) for p in range(self.n_qubits))

    def spin_generators(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """S^z = ½Σ(n↑−n↓), S⁺ = Σ c†↑ c↓ → (S^x, S^y, S^z)."""
        Sz = 0.5 * sum(self._n(self.up(i)) - self._n(self.down(i)) for i in range(self.n_sites))
        Sp = sum(self._c[self.up(i)].conj().T @ self._c[self.down(i)] for i in range(self.n_sites))
        Sm = Sp.conj().T
        return 0.5 * (Sp + Sm), (Sp - Sm) / (2j), Sz

    def charge_generators(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """η⁺ = Σ(−1)^i c†_{i↑} c†_{i↓}, η^z = ½Σ(n_i − 1) → (η^x, η^y, η^z)."""
        dim = 2**self.n_qubits
        eta_p = sum(((-1) ** i) * self._c[self.up(i)].conj().T @ self._c[self.down(i)].conj().T for i in range(self.n_sites))
        eta_m = eta_p.conj().T
        eta_z = 0.5 * (self.number_operator() - self.n_sites * np.eye(dim))
        return 0.5 * (eta_p + eta_m), (eta_p - eta_m) / (2j), eta_z

    def ground_energy(self, n_particles: Optional[int] = None) -> float:
        """Lowest eigenvalue, optionally within the fixed-particle-number sector."""
        H = self.get_matrix()
        if n_particles is None:
            return float(np.linalg.eigvalsh(H)[0])
        dim = 2**self.n_qubits
        keep = [b for b in range(dim) if bin(b).count("1") == n_particles]
        return float(np.linalg.eigvalsh(H[np.ix_(keep, keep)])[0])
