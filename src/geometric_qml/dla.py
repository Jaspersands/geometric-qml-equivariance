"""
Dynamical Lie Algebra (DLA) closure.

For a circuit built from gates exp(−iθ_k G_k), the DLA is
𝔤 = span_ℝ Lie({iG_k}) — the smallest real Lie algebra containing all gate
generators (fixed gates such as CNOT are included through their own
generators, giving the standard upper bound). Its dimension governs
expressibility and, through Var[∂C] ∝ 1/dim 𝔤, the presence of barren
plateaus (Larocca et al. 2022, Ragone et al. 2023).

Algorithm
---------
𝔤 is spanned by right-nested commutators [g_{i1}, [g_{i2}, … g_{ik}]] of the
generators, so it suffices to commute the *generators* with the elements added
in the previous round (not every pair of basis elements): with n_g generators
and final dimension D this costs n_g·D commutators instead of D². Each round is
batched — all candidates are projected off the current orthonormal basis with
one matrix product and orthonormalised with a pivoted QR.

Symmetry reduction
------------------
If every generator commutes with total spin, each one is block diagonal in the
Schur basis, G = ⊕_S A_S ⊗ 1_{2S+1}, and G ↦ ⊕_S A_S is a faithful
representation of the whole commutant. ``su2_reduced_generators`` computes the
blocks A_S (m_S × m_S, m_S the multiplicity of spin S) from the highest-weight
subspaces; the closure then runs on Σ_S m_S² numbers per element instead of
4ⁿ. For N = 8 that is 1 430 instead of 65 536, and the closure takes seconds.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse as sp


def compute_commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def orthonormalize_basis(basis: List[np.ndarray], tol: float = 1e-8) -> List[np.ndarray]:
    """Gram–Schmidt under ⟨A,B⟩ = Tr(A†B); drops (near-)dependent elements."""
    ortho: List[np.ndarray] = []
    for mat in basis:
        curr = np.array(mat, dtype=np.complex128)
        for b in ortho:
            curr = curr - np.trace(b.conj().T @ curr) * b
        norm = np.sqrt(np.real(np.trace(curr.conj().T @ curr)))
        if norm > tol:
            ortho.append(curr / norm)
    return ortho


# --------------------------------------------------------------------------- #
# block-diagonal (direct-sum) matrices
# --------------------------------------------------------------------------- #
class BlockMatrix:
    """A direct sum ⊕_b M_b of square blocks, with the operations the closure needs."""

    def __init__(self, blocks: Sequence[np.ndarray]):
        self.blocks = [np.asarray(b, dtype=np.complex128) for b in blocks]

    def __matmul__(self, other: "BlockMatrix") -> "BlockMatrix":
        return BlockMatrix([a @ b for a, b in zip(self.blocks, other.blocks)])

    def __sub__(self, other: "BlockMatrix") -> "BlockMatrix":
        return BlockMatrix([a - b for a, b in zip(self.blocks, other.blocks)])

    def vector(self) -> np.ndarray:
        return np.concatenate([b.reshape(-1) for b in self.blocks])

    @staticmethod
    def from_vector(v: np.ndarray, sizes: Sequence[int]) -> "BlockMatrix":
        out, k = [], 0
        for m in sizes:
            out.append(v[k:k + m * m].reshape(m, m))
            k += m * m
        return BlockMatrix(out)


def _as_vector(x) -> np.ndarray:
    return x.vector() if isinstance(x, BlockMatrix) else np.asarray(x).reshape(-1)


def _complex_to_real(v: np.ndarray) -> np.ndarray:
    return np.concatenate([v.real, v.imag], axis=0)


def _real_to_complex(v: np.ndarray) -> np.ndarray:
    h = v.shape[0] // 2
    return v[:h] + 1j * v[h:]


class DynamicalLieAlgebraAnalyzer:
    """
    Parameters
    ----------
    generators : Hermitian gate generators G_k — dense matrices or ``BlockMatrix``
        (e.g. from ``su2_reduced_generators``)
    max_iterations : safety cap on closure rounds
    max_dimension : stop (and report ``closed=False``) beyond this many elements
    tol : relative tolerance for linear independence
    hilbert_dim : dimension of the underlying Hilbert space when the generators are
        given in a reduced representation (used for ``full_su_dimension``)
    """

    def __init__(self, generators: List, max_iterations: int = 200, tol: float = 1e-8, max_dimension: int = 20000,
                 hilbert_dim: Optional[int] = None):
        self.block = isinstance(generators[0], BlockMatrix) if generators else False
        self.raw_generators = []
        for g in generators:
            if self.block:
                # remove the component along the identity (a global phase)
                sizes = [b.shape[0] for b in g.blocks]
                weights = np.array(g.weights if hasattr(g, "weights") else [1.0] * len(sizes))
                tr = sum(w * np.trace(b) for w, b in zip(weights, g.blocks)) / sum(w * m for w, m in zip(weights, sizes))
                self.raw_generators.append(BlockMatrix([1j * (b - tr * np.eye(b.shape[0])) for b in g.blocks]))
            else:
                g = np.asarray(g, dtype=np.complex128)
                g = g - (np.trace(g) / g.shape[0]) * np.eye(g.shape[0])
                self.raw_generators.append(1.0j * g)
        self.sizes = [b.shape[0] for b in self.raw_generators[0].blocks] if self.block and self.raw_generators else None
        self.hilbert_dim = hilbert_dim
        self.max_iter = int(max_iterations)
        self.tol = float(tol)
        self.max_dimension = int(max_dimension)

    def _unvec(self, v: np.ndarray):
        if self.block:
            return BlockMatrix.from_vector(v, self.sizes)
        d = int(round(np.sqrt(v.shape[0])))
        return v.reshape(d, d)

    def _orthonormal_new(self, Q: np.ndarray, C: np.ndarray, scale: float) -> np.ndarray:
        """Columns of C (real) orthogonal to Q, orthonormalised; rank decided by pivoted QR."""
        if C.shape[1] == 0:
            return C
        for _ in range(2):                                     # re-orthogonalise once for stability
            if Q.shape[1]:
                C = C - Q @ (Q.T @ C)
        norms = np.linalg.norm(C, axis=0)
        C = C[:, norms > self.tol * scale]
        if C.shape[1] == 0:
            return C
        q, r, _ = scipy.linalg.qr(C, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r))
        rank = int(np.sum(diag > self.tol * max(scale, diag[0] if len(diag) else 0)))
        new = q[:, :rank]
        if Q.shape[1]:
            new = new - Q @ (Q.T @ new)
            new, _ = np.linalg.qr(new)
        return new

    def compute_dla_dimension(self) -> Dict[str, Any]:
        if not self.raw_generators:
            return {"dla_dimension": 0, "full_su_dimension": 0, "is_sub_exponential": True,
                    "iteration_dimensions": [0], "n_qubits": 0, "closed": True, "basis": []}
        gens = self.raw_generators
        G = np.stack([_complex_to_real(_as_vector(g)) for g in gens], axis=1)
        scale = float(np.max(np.linalg.norm(G, axis=0)))
        Q = self._orthonormal_new(np.zeros((G.shape[0], 0)), G, scale)
        frontier = Q.copy()
        iteration_dims = [Q.shape[1]]
        closed = False
        for _ in range(self.max_iter):
            cands = []
            for k in range(frontier.shape[1]):
                x = self._unvec(_real_to_complex(frontier[:, k]))
                for g in gens:
                    cands.append(_complex_to_real(_as_vector(compute_commutator(g, x))))
            C = np.stack(cands, axis=1)
            new = self._orthonormal_new(Q, C, scale)
            if new.shape[1] == 0:
                closed = True
                break
            Q = np.concatenate([Q, new], axis=1)
            frontier = new
            iteration_dims.append(Q.shape[1])
            if Q.shape[1] >= self.max_dimension:
                break
        dim = Q.shape[1]
        if self.hilbert_dim is not None:
            hdim = self.hilbert_dim
        elif self.block:
            hdim = int(sum(self.sizes))
        else:
            hdim = int(round(np.sqrt(G.shape[0] // 2)))
        n_qubits = int(round(np.log2(hdim))) if hdim > 0 else 0
        full = 4**n_qubits - 1
        basis = [self._unvec(_real_to_complex(Q[:, k])) for k in range(min(dim, 5000))]
        return {
            "dla_dimension": dim,
            "full_su_dimension": full,
            "is_sub_exponential": dim < 0.8 * full,
            "iteration_dimensions": iteration_dims,
            "n_qubits": n_qubits,
            "closed": closed,
            "basis": basis,
        }


# --------------------------------------------------------------------------- #
# SU(2) symmetry reduction
# --------------------------------------------------------------------------- #
def spin_multiplicities(n_qubits: int) -> Dict[float, int]:
    """m_S = C(N, N/2 − S) − C(N, N/2 − S − 1) for every total spin S."""
    n = int(n_qubits)
    out: Dict[float, int] = {}
    S = n / 2.0
    while S >= 0:
        k = int(round(n / 2.0 - S))
        m = math.comb(n, k) - (math.comb(n, k - 1) if k >= 1 else 0)
        out[float(S)] = int(m)
        S -= 1.0
    return out


def _highest_weight_isometries(n_qubits: int) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """
    For each S: (basis indices of the S_z = S sector, isometry V_S onto the
    S² = S(S+1) eigenspace within it). Columns of V_S span the multiplicity space.
    """
    n = int(n_qubits)
    out = {}
    for S, m in spin_multiplicities(n).items():
        n_down = int(round(n / 2.0 - S))                  # qubit value 1 = spin down
        idx = np.array([b for b in range(2**n) if bin(b).count("1") == n_down], dtype=np.int64)
        pos = {int(b): i for i, b in enumerate(idx)}
        # S² = 3N/4 + ½Σ_{i<j} σ_i·σ_j and σ_i·σ_j = 2P_ij − 1  ⇒  S² = −N(N−4)/4 + Σ_{i<j} P_ij
        dim = len(idx)
        rows, cols, vals = [], [], []
        for a, b in enumerate(idx):
            b = int(b)
            diag = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    bi, bj = (b >> (n - 1 - i)) & 1, (b >> (n - 1 - j)) & 1
                    if bi == bj:
                        diag += 1.0                        # P_ij|b⟩ = |b⟩
                    else:
                        c = b ^ ((1 << (n - 1 - i)) | (1 << (n - 1 - j)))
                        rows.append(pos[c]); cols.append(a); vals.append(1.0)
            rows.append(a); cols.append(a); vals.append(diag)
        P = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim)).toarray()
        S2 = P - n * (n - 4) / 4.0 * np.eye(dim)
        w, V = np.linalg.eigh(S2)
        sel = np.abs(w - S * (S + 1)) < 1e-8
        if int(sel.sum()) != m:
            raise RuntimeError(f"multiplicity mismatch for S={S}: {int(sel.sum())} vs {m}")
        out[S] = (idx, V[:, sel])
    return out


def su2_reduced_generators(generators: Sequence, n_qubits: int) -> List[BlockMatrix]:
    """
    Blocks A_S = V_S† G V_S of SU(2)-invariant generators (dense arrays or scipy
    sparse matrices on 2ⁿ). The returned ``BlockMatrix`` objects carry
    ``weights`` = 2S+1 so traces are computed as in the full space.
    """
    iso = _highest_weight_isometries(n_qubits)
    out = []
    for g in generators:
        blocks, weights = [], []
        for S, (idx, V) in iso.items():
            sub = g[idx][:, idx]
            sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
            blocks.append(V.conj().T @ sub @ V)
            weights.append(2 * S + 1)
        bm = BlockMatrix(blocks)
        bm.weights = weights
        out.append(bm)
    return out


def swap_generator_sparse(i: int, j: int, n_qubits: int) -> sp.csr_matrix:
    """The SWAP operator P_ij = (1 + X_iX_j + Y_iY_j + Z_iZ_j)/2 as a sparse permutation matrix."""
    n = int(n_qubits)
    dim = 2**n
    b = np.arange(dim)
    bi = (b >> (n - 1 - i)) & 1
    bj = (b >> (n - 1 - j)) & 1
    c = np.where(bi != bj, b ^ ((1 << (n - 1 - i)) | (1 << (n - 1 - j))), b)
    return sp.csr_matrix((np.ones(dim), (c, b)), shape=(dim, dim))


def equivariant_dla_dimension_by_closure(edges: Sequence[Tuple[int, int]], n_qubits: int, **kwargs) -> Dict[str, Any]:
    """DLA of independent exchange gates on ``edges``, computed in the SU(2)-reduced representation."""
    gens = [swap_generator_sparse(i, j, n_qubits) for i, j in edges]
    red = su2_reduced_generators(gens, n_qubits)
    return DynamicalLieAlgebraAnalyzer(red, hilbert_dim=2**n_qubits, **kwargs).compute_dla_dimension()


def analyze_ansatz(ansatz, **kwargs) -> Dict[str, Any]:
    """Run the closure on ``ansatz.generators()``; returns the analyzer's result dict."""
    return DynamicalLieAlgebraAnalyzer(ansatz.generators(), **kwargs).compute_dla_dimension()


def equivariant_dla_dimension_formula(n_qubits: int) -> int:
    """
    Closed form for the DLA of exchange-generated SU(2)-equivariant circuits on
    a connected graph with independent bond angles:

        dim 𝔤 = Σ_S (m_S² − 1) + 1,

    where m_S is the multiplicity of the total-spin-S irrep (Schur–Weyl duality:
    the exchange generators span the commutant of SU(2), and their Lie closure is
    ⊕_S su(m_S) plus one central element). Verified numerically against the
    closure for N = 2, 4, 6 (dense) and N = 8 (SU(2)-reduced). Note m_S grows like a Catalan number, so the DLA is
    exponentially smaller than su(4ⁿ) but not polynomial — full barren-plateau
    freedom needs additional (e.g. S_N weight-sharing) symmetry.
    """
    sectors = spin_multiplicities(n_qubits)
    return int(sum(m * m - 1 for m in sectors.values()) + 1)
