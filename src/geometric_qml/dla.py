"""
Dynamical Lie Algebra (DLA) closure.

For a circuit built from gates exp(−iθ_k G_k), the DLA is
𝔤 = span_ℝ Lie({iG_k}) — the smallest real Lie algebra containing all gate
generators (fixed gates such as CNOT are included through their own
generators, giving the standard upper bound). Its dimension governs
expressibility and, through Var[∂C] ∝ 1/dim 𝔤, the presence of barren
plateaus (Larocca et al. 2022, Ragone et al. 2023).

The closure is computed by repeatedly taking commutators of newly added
elements with the current orthonormal basis (Hilbert–Schmidt inner product)
until nothing new appears.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


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


class DynamicalLieAlgebraAnalyzer:
    """
    Parameters
    ----------
    generators : Hermitian gate generators G_k (dense matrices)
    max_iterations : safety cap on closure rounds
    max_dimension : stop (and report ``closed=False``) beyond this many elements
    tol : Hilbert–Schmidt tolerance for linear independence
    """

    def __init__(self, generators: List[np.ndarray], max_iterations: int = 40, tol: float = 1e-8, max_dimension: int = 2000):
        # Traceless parts only: the identity component is a global phase and is not
        # part of su(2ⁿ).
        self.raw_generators = []
        for g in generators:
            g = np.asarray(g, dtype=np.complex128)
            g = g - (np.trace(g) / g.shape[0]) * np.eye(g.shape[0])
            self.raw_generators.append(1.0j * g)
        self.max_iter = int(max_iterations)
        self.tol = float(tol)
        self.max_dimension = int(max_dimension)

    def compute_dla_dimension(self) -> Dict[str, Any]:
        basis = orthonormalize_basis(self.raw_generators, self.tol)
        if not basis:
            return {"dla_dimension": 0, "full_su_dimension": 0, "is_sub_exponential": True,
                    "iteration_dimensions": [0], "n_qubits": 0, "closed": True, "basis": []}
        dim = basis[0].shape[0]
        # flattened orthonormal vectors for fast projection
        B = np.stack([b.reshape(-1) for b in basis])            # (k, d²)
        new_elems = list(basis)
        iteration_dims = [len(basis)]
        closed = False
        for _ in range(self.max_iter):
            candidates = []
            for a in new_elems:
                for b in basis:
                    c = compute_commutator(a, b).reshape(-1)
                    c = c - B.T @ (B.conj() @ c)               # project out current span
                    if np.linalg.norm(c) > self.tol:
                        candidates.append(c)
            added = []
            for c in candidates:
                c = c - B.T @ (B.conj() @ c)
                for e in added:
                    c = c - e * (e.conj() @ c)
                nrm = np.linalg.norm(c)
                if nrm > self.tol:
                    added.append(c / nrm)
            if not added:
                closed = True
                break
            new_elems = [v.reshape(dim, dim) for v in added]
            basis.extend(new_elems)
            B = np.vstack([B, np.stack(added)])
            iteration_dims.append(len(basis))
            if len(basis) >= self.max_dimension:
                break
        n_qubits = int(round(np.log2(dim)))
        full = 4**n_qubits - 1
        return {
            "dla_dimension": len(basis),
            "full_su_dimension": full,
            "is_sub_exponential": len(basis) < 0.8 * full,
            "iteration_dimensions": iteration_dims,
            "n_qubits": n_qubits,
            "closed": closed,
            "basis": basis,
        }


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
    closure for N = 2, 4, 6. Note m_S grows like a Catalan number, so the DLA is
    exponentially smaller than su(4ⁿ) but not polynomial — full barren-plateau
    freedom needs additional (e.g. S_N weight-sharing) symmetry.
    """
    from .models import total_spin_sectors

    sectors = total_spin_sectors(n_qubits)
    return int(sum(m * m - 1 for m in sectors.values()) + 1)
