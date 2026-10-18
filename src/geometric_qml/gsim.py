"""
Lie-algebraic simulation (g-sim) of circuits whose observable lies in the DLA.

If O ∈ i𝔤 then U†OU stays in i𝔤 for every circuit built from the DLA's gates,
so the loss can be propagated in the adjoint representation, whose dimension
is dim 𝔤 rather than 2^N (Goh, Larocca, Cincio, Cerezo & Sauvage,
arXiv:2308.01432). With an orthonormal Hermitian basis {B_a} of i𝔤,

    O = Σ_a o_a B_a,    e^{iθG} B_a e^{−iθG} = Σ_b [exp(θ Γ_G)]_{ba} B_b,
    (Γ_G)_{ba} = Tr[B_b · i[G, B_a]],    ℓ = Tr ρ · Tr O / d + Σ_a o_a Tr[ρ B_a].

Choosing the basis adapted to the ideals makes every Γ_G block diagonal, one
block of size dim 𝔤_j per simple ideal; the centre does not move.

``permutation_equivariant_simulator`` builds this for the S_N-equivariant
circuit, whose DLA ⊕_S su(2S+1) ⊕ ℝ·C has dimension C(N+3, 3) − ⌊N/2⌋. Only
spin matrices of size up to N+1 are needed: at N = 24 (dimension 2 913) a
thousand 32-layer circuits take about ten seconds, where a statevector would
hold 2^24 amplitudes per circuit. This is the classical-simulability side of
the trade-off discussed by Cerezo et al., arXiv:2312.09121.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dla import spin_matrices, spin_multiplicities


def gell_mann_basis(d: int) -> List[np.ndarray]:
    """Orthonormal (Tr λ_a λ_b = δ_ab) traceless Hermitian basis of i·su(d), d² − 1 elements."""
    out = []
    for j in range(d):
        for k in range(j + 1, d):
            s = np.zeros((d, d), dtype=np.complex128)
            s[j, k] = s[k, j] = 1 / math.sqrt(2)
            a = np.zeros((d, d), dtype=np.complex128)
            a[j, k], a[k, j] = -1j / math.sqrt(2), 1j / math.sqrt(2)
            out += [s, a]
    for l in range(1, d):
        diag = np.zeros(d)
        diag[:l] = 1.0
        diag[l] = -l
        out.append(np.diag(diag / math.sqrt(l * (l + 1))).astype(np.complex128))
    return out


class LieAlgebraicSimulator:
    """
    Expectation values in the adjoint representation.

    Parameters
    ----------
    gammas : gammas[k][j] is the (dim 𝔤_j × dim 𝔤_j) block of Γ for generator k on ideal j
    e_blocks : Tr[ρ B_a] for the basis elements of each ideal
    o_blocks : coefficients o_a of the observable on each ideal
    constant : the part of ℓ that no circuit changes (identity and centre)
    """

    def __init__(self, gammas: Sequence[Sequence[np.ndarray]], e_blocks: Sequence[np.ndarray],
                 o_blocks: Sequence[np.ndarray], constant: float = 0.0):
        self.e = [np.asarray(e, dtype=float) for e in e_blocks]
        self.o = [np.asarray(o, dtype=float) for o in o_blocks]
        self.constant = float(constant)
        self.dimension = sum(len(e) for e in self.e)
        # exp(θΓ) = V exp(−iθw) V† with iΓ = V w V†
        self._eig: List[List[Tuple[np.ndarray, np.ndarray]]] = []
        for per_gen in gammas:
            row = []
            for G in per_gen:
                w, V = np.linalg.eigh(1j * np.asarray(G, dtype=float))
                row.append((w, V))
            self._eig.append(row)

    def predicted_variance(self) -> float:
        """Σ_j P_j(ρ) P_j(O) / dim 𝔤_j, with P_j(ρ) = Σ_a e_a² and P_j(O) = Σ_a o_a² on ideal j."""
        return float(sum(np.dot(e, e) * np.dot(o, o) / len(e) for e, o in zip(self.e, self.o)))

    def expectation(self, sequence: Sequence[int], params: np.ndarray) -> np.ndarray:
        """
        ℓ for the circuit U = U_m ⋯ U_1, U_j = exp(−iθ_j G_{sequence[j]}).
        ``params`` has shape (m,) or (n_samples, m); returns one value per sample.
        """
        th = np.atleast_2d(np.asarray(params, dtype=float))
        total = np.full(th.shape[0], self.constant)
        for j, (e, o) in enumerate(zip(self.e, self.o)):
            if not np.any(e) or not np.any(o):
                continue
            # coefficients in each generator's eigenbasis are recomputed per gate; keep vectors complex
            vec = np.tile(o[:, None], (1, th.shape[0])).astype(np.complex128)
            for pos in range(len(sequence) - 1, -1, -1):      # Heisenberg picture: last gate first
                w, V = self._eig[sequence[pos]][j]
                vec = V @ (np.exp(-1j * np.outer(w, th[:, pos])) * (V.conj().T @ vec))
            total += np.real(e @ vec)
        return total


def _block_gamma(G: np.ndarray, basis: List[np.ndarray], Lmat: np.ndarray) -> np.ndarray:
    cols = [np.real(Lmat @ (1j * (G @ B - B @ G)).ravel()) for B in basis]
    return np.stack(cols, axis=1)


def permutation_equivariant_simulator(n_qubits: int, initial_weight: Optional[int] = None,
                                      observable: str = "xx", dynamics: bool = True) -> LieAlgebraicSimulator:
    """
    g-sim for the S_N-equivariant circuit (generators ΣX, ΣY, Σ_{k<l}ZZ, in that order).

    ``dynamics=False`` skips the Γ matrices (enough for ``predicted_variance``).
    ``initial_weight``: the input is a computational basis state with this many 1s
    (default N/2 rounded down, e.g. the Néel state); by S_N symmetry every such state
    gives the same losses. Its spin-space block is (m_S / C(N, w)) |S, M⟩⟨S, M| with
    M = N/2 − w. ``observable``: ``'xx'`` for Σ_{i<j} X_iX_j or ``'z'`` for Σ_i Z_i.
    """
    n = int(n_qubits)
    w = n // 2 if initial_weight is None else int(initial_weight)
    M = n / 2.0 - w
    mult = spin_multiplicities(n)
    gammas: List[List[np.ndarray]] = [[], [], []]
    e_blocks, o_blocks = [], []
    constant = 0.0
    for S in sorted(mult):
        d = int(round(2 * S + 1))
        m = mult[S]
        sx, sy, sz = spin_matrices(S)
        if observable == "xx":
            O_S = 2 * sx @ sx - 0.5 * n * np.eye(d)
        elif observable == "z":
            O_S = 2 * sz
        else:
            raise ValueError("observable must be 'xx' or 'z'")
        # state block on V_S
        R = np.zeros((d, d))
        if abs(M) <= S + 1e-9:
            R[int(round(S - M)), int(round(S - M))] = m / math.comb(n, w)
        constant += np.real(np.trace(R)) * np.real(np.trace(O_S)) / d          # identity part of each block
        if d < 2:
            continue
        basis = gell_mann_basis(d)
        Lmat = np.stack([b.ravel() for b in basis], axis=0).conj()
        if dynamics:
            gens = [2 * sx, 2 * sy, 2 * sz @ sz]
            for k, G in enumerate(gens):
                gammas[k].append(_block_gamma(G, basis, Lmat))
        # B_a = λ_a ⊗ 1_W / √m:  Tr[ρ B_a] = Tr[R λ_a]/√m,  o_a = Tr[B_a O] = √m Tr[λ_a O_S]
        e_blocks.append(np.real(Lmat.conj() @ R.T.ravel()) / math.sqrt(m))
        o_blocks.append(np.real(Lmat @ O_S.ravel()) * math.sqrt(m))
    return LieAlgebraicSimulator(gammas if dynamics else [[], [], []], e_blocks, o_blocks, constant)


def permutation_equivariant_sequence(n_layers: int) -> List[int]:
    return [k for _ in range(n_layers) for k in (0, 1, 2)]
