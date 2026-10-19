"""
NumPy statevector simulation of the three circuit families, batched over
parameter samples, with adjoint gradients for variational searches.

A circuit is a list of gates. Each gate is a tuple ``(kind, qubits)`` and,
except for ``cnot``, consumes one parameter in order:

==========  ===========================  ======================================
kind        unitary                      generator G (U = exp(−iθG))
==========  ===========================  ======================================
``exch``    exp(−iθ σ_i·σ_j)             X_iX_j + Y_iY_j + Z_iZ_j = 2 SWAP − 1
``ry``      RY(θ) = exp(−iθY/2)          Y/2
``rz``      RZ(θ) = exp(−iθZ/2)          Z/2
``cnot``    CNOT (fixed)                 none
``sumx``    exp(−iθ Σ_k X_k)             Σ_k X_k
``sumy``    exp(−iθ Σ_k Y_k)             Σ_k Y_k
``sumzz``   exp(−iθ Σ_{k<l} Z_kZ_l)      Σ_{k<l} Z_kZ_l
==========  ===========================  ======================================

Qubit 0 is the most significant bit, as in ``models.pauli_string_matrix``.
States are arrays of shape ``(n_samples, 2**n)``; ``params`` has shape
``(n_samples, n_params)``. Nothing here needs PennyLane.

``exch``, ``rz`` and ``sumzz`` conserve the number of qubits in |1⟩ (total
S_z), so a circuit made of them can also run inside one S_z sector
(``SzSector``, ``run_circuit_sector``, ``sample_losses(..., sector=True)``):
C(16, 8) = 12 870 amplitudes at N = 16 for the singlet product input, instead
of 65 536.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

Gate = Tuple[str, Tuple[int, ...]]

_PARAMETRISED = {"exch", "ry", "rz", "sumx", "sumy", "sumzz"}


# --------------------------------------------------------------------------- #
# circuit layouts (match the classes in ``ansatz``)
# --------------------------------------------------------------------------- #
def su2_equivariant_circuit(n_qubits: int, n_layers: int, couplings: Optional[Sequence[Tuple[int, int]]] = None) -> List[Gate]:
    """Exchange gates on every bond, one free angle per gate, repeated ``n_layers`` times."""
    bonds = list(couplings) if couplings is not None else [(i, i + 1) for i in range(n_qubits - 1)]
    return [("exch", (i, j)) for _ in range(n_layers) for i, j in bonds]


def hardware_efficient_circuit(n_qubits: int, n_layers: int) -> List[Gate]:
    """RY·RZ on every qubit and a CNOT ladder, ``n_layers`` times, then a final rotation layer."""
    gates: List[Gate] = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            gates += [("ry", (q,)), ("rz", (q,))]
        gates += [("cnot", (q, q + 1)) for q in range(n_qubits - 1)]
    for q in range(n_qubits):
        gates += [("ry", (q,)), ("rz", (q,))]
    return gates


def permutation_equivariant_circuit(n_qubits: int, n_layers: int) -> List[Gate]:
    """Layers of exp(−iθ₁ΣX) then exp(−iθ₂ΣY) then exp(−iθ₃Σ_{k<l}ZZ) (Schatzki et al. 2024, Eq. 9)."""
    return [g for _ in range(n_layers) for g in (("sumx", ()), ("sumy", ()), ("sumzz", ()))]


def num_params(gates: Sequence[Gate]) -> int:
    return sum(1 for kind, _ in gates if kind in _PARAMETRISED)


# --------------------------------------------------------------------------- #
# primitive operations on batched states
# --------------------------------------------------------------------------- #
def _split(psi: np.ndarray, n: int, q: int) -> np.ndarray:
    """View of a batched state as (samples, 2^q, 2, 2^(n−q−1))."""
    return psi.reshape(psi.shape[0], 2**q, 2, 2 ** (n - q - 1))


def _apply_1q(psi: np.ndarray, n: int, q: int, u: np.ndarray) -> np.ndarray:
    """Apply per-sample 2×2 matrices u (shape (S, 2, 2) or (2, 2)) to qubit q."""
    v = _split(psi, n, q)
    a, b = v[:, :, 0, :], v[:, :, 1, :]
    if u.ndim == 2:
        u = np.broadcast_to(u, (psi.shape[0], 2, 2))
    u = u[:, :, :, None, None]
    out = np.empty_like(v)
    out[:, :, 0, :] = u[:, 0, 0] * a + u[:, 0, 1] * b
    out[:, :, 1, :] = u[:, 1, 0] * a + u[:, 1, 1] * b
    return out.reshape(psi.shape)


def _swap(psi: np.ndarray, n: int, i: int, j: int) -> np.ndarray:
    t = psi.reshape((psi.shape[0],) + (2,) * n)
    return np.ascontiguousarray(np.swapaxes(t, 1 + i, 1 + j)).reshape(psi.shape)


@lru_cache(maxsize=64)
def _cnot_perm(n: int, c: int, t: int) -> np.ndarray:
    b = np.arange(2**n)
    flip = ((b >> (n - 1 - c)) & 1) << (n - 1 - t)
    return b ^ flip                                   # an involution: new[k] = old[perm[k]]


@lru_cache(maxsize=64)
def _sumzz_diag(n: int) -> np.ndarray:
    b = np.arange(2**n)
    ones = np.zeros(2**n, dtype=np.int64)
    for q in range(n):
        ones += (b >> (n - 1 - q)) & 1
    m = n - 2 * ones                                  # Σ_k z_k
    return ((m * m - n) // 2).astype(float)           # Σ_{k<l} z_k z_l


def _rot_matrices(theta: np.ndarray, pauli: str, half: bool) -> np.ndarray:
    """exp(−i a P) for each angle, a = θ/2 if ``half`` else θ; shape (S, 2, 2)."""
    a = 0.5 * theta if half else theta
    c, s = np.cos(a), np.sin(a)
    u = np.zeros((theta.shape[0], 2, 2), dtype=np.complex128)
    if pauli == "X":
        u[:, 0, 0] = u[:, 1, 1] = c
        u[:, 0, 1] = u[:, 1, 0] = -1j * s
    elif pauli == "Y":
        u[:, 0, 0] = u[:, 1, 1] = c
        u[:, 0, 1] = -s
        u[:, 1, 0] = s
    else:  # Z
        u[:, 0, 0] = c - 1j * s
        u[:, 1, 1] = c + 1j * s
    return u


def apply_gate(psi: np.ndarray, n: int, gate: Gate, theta: Optional[np.ndarray] = None, adjoint: bool = False) -> np.ndarray:
    """Apply one gate (or its inverse) with per-sample angles ``theta`` of shape (S,)."""
    kind, qs = gate
    if kind == "cnot":
        return psi[:, _cnot_perm(n, qs[0], qs[1])]
    th = -theta if adjoint else theta
    if kind == "exch":
        # exp(−iθ(2P − 1)) = e^{iθ}(cos 2θ − i sin 2θ P)
        sw = _swap(psi, n, qs[0], qs[1])
        ph = np.exp(1j * th)[:, None]
        return ph * (np.cos(2 * th)[:, None] * psi - 1j * np.sin(2 * th)[:, None] * sw)
    if kind == "ry":
        return _apply_1q(psi, n, qs[0], _rot_matrices(th, "Y", half=True))
    if kind == "rz":
        return _apply_1q(psi, n, qs[0], _rot_matrices(th, "Z", half=True))
    if kind in ("sumx", "sumy"):
        u = _rot_matrices(th, "X" if kind == "sumx" else "Y", half=False)
        for q in range(n):
            psi = _apply_1q(psi, n, q, u)
        return psi
    if kind == "sumzz":
        return psi * np.exp(-1j * th[:, None] * _sumzz_diag(n)[None, :])
    raise ValueError(f"unknown gate {kind!r}")


def apply_generator(psi: np.ndarray, n: int, gate: Gate) -> np.ndarray:
    """G ψ for the gate's generator (used by the adjoint gradient)."""
    kind, qs = gate
    if kind == "exch":
        return 2.0 * _swap(psi, n, qs[0], qs[1]) - psi
    if kind in ("ry", "rz"):
        p = np.array([[0, -0.5j], [0.5j, 0]]) if kind == "ry" else np.diag([0.5, -0.5]).astype(complex)
        return _apply_1q(psi, n, qs[0], p)
    if kind in ("sumx", "sumy"):
        p = np.array([[0, 1], [1, 0]], dtype=complex) if kind == "sumx" else np.array([[0, -1j], [1j, 0]])
        return sum(_apply_1q(psi, n, q, p) for q in range(n))
    if kind == "sumzz":
        return psi * _sumzz_diag(n)[None, :]
    raise ValueError(f"gate {kind!r} has no generator")


# --------------------------------------------------------------------------- #
# circuits
# --------------------------------------------------------------------------- #
def run_circuit(gates: Sequence[Gate], n: int, params: np.ndarray, psi0: np.ndarray) -> np.ndarray:
    """Final states for a batch of parameter vectors. ``psi0`` is one state (2^n,) or a batch."""
    params = np.atleast_2d(np.asarray(params, dtype=float))
    S = params.shape[0]
    psi = np.array(np.broadcast_to(psi0, (S, 2**n)), dtype=np.complex128)
    k = 0
    for g in gates:
        if g[0] == "cnot":
            psi = apply_gate(psi, n, g)
        else:
            psi = apply_gate(psi, n, g, params[:, k])
            k += 1
    if k != params.shape[1]:
        raise ValueError(f"circuit has {k} parameters, got {params.shape[1]}")
    return psi


def expectation(psi: np.ndarray, op) -> np.ndarray:
    """⟨ψ|O|ψ⟩ for each row of ``psi``; ``op`` is a dense or scipy.sparse Hermitian matrix."""
    opsi = (op @ psi.T).T
    return np.real(np.einsum("si,si->s", psi.conj(), opsi))


def energy_and_gradient(gates: Sequence[Gate], n: int, params: np.ndarray, psi0: np.ndarray, op) -> Tuple[float, np.ndarray]:
    """⟨O⟩ and its gradient by the adjoint method (one forward and one backward sweep)."""
    params = np.asarray(params, dtype=float)
    psi = run_circuit(gates, n, params[None, :], psi0)
    lam = (op @ psi.T).T
    energy = float(np.real(np.vdot(psi[0], lam[0])))
    grad = np.zeros_like(params)
    k = params.shape[0]
    for g in reversed(gates):
        if g[0] == "cnot":
            psi = apply_gate(psi, n, g)
            lam = apply_gate(lam, n, g)
            continue
        k -= 1
        grad[k] = 2.0 * float(np.imag(np.vdot(lam[0], apply_generator(psi, n, g)[0])))
        th = params[k:k + 1]
        psi = apply_gate(psi, n, g, th, adjoint=True)
        lam = apply_gate(lam, n, g, th, adjoint=True)
    return energy, grad


def sample_losses(gates: Sequence[Gate], n: int, psi0: np.ndarray, op, n_samples: int, seed: Optional[int] = None,
                  batch: int = 512, sector: bool = False) -> np.ndarray:
    """
    ℓ(θ) = ⟨ψ(θ)|O|ψ(θ)⟩ at ``n_samples`` parameter vectors drawn uniformly from [0, 2π).

    With ``sector=True`` the circuit runs inside the S_z sector that holds ψ₀; every
    gate must conserve S_z. The angles depend on the seed only, not on ``batch`` or ``sector``.
    """
    theta = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=(int(n_samples), num_params(gates)))
    return losses_at(gates, n, psi0, op, theta, batch, sector)


def losses_at(gates: Sequence[Gate], n: int, psi0: np.ndarray, op, theta: np.ndarray, batch: int = 512,
              sector: bool = False) -> np.ndarray:
    """ℓ(θ) for each row of ``theta``, ``batch`` rows at a time (see ``sample_losses``)."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    if sector:
        sec = SzSector.containing(psi0, n)
        phi0, block = sec.restrict(psi0), sec.block(op)
        run = lambda th: expectation(run_circuit_sector(gates, sec, th, phi0), block)  # noqa: E731
    else:
        run = lambda th: expectation(run_circuit(gates, n, th, psi0), op)  # noqa: E731
    return np.concatenate([run(theta[k:k + batch]) for k in range(0, theta.shape[0], batch)])


# --------------------------------------------------------------------------- #
# one S_z sector
# --------------------------------------------------------------------------- #
_SZ_CONSERVING = {"exch", "rz", "sumzz"}


class SzSector:
    """
    The computational basis states with ``n_ones`` qubits in |1⟩, in increasing order.

    An exchange gate maps this set to itself (SWAP permutes it), and RZ and ΣZZ are
    diagonal, so a state in the sector stays there. Arrays restricted to the sector
    have shape ``(n_samples, dim)`` with ``dim`` = C(n, n_ones).
    """

    def __init__(self, n_qubits: int, n_ones: int):
        self.n = int(n_qubits)
        self.n_ones = int(n_ones)
        b = np.arange(2**self.n, dtype=np.int64)
        pop = np.zeros_like(b)
        for q in range(self.n):
            pop += (b >> q) & 1
        self.states = b[pop == self.n_ones]
        self.dim = int(self.states.size)
        self._swaps: dict = {}

    @classmethod
    def containing(cls, psi: np.ndarray, n_qubits: int, tol: float = 1e-12) -> "SzSector":
        """The sector holding ``psi``; raises if ``psi`` has weight in more than one."""
        psi = np.asarray(psi)
        support = np.flatnonzero(np.abs(psi) > tol)
        if support.size == 0:
            raise ValueError("the state is zero")
        ones = {bin(int(k)).count("1") for k in support}
        if len(ones) > 1:
            raise ValueError(f"the state has weight in several S_z sectors ({sorted(ones)} qubits in |1⟩)")
        sec = cls(n_qubits, ones.pop())
        outside = np.linalg.norm(psi) ** 2 - np.linalg.norm(psi[sec.states]) ** 2
        if outside > tol:
            raise ValueError("the state has weight outside its S_z sector")
        return sec

    def bits(self, q: int) -> np.ndarray:
        """Value of qubit q in each basis state of the sector."""
        return (self.states >> (self.n - 1 - q)) & 1

    def swap_index(self, i: int, j: int) -> np.ndarray:
        """Positions k' with states[k'] = states[k] with qubits i and j exchanged."""
        key = (min(i, j), max(i, j))
        if key not in self._swaps:
            bi, bj = self.bits(i), self.bits(j)
            flip = (bi ^ bj) * ((1 << (self.n - 1 - i)) | (1 << (self.n - 1 - j)))
            self._swaps[key] = np.searchsorted(self.states, self.states ^ flip)
        return self._swaps[key]

    def restrict(self, psi: np.ndarray) -> np.ndarray:
        return np.asarray(psi)[..., self.states]

    def embed(self, phi: np.ndarray) -> np.ndarray:
        """Full 2^n-dimensional state(s) from sector amplitudes."""
        phi = np.asarray(phi)
        out = np.zeros(phi.shape[:-1] + (2**self.n,), dtype=np.complex128)
        out[..., self.states] = phi
        return out

    def block(self, op):
        """O restricted to the sector; ⟨ψ|O|ψ⟩ for ψ in the sector needs nothing else."""
        if sp.issparse(op):
            return op.tocsr()[self.states][:, self.states]
        return np.asarray(op)[np.ix_(self.states, self.states)]


def apply_gate_sector(phi: np.ndarray, sector: SzSector, gate: Gate, theta: np.ndarray) -> np.ndarray:
    """Apply one S_z-conserving gate to sector amplitudes ``phi`` (shape (S, dim))."""
    kind, qs = gate
    if kind == "exch":
        # exp(−iθ(2P − 1)) = e^{iθ}(cos 2θ − i sin 2θ P), and P permutes the sector
        ph = np.exp(1j * theta)
        a = (ph * np.cos(2 * theta))[:, None]
        b = (-1j * ph * np.sin(2 * theta))[:, None]
        return a * phi + b * phi[:, sector.swap_index(qs[0], qs[1])]
    if kind == "rz":
        z = 1.0 - 2.0 * sector.bits(qs[0])
        return phi * np.exp(-0.5j * theta[:, None] * z[None, :])
    if kind == "sumzz":
        return phi * np.exp(-1j * theta[:, None] * _sumzz_diag(sector.n)[sector.states][None, :])
    raise ValueError(f"gate {kind!r} does not conserve S_z")


def run_circuit_sector(gates: Sequence[Gate], sector: SzSector, params: np.ndarray, phi0: np.ndarray) -> np.ndarray:
    """``run_circuit`` inside one S_z sector: ``phi0`` and the result hold the sector amplitudes."""
    bad = sorted({kind for kind, _ in gates if kind not in _SZ_CONSERVING})
    if bad:
        raise ValueError(f"gates {bad} do not conserve S_z")
    params = np.atleast_2d(np.asarray(params, dtype=float))
    if params.shape[1] != len(gates):
        raise ValueError(f"circuit has {len(gates)} parameters, got {params.shape[1]}")
    phi = np.array(np.broadcast_to(phi0, (params.shape[0], sector.dim)), dtype=np.complex128)
    for k, g in enumerate(gates):
        phi = apply_gate_sector(phi, sector, g, params[:, k])
    return phi
