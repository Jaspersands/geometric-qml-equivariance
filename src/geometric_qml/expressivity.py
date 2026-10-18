"""
Expressivity under a fixed parameter budget: variational ground-state searches
for the Heisenberg chain or ring with the three circuit families.

All three start from the same singlet product state and use L-BFGS with
adjoint gradients (``statevector.energy_and_gradient``) from several random
starting points. The number of layers of each family is the largest that keeps
the parameter count at or below the budget.

What to expect
--------------
* Exchange circuits stay in the S = 0 sector, which contains the ground state of
  both lattices, so they can reach it.
* Hardware-efficient circuits can in principle reach any state, but with about a
  hundred parameters for a 256-dimensional space they do not get close.
* An S_N-equivariant circuit acts as ⊕_S U_S ⊗ 1 on the Schur–Weyl blocks while an
  SU(2)-invariant Hamiltonian acts as ⊕_S 1 ⊗ h_S. The energy
  Σ_S Tr[U_S ρ_S U_S†] ⟨h_S⟩ therefore never changes: the search cannot move at
  all, for the chain, the ring, or any input state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import scipy.optimize

from .models import HeisenbergSpinChain, create_singlet_product_state_vector
from .statevector import (energy_and_gradient, hardware_efficient_circuit, num_params,
                          permutation_equivariant_circuit, run_circuit, su2_equivariant_circuit, expectation)


def layers_for_budget(kind: str, n_qubits: int, budget: int, n_bonds: int) -> int:
    if kind == "hea":
        return max(budget // (2 * n_qubits) - 1, 0)
    if kind == "su2":
        return budget // n_bonds
    if kind == "sn":
        return budget // 3
    raise ValueError(kind)


def vqe(gates, n_qubits: int, psi0: np.ndarray, H, n_restarts: int = 4, seed: int = 0,
        maxiter: int = 2000, init_scale: float = np.pi) -> Dict[str, Any]:
    """Best of ``n_restarts`` L-BFGS runs from angles uniform in [−init_scale, init_scale]."""
    rng = np.random.default_rng(seed)
    P = num_params(gates)
    runs = []
    for _ in range(n_restarts):
        x0 = rng.uniform(-init_scale, init_scale, P)
        res = scipy.optimize.minimize(lambda x: energy_and_gradient(gates, n_qubits, x, psi0, H), x0, jac=True,
                                      method="L-BFGS-B", options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-14})
        runs.append((float(res.fun), res.x, int(res.nit)))
    runs.sort(key=lambda r: r[0])
    return {"energies": [r[0] for r in runs], "best_params": runs[0][1], "iterations": [r[2] for r in runs]}


def expressivity_comparison(n_qubits: int = 8, lattice: str = "1d_chain", budget: int = 112, n_restarts: int = 4,
                            seed: int = 0, maxiter: int = 2000) -> Dict[str, Any]:
    """
    Ground-state energy error and fidelity for each family at the same parameter budget.
    Returns {kind: {n_params, depth, energy, error, fidelity, energies}} plus the exact energy.
    """
    model = HeisenbergSpinChain(n_qubits, lattice_type=lattice)
    H = model.get_sparse_matrix()
    e0, gs = model.exact_ground_state()
    psi0 = create_singlet_product_state_vector(n_qubits)
    out: Dict[str, Any] = {"exact_energy": e0, "n_qubits": n_qubits, "lattice": lattice, "budget": budget,
                           "initial_energy": float(np.real(np.vdot(psi0, H @ psi0)))}
    for kind in ("hea", "su2", "sn"):
        L = layers_for_budget(kind, n_qubits, budget, len(model.couplings))
        gates = {"hea": hardware_efficient_circuit(n_qubits, L),
                 "su2": su2_equivariant_circuit(n_qubits, L, model.couplings),
                 "sn": permutation_equivariant_circuit(n_qubits, L)}[kind]
        r = vqe(gates, n_qubits, psi0, H, n_restarts=n_restarts, seed=seed, maxiter=maxiter)
        psi = run_circuit(gates, n_qubits, r["best_params"][None, :], psi0)[0]
        out[kind] = {"n_params": num_params(gates), "depth": L, "energy": r["energies"][0],
                     "error": r["energies"][0] - e0, "fidelity": float(abs(np.vdot(gs, psi)) ** 2),
                     "energies": r["energies"], "iterations": r["iterations"]}
    return out


def sn_energy_spread(n_qubits: int = 8, lattice: str = "1d_chain", n_layers: int = 10, n_samples: int = 200,
                     seed: int = 0, initial: str = "singlet") -> Dict[str, float]:
    """Spread of ⟨H⟩ over random S_N circuits: zero up to rounding for any SU(2)-invariant H."""
    from .variance_theory import neel_state_vector

    H = HeisenbergSpinChain(n_qubits, lattice_type=lattice).get_sparse_matrix()
    psi0 = create_singlet_product_state_vector(n_qubits) if initial == "singlet" else neel_state_vector(n_qubits)
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, (n_samples, 3 * n_layers))
    e = expectation(run_circuit(permutation_equivariant_circuit(n_qubits, n_layers), n_qubits, th, psi0), H)
    return {"initial_energy": float(np.real(np.vdot(psi0, H @ psi0))), "mean": float(e.mean()),
            "max_deviation": float(np.max(np.abs(e - np.real(np.vdot(psi0, H @ psi0)))))}
