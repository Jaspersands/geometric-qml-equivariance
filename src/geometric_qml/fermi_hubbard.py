"""
Fermi-Hubbard Model with SU(2)_spin x SU(2)_charge Symmetries.

Implements Jordan-Wigner mapped fermionic hopping and On-site Coulomb interaction:
    H = -t * sum_{<i,j>, sigma} (c_{i sigma}^dagger c_{j sigma} + h.c.) + U * sum_i n_{i up} n_{i down}
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    qml = None


class FermiHubbardChain:
    """
    Fermi-Hubbard model on L sites (2L spin-orbitals / qubits).
    Qubit ordering: site 0 up (0), site 0 down (1), site 1 up (2), site 1 down (3), ...
    """

    def __init__(self, n_sites: int = 2, hopping_t: float = 1.0, on_site_u: float = 4.0):
        self.n_sites = n_sites
        self.n_qubits = 2 * n_sites
        self.t = hopping_t
        self.u = on_site_u

    def get_pennylane_hamiltonian(self) -> "qml.Hamiltonian":
        """Builds Jordan-Wigner mapped Fermi-Hubbard Hamiltonian in PennyLane."""
        if not HAS_PENNYLANE:
            raise RuntimeError("PennyLane required.")

        coeffs = []
        ops = []

        # 1. Hopping terms: -t/2 (X_i Z ... Z X_j + Y_i Z ... Z Y_j)
        for i in range(self.n_sites - 1):
            # Up spin hopping: qubit 2*i <-> 2*(i+1)
            q_i_up = 2 * i
            q_j_up = 2 * (i + 1)
            coeffs.extend([-0.5 * self.t, -0.5 * self.t])
            ops.append(qml.PauliX(q_i_up) @ qml.PauliZ(q_i_up + 1) @ qml.PauliX(q_j_up))
            ops.append(qml.PauliY(q_i_up) @ qml.PauliZ(q_i_up + 1) @ qml.PauliY(q_j_up))

            # Down spin hopping: qubit 2*i+1 <-> 2*(i+1)+1
            q_i_dn = 2 * i + 1
            q_j_dn = 2 * (i + 1) + 1
            coeffs.extend([-0.5 * self.t, -0.5 * self.t])
            ops.append(qml.PauliX(q_i_dn) @ qml.PauliZ(q_i_dn + 1) @ qml.PauliX(q_j_dn))
            ops.append(qml.PauliY(q_i_dn) @ qml.PauliZ(q_i_dn + 1) @ qml.PauliY(q_j_dn))

        # 2. On-site interaction: U * n_up * n_down = U/4 (I - Z_up - Z_down + Z_up Z_down)
        for i in range(self.n_sites):
            q_up = 2 * i
            q_dn = 2 * i + 1
            coeffs.append(0.25 * self.u)
            ops.append(qml.PauliZ(q_up) @ qml.PauliZ(q_dn))

            coeffs.extend([-0.25 * self.u, -0.25 * self.u])
            ops.append(qml.PauliZ(q_up))
            ops.append(qml.PauliZ(q_dn))

        return qml.Hamiltonian(coeffs, ops)
