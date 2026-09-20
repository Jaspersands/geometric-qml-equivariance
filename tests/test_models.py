"""
Tests for spin models: lattice geometries, PennyLane-free Hamiltonians,
SU(2) generators, total-spin sectors, and the singlet product state.
"""

import numpy as np
import pytest

from geometric_qml.models import (
    HeisenbergSpinChain,
    lattice_edges,
    get_su2_generators,
    get_casimir_operator,
    compute_singlet_dimension,
    create_singlet_product_state_vector,
    total_spin_sectors,
)


def test_singlet_dimension_catalan():
    assert [compute_singlet_dimension(n) for n in (2, 4, 6, 8, 10)] == [1, 2, 5, 14, 42]
    assert compute_singlet_dimension(3) == 0


@pytest.mark.parametrize(
    "lattice,n,periodic,expected",
    [
        ("1d_chain", 6, False, 5),
        ("ring", 6, False, 6),
        ("1d_chain", 6, True, 6),
        ("2d_grid", 4, False, 4),      # 2×2 square
        ("2d_grid", 6, False, 7),      # 2×3: 4 horizontal + 3 vertical
        ("triangular", 6, False, 9),   # ladder: 5 chain + 4 diagonals
        ("kagome", 6, False, 7),       # two corner-sharing triangles: 3+3 edges + 1 link
    ],
)
def test_lattice_edge_counts(lattice, n, periodic, expected):
    edges = lattice_edges(n, lattice, periodic=periodic)
    assert len(edges) == expected
    assert len(set(tuple(sorted(e)) for e in edges)) == len(edges)  # no duplicates
    assert all(0 <= i < n and 0 <= j < n and i != j for i, j in edges)


def test_lattice_2d_grid_shape_argument():
    edges = lattice_edges(6, "2d_grid", shape=(3, 2))
    assert len(edges) == 7
    with pytest.raises(ValueError):
        lattice_edges(6, "2d_grid", shape=(4, 2))
    with pytest.raises(ValueError):
        lattice_edges(6, "hexagonal")


def test_hamiltonian_matrix_without_pennylane_matches_pennylane():
    model = HeisenbergSpinChain(n_qubits=4, j_coupling=0.7, anisotropy_delta=1.3, lattice_type="ring")
    m_np = model.get_matrix(backend="numpy")
    m_pl = model.get_matrix(backend="pennylane")
    assert np.allclose(m_np, m_pl, atol=1e-10)
    assert np.allclose(m_np, m_np.conj().T)


@pytest.mark.parametrize("lattice", ["1d_chain", "ring", "2d_grid", "triangular", "kagome"])
def test_isotropic_model_commutes_with_total_spin(lattice):
    model = HeisenbergSpinChain(n_qubits=6, anisotropy_delta=1.0, lattice_type=lattice)
    H = model.get_matrix(backend="numpy")
    for S in get_su2_generators(6):
        assert np.linalg.norm(H @ S - S @ H) < 1e-9


def test_anisotropic_model_breaks_su2():
    model = HeisenbergSpinChain(n_qubits=4, anisotropy_delta=2.5)
    H = model.get_matrix(backend="numpy")
    Sx, _, Sz = get_su2_generators(4)
    assert np.linalg.norm(H @ Sx - Sx @ H) > 0.5
    assert np.linalg.norm(H @ Sz - Sz @ H) < 1e-9  # U(1) survives


def test_total_spin_sectors_four_qubits():
    # 4 spin-½: 2 singlets (S=0), 3 triplets (S=1), 1 quintet (S=2): 2·1 + 3·3 + 1·5 = 16
    sectors = total_spin_sectors(4)
    assert sectors == {0.0: 2, 1.0: 3, 2.0: 1}
    assert sum(mult * int(2 * s + 1) for s, mult in sectors.items()) == 16
    assert sectors[0.0] == compute_singlet_dimension(4)


def test_singlet_product_state_vector():
    psi = create_singlet_product_state_vector(4)
    assert np.isclose(np.linalg.norm(psi), 1.0)
    S2 = get_casimir_operator(4)
    assert abs(np.vdot(psi, S2 @ psi)) < 1e-12


def test_exact_ground_state_energy_two_site():
    model = HeisenbergSpinChain(n_qubits=2)
    e0, v0 = model.exact_ground_state()
    assert np.isclose(e0, -3.0)
    singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
    assert np.isclose(abs(np.vdot(singlet, v0)), 1.0)
