"""
Tests for ansatz generators / NumPy unitaries and the Dynamical Lie Algebra
(DLA) closure analyzer.
"""

import numpy as np
import pytest

from geometric_qml.ansatz import EquivariantQuantumAnsatz, HardwareEfficientAnsatz, QAOABaselineAnsatz
from geometric_qml.dla import DynamicalLieAlgebraAnalyzer, analyze_ansatz, orthonormalize_basis, equivariant_dla_dimension_formula
from geometric_qml.models import get_su2_generators, get_casimir_operator, create_singlet_product_state_vector


def test_orthonormalize_drops_dependent_elements():
    a = np.diag([1.0, -1.0]).astype(complex)
    b = 2.0 * a
    c = np.array([[0, 1], [1, 0]], dtype=complex)
    basis = orthonormalize_basis([a, b, c, a + c])
    assert len(basis) == 2


def test_eqnn_two_qubit_dla_is_one_dimensional():
    # A single exchange generator XX+YY+ZZ spans a 1-D (abelian) Lie algebra.
    ansatz = EquivariantQuantumAnsatz(n_qubits=2, n_layers=2)
    res = analyze_ansatz(ansatz)
    assert res["dla_dimension"] == 1
    assert res["closed"] is True
    assert res["full_su_dimension"] == 15


def test_hea_dla_is_full_special_unitary_algebra():
    for n in (2, 3):
        res = analyze_ansatz(HardwareEfficientAnsatz(n_qubits=n, n_layers=1))
        assert res["dla_dimension"] == 4**n - 1
        assert res["is_sub_exponential"] is False


def test_eqnn_four_qubit_chain_dla_closes_and_is_small():
    # Generators: three bond exchanges on a 4-qubit chain. Their Lie closure
    # acts within total-spin sectors; the dimension must be far below 255.
    ansatz = EquivariantQuantumAnsatz(n_qubits=4, n_layers=1, weight_sharing=False)
    res = analyze_ansatz(ansatz)
    assert res["closed"] is True
    assert res["dla_dimension"] < 40
    assert res["is_sub_exponential"] is True
    # Every element of the closure commutes with the total spin.
    Sx, Sy, Sz = get_su2_generators(4)
    for g in res["basis"]:
        for S in (Sx, Sy, Sz):
            assert np.linalg.norm(g @ S - S @ g) < 1e-8
    # Exact value: ⊕_S su(m_S) ⊕ centre → Σ(m_S²−1)+1 = 3 + 8 + 0 + 1 = 12.
    assert res["dla_dimension"] == 12 == equivariant_dla_dimension_formula(4)


def test_eqnn_six_qubit_dla_matches_closed_form():
    ansatz = EquivariantQuantumAnsatz(n_qubits=6, n_layers=1, weight_sharing=False)
    res = analyze_ansatz(ansatz)
    assert res["closed"] is True
    assert res["dla_dimension"] == 129 == equivariant_dla_dimension_formula(6)
    assert res["dla_dimension"] < 4**6 - 1


def test_qaoa_generators_and_dimension():
    ansatz = QAOABaselineAnsatz(n_qubits=3, n_layers=1)
    gens = ansatz.generators()
    assert len(gens) == 2
    res = analyze_ansatz(ansatz, max_iterations=30)
    assert res["closed"] is True
    assert 2 <= res["dla_dimension"] <= 63


def test_eqnn_numpy_unitary_matches_pennylane_state_and_preserves_singlet():
    pytest.importorskip("pennylane")
    import pennylane as qml

    n = 4
    ansatz = EquivariantQuantumAnsatz(n_qubits=n, n_layers=3, weight_sharing=False)
    rng = np.random.default_rng(0)
    params = rng.uniform(0, 2 * np.pi, ansatz.num_params())

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ(p):
        ansatz.apply_circuit(p)
        return qml.state()

    psi_pl = np.asarray(circ(params))
    psi_np = ansatz.unitary(params) @ create_singlet_product_state_vector(n)
    assert np.isclose(abs(np.vdot(psi_pl, psi_np)), 1.0, atol=1e-8)
    assert abs(np.vdot(psi_np, get_casimir_operator(n) @ psi_np)) < 1e-10


def test_hea_numpy_unitary_is_unitary_and_breaks_su2():
    n = 3
    ansatz = HardwareEfficientAnsatz(n_qubits=n, n_layers=2)
    rng = np.random.default_rng(1)
    U = ansatz.unitary(rng.uniform(0, 2 * np.pi, ansatz.num_params()))
    assert np.allclose(U.conj().T @ U, np.eye(8), atol=1e-10)
    Sx, _, _ = get_su2_generators(n)
    assert np.linalg.norm(U @ Sx - Sx @ U) > 0.1


def test_dla_analyzer_direct_generators():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.diag([1.0, -1.0]).astype(complex)
    res = DynamicalLieAlgebraAnalyzer([sx, sz]).compute_dla_dimension()
    assert res["dla_dimension"] == 3  # su(2)
    assert res["iteration_dimensions"][0] == 2
