"""
Unit tests for Geometric QML and SU(2) Equivariance.
"""

import pytest
import numpy as np
from geometric_qml.models import (
    HeisenbergSpinChain,
    get_su2_generators,
    get_casimir_operator,
    compute_singlet_dimension,
)
from geometric_qml.ansatz import (
    EquivariantQuantumAnsatz,
    HardwareEfficientAnsatz,
)
from geometric_qml.barren_plateau import GradientVarianceAnalyzer
from geometric_qml.sycamore_transpile import (
    SycamoreEquivariantTranspiler,
    decompose_heisenberg_interaction_to_fsim,
)
from geometric_qml.vqe import EquivariantVQE


def test_singlet_dimension_catalan():
    # Catalan numbers: C_1=1, C_2=2, C_3=5, C_4=14, C_5=42
    assert compute_singlet_dimension(2) == 1
    assert compute_singlet_dimension(4) == 2
    assert compute_singlet_dimension(6) == 5
    assert compute_singlet_dimension(8) == 14
    assert compute_singlet_dimension(10) == 42


def test_heisenberg_model_hermiticity():
    model = HeisenbergSpinChain(n_qubits=4, j_coupling=1.0, anisotropy_delta=1.0)
    mat = model.get_matrix()
    assert np.allclose(mat, mat.conj().T), "Hamiltonian matrix must be Hermitian"
    assert mat.shape == (16, 16)


def test_su2_symmetry_commutation():
    # Verify [H, S^alpha] = 0 for XXX isotropic model
    model = HeisenbergSpinChain(n_qubits=4, j_coupling=1.0, anisotropy_delta=1.0)
    H_mat = model.get_matrix()
    Sx, Sy, Sz = get_su2_generators(4)

    for S_alpha, name in [(Sx, "Sx"), (Sy, "Sy"), (Sz, "Sz")]:
        comm = H_mat @ S_alpha - S_alpha @ H_mat
        norm = np.linalg.norm(comm)
        assert np.isclose(norm, 0.0, atol=1e-7), f"Commutator [H, {name}] must vanish for isotropic XXX model"


def test_casimir_eigenvalue():
    # S^2 on N qubits has non-negative real eigenvalues
    S2 = get_casimir_operator(4)
    evals = np.linalg.eigvalsh(S2)
    assert np.all(evals >= -1e-7), "Casimir eigenvalues must be non-negative"


def test_ansatz_parameter_counts():
    eqnn = EquivariantQuantumAnsatz(n_qubits=4, n_layers=3, weight_sharing=True)
    assert eqnn.num_params() == 3

    hea = HardwareEfficientAnsatz(n_qubits=4, n_layers=2)
    # (2 + 1) * 4 * 2 = 24 parameters
    assert hea.num_params() == 24


def test_gradient_variance_sampling():
    analyzer = GradientVarianceAnalyzer(n_qubits=4, ansatz_type="equivariant", n_layers=2)
    res = analyzer.compute_gradient_variance(n_samples=10, seed=123)
    assert "var_param_0" in res
    assert res["var_param_0"] > 0.0
    assert res["n_qubits"] == 4


def test_sycamore_fsim_decomposition():
    import cirq
    q0, q1 = cirq.LineQubit.range(2)
    ops = decompose_heisenberg_interaction_to_fsim(q0, q1, theta=0.4)
    assert len(ops) == 4
    assert isinstance(ops[0].gate, cirq.FSimGate)


def test_sycamore_transpiler_circuit_build():
    transpiler = SycamoreEquivariantTranspiler(n_qubits=4)
    circuit = transpiler.transpile_circuit(n_layers=2, params=np.array([0.3, 0.5]))
    assert len(circuit) > 0
    assert len(circuit.all_qubits()) == 4


def test_vqe_convergence():
    vqe = EquivariantVQE(n_qubits=2, n_layers=2, learning_rate=0.15)
    res = vqe.train(steps=15)
    # Ground state of 2-qubit XXX Heisenberg: E = -3.0
    assert abs(res["exact_ground_energy"] - (-3.0)) < 1e-4
    assert res["energy_error"] < 0.2
