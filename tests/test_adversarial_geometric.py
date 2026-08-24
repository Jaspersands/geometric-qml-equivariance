"""
Adversarial and Stress Test Suite for geometric_qml.
"""

import pytest
import numpy as np
import pennylane as qml
from geometric_qml.models import (
    HeisenbergSpinChain,
    get_su2_generators,
    get_casimir_operator,
    compute_singlet_dimension,
)
from geometric_qml.ansatz import EquivariantQuantumAnsatz, HardwareEfficientAnsatz
from geometric_qml.barren_plateau import GradientVarianceAnalyzer


def test_adversarial_symmetry_breaking_detection():
    """Verify that anisotropy Delta != 1.0 breaks SU(2) symmetry while Delta == 1.0 preserves it."""
    # Isotropic Delta = 1.0 -> Commutator = 0
    h_iso = HeisenbergSpinChain(n_qubits=4, anisotropy_delta=1.0)
    mat_iso = h_iso.get_matrix()
    Sx, Sy, Sz = get_su2_generators(4)
    comm_x = mat_iso @ Sx - Sx @ mat_iso
    assert np.isclose(np.linalg.norm(comm_x), 0.0, atol=1e-7)

    # Anisotropic Delta = 2.5 -> Commutator [H, Sx] != 0
    h_aniso = HeisenbergSpinChain(n_qubits=4, anisotropy_delta=2.5)
    mat_aniso = h_aniso.get_matrix()
    comm_x_aniso = mat_aniso @ Sx - Sx @ mat_aniso
    assert np.linalg.norm(comm_x_aniso) > 0.5, "Anisotropic XXZ must break SU(2) symmetry"


def test_adversarial_singlet_confinement():
    """Verify that EQNN parameterized evolution strictly confines the quantum state to the S_tot = 0 singlet subspace."""
    n_q = 4
    dev = qml.device("default.qubit", wires=n_q)
    ansatz = EquivariantQuantumAnsatz(n_qubits=n_q, n_layers=4, weight_sharing=False)
    
    @qml.qnode(dev)
    def circuit(params):
        ansatz.apply_circuit(params)
        return qml.state()

    # Draw 5 random parameter vectors
    rng = np.random.default_rng(42)
    casimir_matrix = get_casimir_operator(n_q)

    for _ in range(5):
        params = rng.uniform(0.0, 2 * np.pi, size=ansatz.num_params())
        state = circuit(params)
        
        # Expectation of total spin Casimir: <psi| S^2 |psi> must be identically 0 for singlet states
        s2_exp = np.real(state.conj().T @ casimir_matrix @ state)
        assert np.isclose(s2_exp, 0.0, atol=1e-6), f"State leaked out of singlet manifold: <S^2> = {s2_exp}"


def test_adversarial_odd_qubits_validation():
    """Verifies that odd number of qubits raises ValueError."""
    with pytest.raises(ValueError):
        HeisenbergSpinChain(n_qubits=5)


def test_adversarial_deep_circuit_scaling():
    """Stress test deep circuit depth (L=8) with 6 qubits."""
    analyzer = GradientVarianceAnalyzer(n_qubits=6, ansatz_type="equivariant", n_layers=8)
    res = analyzer.compute_gradient_variance(n_samples=5, seed=123)
    assert not np.isnan(res["var_param_0"])
    assert res["var_param_0"] > 0.0
