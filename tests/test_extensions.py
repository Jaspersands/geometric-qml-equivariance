"""
Tests for Fermi-Hubbard symmetries, the QGNN, Sycamore routing, exporters,
and the CLI.
"""

import json

import numpy as np
import pytest

from geometric_qml import __version__
from geometric_qml.fermi_hubbard import FermiHubbardChain
from geometric_qml.qgnn import QuantumGraphNeuralNetwork, permutation_matrix
from geometric_qml.models import HeisenbergSpinChain
from geometric_qml.export_openfermion import export_heisenberg_to_dict, hamiltonian_to_string, save_hamiltonian_json
from geometric_qml.cli import main as cli_main


def test_version():
    assert __version__ == "0.3.0"


# ----------------------------------------------------------------------------- #
# Fermi–Hubbard
# ----------------------------------------------------------------------------- #
def test_fermi_hubbard_matrix_matches_pennylane_and_is_hermitian():
    pytest.importorskip("pennylane")
    import pennylane as qml
    fh = FermiHubbardChain(n_sites=2, hopping_t=1.0, on_site_u=4.0)
    H = fh.get_matrix()
    H_pl = qml.matrix(fh.get_pennylane_hamiltonian(), wire_order=list(range(fh.n_qubits)))
    assert np.allclose(H, H_pl, atol=1e-10)
    assert np.allclose(H, H.conj().T)


def test_fermi_hubbard_two_site_spectrum():
    # Half-filled 2-site Hubbard at t=1, U=4: E0 = (U − sqrt(U² + 16t²))/2
    fh = FermiHubbardChain(n_sites=2, hopping_t=1.0, on_site_u=4.0)
    e0 = fh.ground_energy(n_particles=2)
    expected = 0.5 * (4.0 - np.sqrt(16.0 + 16.0))
    assert np.isclose(e0, expected, atol=1e-9)


def test_fermi_hubbard_spin_and_charge_su2_symmetries():
    fh = FermiHubbardChain(n_sites=2, hopping_t=1.0, on_site_u=3.0, particle_hole_symmetric=True)
    H = fh.get_matrix()
    for S in fh.spin_generators():
        assert np.linalg.norm(H @ S - S @ H) < 1e-9
    for E in fh.charge_generators():
        assert np.linalg.norm(H @ E - E @ H) < 1e-9
    # generators satisfy su(2): [Sx, Sy] = i Sz
    Sx, Sy, Sz = fh.spin_generators()
    assert np.allclose(Sx @ Sy - Sy @ Sx, 1j * Sz, atol=1e-10)
    Ex, Ey, Ez = fh.charge_generators()
    assert np.allclose(Ex @ Ey - Ey @ Ex, 1j * Ez, atol=1e-10)


def test_fermi_hubbard_number_operator_commutes():
    fh = FermiHubbardChain(n_sites=3)
    H = fh.get_matrix()
    N = fh.number_operator()
    assert np.linalg.norm(H @ N - N @ H) < 1e-9


# ----------------------------------------------------------------------------- #
# QGNN
# ----------------------------------------------------------------------------- #
def test_qgnn_is_equivariant_under_graph_automorphism():
    ring = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=float)
    qgnn = QuantumGraphNeuralNetwork(adjacency_matrix=ring, n_layers=2)
    assert qgnn.num_params() == 4
    params = np.array([0.3, -0.7, 1.1, 0.4])
    rotation = [1, 2, 3, 0]              # automorphism of the 4-ring
    assert qgnn.is_permutation_equivariant(params, rotation)
    transposition = [1, 0, 2, 3]         # not an automorphism
    assert not qgnn.is_permutation_equivariant(params, transposition)


def test_permutation_matrix_is_unitary_and_acts_correctly():
    P = permutation_matrix([1, 2, 0], 3)
    assert np.allclose(P @ P.conj().T, np.eye(8))
    e = np.zeros(8); e[0b100] = 1.0        # qubit 0 excited
    out = P @ e
    assert out[0b010] == 1.0               # moved to qubit 1


# ----------------------------------------------------------------------------- #
# Sycamore
# ----------------------------------------------------------------------------- #
def test_sycamore_snake_placement_keeps_chain_bonds_adjacent():
    cirq = pytest.importorskip("cirq")
    from geometric_qml.sycamore_transpile import SycamoreEquivariantTranspiler
    tr = SycamoreEquivariantTranspiler(n_qubits=8, grid_width=4)
    bonds = [(i, i + 1) for i in range(7)]
    assert tr.check_adjacency(bonds)
    assert tr.grid_qubits[3].is_adjacent(tr.grid_qubits[4])   # the snake turn
    circuit = tr.transpile_circuit(n_layers=2, params=np.array([0.3, 0.5]))
    assert circuit.are_all_measurements_terminal()
    assert len(circuit.all_qubits()) == 8
    with pytest.raises(ValueError):
        tr.transpile_circuit(n_layers=1, params=np.array([0.1]), couplings=[(0, 5)])


def test_sycamore_decomposition_is_exact():
    cirq = pytest.importorskip("cirq")
    import scipy.linalg
    from geometric_qml.sycamore_transpile import decompose_heisenberg_interaction_to_fsim
    q0, q1 = cirq.LineQubit.range(2)
    theta = 0.37
    ops = decompose_heisenberg_interaction_to_fsim(q0, q1, theta)
    u = cirq.unitary(cirq.Circuit(ops))
    X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
    G = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)
    target = scipy.linalg.expm(-1j * theta * G)
    fid = abs(np.trace(target.conj().T @ u)) ** 2 / 16
    assert fid > 1 - 1e-10
    ops_p = decompose_heisenberg_interaction_to_fsim(q0, q1, theta, use_phased_fsim=True)
    assert any(isinstance(op.gate, cirq.PhasedFSimGate) for op in ops_p)
    assert abs(np.trace(target.conj().T @ cirq.unitary(cirq.Circuit(ops_p)))) ** 2 / 16 > 1 - 1e-10


# ----------------------------------------------------------------------------- #
# Export
# ----------------------------------------------------------------------------- #
def test_exporter_formats(tmp_path):
    model = HeisenbergSpinChain(n_qubits=4, lattice_type="ring")
    d = export_heisenberg_to_dict(model)
    assert d["num_terms"] == 12
    of = hamiltonian_to_string(model, fmt="openfermion")
    qk = hamiltonian_to_string(model, fmt="qiskit")
    assert of.count("[") == 12 and "X0 X1" in of
    assert qk.count("\n") + 1 == 12 and qk.splitlines()[0].split()[0] == "IIXX"  # little-endian labels
    path = tmp_path / "h.json"
    save_hamiltonian_json(model, str(path))
    assert json.loads(path.read_text())["n_qubits"] == 4
    with pytest.raises(ValueError):
        hamiltonian_to_string(model, fmt="cirq")


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def test_cli_dla_runs_closure(capsys):
    assert cli_main(["dla", "--ansatz", "equivariant", "--qubits", "4", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dla_dimension"] == 12 and out["closed"] is True
    assert cli_main(["dla", "--ansatz", "hea", "--qubits", "2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["dla_dimension"] == 15


def test_cli_sectors_and_variance(capsys):
    assert cli_main(["sectors", "--qubits", "6", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["singlet_dimension"] == 5
    pytest.importorskip("pennylane")
    assert cli_main(["variance", "--qubits", "4", "--ansatz", "equivariant", "--samples", "4", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["var_param_0"] > 0
