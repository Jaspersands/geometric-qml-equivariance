"""
Tests for Project 2 Extensions: DLA analyzer, Fermi-Hubbard, QGNN, and OpenFermion exporter.
"""

import pytest
import numpy as np
from geometric_qml.dla import DynamicalLieAlgebraAnalyzer
from geometric_qml.fermi_hubbard import FermiHubbardChain
from geometric_qml.qgnn import QuantumGraphNeuralNetwork
from geometric_qml.models import HeisenbergSpinChain
from geometric_qml.export_openfermion import export_heisenberg_to_dict, save_hamiltonian_json


def test_dla_dimension_analysis():
    # 2-qubit exchange generators: {XX, YY, ZZ}
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    xx = np.kron(sx, sx)
    yy = np.kron(sy, sy)
    zz = np.kron(sz, sz)

    analyzer = DynamicalLieAlgebraAnalyzer(generators=[xx, yy, zz], max_iterations=4)
    res = analyzer.compute_dla_dimension()

    assert "dla_dimension" in res
    assert res["dla_dimension"] < res["full_su_dimension"]
    assert res["is_sub_exponential"] is True


def test_fermi_hubbard_hamiltonian():
    fh = FermiHubbardChain(n_sites=2, hopping_t=1.0, on_site_u=2.0)
    assert fh.n_qubits == 4
    h_qml = fh.get_pennylane_hamiltonian()
    assert len(h_qml.ops) > 0


def test_qgnn_parameter_count_and_execution():
    # 4-node ring graph adjacency matrix
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ])
    qgnn = QuantumGraphNeuralNetwork(adjacency_matrix=adj, n_layers=2)
    assert qgnn.num_params() == 4


def test_openfermion_export(tmp_path):
    model = HeisenbergSpinChain(n_qubits=4, j_coupling=1.0, anisotropy_delta=1.0)
    data = export_heisenberg_to_dict(model)
    assert data["n_qubits"] == 4
    assert data["num_terms"] == 9 # 3 bonds * 3 terms

    file_path = str(tmp_path / "model.json")
    save_hamiltonian_json(model, file_path)
