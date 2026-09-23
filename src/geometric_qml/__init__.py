"""
geometric_qml
=============
SU(2)-equivariant quantum neural networks versus barren plateaus: lattice
Heisenberg models, exchange-generated equivariant ansätze, Dynamical Lie
Algebra closure (with a verified closed form), gradient-variance scaling,
equivariant VQE, Fermi–Hubbard spin/charge symmetries, permutation-
equivariant QGNNs and Google Sycamore transpilation.
"""

from .models import (
    HeisenbergSpinChain,
    lattice_edges,
    LATTICES,
    pauli_string_matrix,
    get_su2_generators,
    get_casimir_operator,
    compute_singlet_dimension,
    total_spin_sectors,
    create_singlet_product_state,
    create_singlet_product_state_vector,
)
from .ansatz import EquivariantQuantumAnsatz, HardwareEfficientAnsatz, QAOABaselineAnsatz
from .barren_plateau import GradientVarianceAnalyzer, run_barren_plateau_scaling_study, fit_variance_scaling
from .sycamore_transpile import SycamoreEquivariantTranspiler, decompose_heisenberg_interaction_to_fsim
from .vqe import EquivariantVQE
from .dla import DynamicalLieAlgebraAnalyzer, analyze_ansatz, equivariant_dla_dimension_formula
from .fermi_hubbard import FermiHubbardChain, jordan_wigner_annihilation
from .qgnn import QuantumGraphNeuralNetwork, permutation_matrix
from .export_openfermion import export_heisenberg_to_dict, hamiltonian_to_string, save_hamiltonian_json

__version__ = "0.4.0"
__all__ = [
    "HeisenbergSpinChain", "lattice_edges", "LATTICES", "pauli_string_matrix",
    "get_su2_generators", "get_casimir_operator", "compute_singlet_dimension", "total_spin_sectors",
    "create_singlet_product_state", "create_singlet_product_state_vector",
    "EquivariantQuantumAnsatz", "HardwareEfficientAnsatz", "QAOABaselineAnsatz",
    "GradientVarianceAnalyzer", "run_barren_plateau_scaling_study", "fit_variance_scaling",
    "SycamoreEquivariantTranspiler", "decompose_heisenberg_interaction_to_fsim",
    "EquivariantVQE",
    "DynamicalLieAlgebraAnalyzer", "analyze_ansatz", "equivariant_dla_dimension_formula",
    "FermiHubbardChain", "jordan_wigner_annihilation",
    "QuantumGraphNeuralNetwork", "permutation_matrix",
    "export_heisenberg_to_dict", "hamiltonian_to_string", "save_hamiltonian_json",
]
