"""
geometric_qml
=============
Symmetric variational circuits and their Lie algebras: exchange-generated
SU(2)-equivariant and S_N-equivariant ansätze, dynamical Lie algebra closure
with closed forms, the deep-circuit loss variance predicted from the algebra
and checked against sampling, expressivity under a fixed parameter budget,
Lie-algebraic simulation, lattice Heisenberg and Fermi–Hubbard models, graph
networks and Sycamore transpilation.
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
from .ansatz import EquivariantQuantumAnsatz, HardwareEfficientAnsatz, QAOABaselineAnsatz, PermutationEquivariantAnsatz
from .barren_plateau import GradientVarianceAnalyzer, run_barren_plateau_scaling_study, fit_variance_scaling
from .sycamore_transpile import SycamoreEquivariantTranspiler, decompose_heisenberg_interaction_to_fsim
from .vqe import EquivariantVQE
from .dla import (
    DynamicalLieAlgebraAnalyzer,
    analyze_ansatz,
    equivariant_dla_dimension_formula,
    permutation_dla_dimension_formula,
    permutation_dla_dimension_by_closure,
)
from .variance_theory import (
    SchurBasis,
    schur_purities,
    reductive_decomposition,
    g_purities,
    predict_loss,
    compare_prediction,
    variance_study,
    depth_scan,
    neel_state_vector,
    singlet_input_prediction,
)
from .gsim import LieAlgebraicSimulator, permutation_equivariant_simulator
from .fermi_hubbard import FermiHubbardChain, jordan_wigner_annihilation
from .qgnn import QuantumGraphNeuralNetwork, permutation_matrix
from .export_openfermion import export_heisenberg_to_dict, hamiltonian_to_string, save_hamiltonian_json

__version__ = "0.5.0"
__all__ = [
    "HeisenbergSpinChain", "lattice_edges", "LATTICES", "pauli_string_matrix",
    "get_su2_generators", "get_casimir_operator", "compute_singlet_dimension", "total_spin_sectors",
    "create_singlet_product_state", "create_singlet_product_state_vector",
    "EquivariantQuantumAnsatz", "HardwareEfficientAnsatz", "QAOABaselineAnsatz", "PermutationEquivariantAnsatz",
    "GradientVarianceAnalyzer", "run_barren_plateau_scaling_study", "fit_variance_scaling",
    "SycamoreEquivariantTranspiler", "decompose_heisenberg_interaction_to_fsim",
    "EquivariantVQE",
    "DynamicalLieAlgebraAnalyzer", "analyze_ansatz", "equivariant_dla_dimension_formula",
    "permutation_dla_dimension_formula", "permutation_dla_dimension_by_closure",
    "SchurBasis", "schur_purities", "reductive_decomposition", "g_purities", "predict_loss",
    "compare_prediction", "variance_study", "depth_scan", "neel_state_vector", "singlet_input_prediction",
    "LieAlgebraicSimulator", "permutation_equivariant_simulator",
    "FermiHubbardChain", "jordan_wigner_annihilation",
    "QuantumGraphNeuralNetwork", "permutation_matrix",
    "export_heisenberg_to_dict", "hamiltonian_to_string", "save_hamiltonian_json",
]
