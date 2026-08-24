"""
geometric_qml
=============
SU(2) and Lattice-Equivariant Quantum Neural Networks
vs. Barren Plateaus with Google Sycamore Transpilation.
"""

from .models import (
    HeisenbergSpinChain,
    get_su2_generators,
    get_casimir_operator,
    compute_singlet_dimension,
    create_singlet_product_state,
)
from .ansatz import (
    EquivariantQuantumAnsatz,
    HardwareEfficientAnsatz,
    QAOABaselineAnsatz,
)
from .barren_plateau import (
    GradientVarianceAnalyzer,
    run_barren_plateau_scaling_study,
)
from .sycamore_transpile import (
    SycamoreEquivariantTranspiler,
    decompose_heisenberg_interaction_to_fsim,
)
from .vqe import EquivariantVQE
from .dla import DynamicalLieAlgebraAnalyzer
from .fermi_hubbard import FermiHubbardChain
from .qgnn import QuantumGraphNeuralNetwork
from .export_openfermion import export_heisenberg_to_dict, save_hamiltonian_json

__version__ = "0.2.0"
__all__ = [
    "HeisenbergSpinChain",
    "get_su2_generators",
    "get_casimir_operator",
    "compute_singlet_dimension",
    "create_singlet_product_state",
    "EquivariantQuantumAnsatz",
    "HardwareEfficientAnsatz",
    "QAOABaselineAnsatz",
    "GradientVarianceAnalyzer",
    "run_barren_plateau_scaling_study",
    "SycamoreEquivariantTranspiler",
    "decompose_heisenberg_interaction_to_fsim",
    "EquivariantVQE",
    "DynamicalLieAlgebraAnalyzer",
    "FermiHubbardChain",
    "QuantumGraphNeuralNetwork",
    "export_heisenberg_to_dict",
    "save_hamiltonian_json",
]
