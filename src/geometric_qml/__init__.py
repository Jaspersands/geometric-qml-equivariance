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

__version__ = "0.1.0"
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
]
