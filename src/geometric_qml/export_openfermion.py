"""
OpenFermion and Qiskit Nature Operator Exporters.

Exports geometric quantum Hamiltonians and symmetry projectors into standard formats.
"""

from __future__ import annotations
import json
from typing import Dict, Any, Optional
from .models import HeisenbergSpinChain


def export_heisenberg_to_dict(model: HeisenbergSpinChain) -> Dict[str, Any]:
    """Serializes the Heisenberg spin Hamiltonian into operator dictionary."""
    terms = []
    for i, j in model.couplings:
        terms.append({"qubits": [i, j], "type": "XX", "coefficient": float(model.j)})
        terms.append({"qubits": [i, j], "type": "YY", "coefficient": float(model.j)})
        terms.append({"qubits": [i, j], "type": "ZZ", "coefficient": float(model.j * model.delta)})

    return {
        "n_qubits": model.n_qubits,
        "lattice_type": model.lattice_type,
        "anisotropy_delta": model.delta,
        "num_terms": len(terms),
        "pauli_terms": terms,
    }


def save_hamiltonian_json(model: HeisenbergSpinChain, file_path: str):
    """Saves serialized Hamiltonian to JSON file."""
    data = export_heisenberg_to_dict(model)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
