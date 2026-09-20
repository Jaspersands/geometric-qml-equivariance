"""
Exporters for spin Hamiltonians: a JSON term list, an OpenFermion
``QubitOperator``-style string and a Qiskit ``SparsePauliOp``-style list.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .models import HeisenbergSpinChain


def export_heisenberg_to_dict(model: HeisenbergSpinChain) -> Dict[str, Any]:
    terms: List[Dict[str, Any]] = []
    for coeff, spec in model.pauli_terms():
        qubits = sorted(spec)
        terms.append({"qubits": qubits, "type": "".join(spec[q] for q in qubits), "coefficient": float(coeff)})
    return {
        "n_qubits": model.n_qubits,
        "lattice_type": model.lattice_type,
        "anisotropy_delta": model.delta,
        "j_coupling": model.j,
        "edges": [list(e) for e in model.couplings],
        "num_terms": len(terms),
        "pauli_terms": terms,
    }


def hamiltonian_to_string(model: HeisenbergSpinChain, fmt: str = "openfermion") -> str:
    """
    ``fmt='openfermion'`` → ``1.0 [X0 X1] +\\n1.0 [Y0 Y1] + …`` (parseable by ``QubitOperator``)
    ``fmt='qiskit'``      → one ``LABEL coeff`` per line with Qiskit's little-endian labels
    """
    if fmt == "openfermion":
        parts = []
        for coeff, spec in model.pauli_terms():
            body = " ".join(f"{spec[q]}{q}" for q in sorted(spec))
            parts.append(f"{coeff:.10g} [{body}]")
        return " +\n".join(parts)
    if fmt == "qiskit":
        lines = []
        for coeff, spec in model.pauli_terms():
            label = "".join(spec.get(q, "I") for q in range(model.n_qubits))[::-1]  # qubit 0 rightmost
            lines.append(f"{label} {coeff:.10g}")
        return "\n".join(lines)
    raise ValueError("fmt must be 'openfermion' or 'qiskit'")


def save_hamiltonian_json(model: HeisenbergSpinChain, file_path: str) -> None:
    with open(file_path, "w") as f:
        json.dump(export_heisenberg_to_dict(model), f, indent=2)
