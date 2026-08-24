# Geometric QML: SU(2) Equivariant Quantum Circuits vs. Barren Plateaus

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/Differentiable-PennyLane-purple.svg)](https://pennylane.ai/)
[![Cirq Sycamore](https://img.shields.io/badge/Hardware-Google%20Sycamore%20FSim-teal.svg)](https://quantumai.google/cirq)

> **Equivariant Quantum Neural Networks (EQNNs) preserving global $SU(2)$ spin-rotation and spatial permutation symmetries for Heisenberg spin lattices and Fermi-Hubbard chains in PennyLane, demonstrating Dynamical Lie Algebra (DLA) dimension scaling, barren plateau evasion, and native Google Sycamore $\text{PhasedFSim}$ hardware transpilation.**

---

## 🌌 Overview & Features (v0.2.0)

- **$SU(2)$ Equivariant Quantum Neural Networks**:
  - Exact commutation with total spin angular momentum $[U(\vec{\theta}), S^\alpha] = 0$.
  - Dynamics strictly confined to the singlet manifold ($\dim = C_{N/2} \ll 2^N$).
- **Dynamical Lie Algebra (DLA) Dimension Analyzer (`DynamicalLieAlgebraAnalyzer`)**:
  - Iterative commutator basis closure $\mathfrak{g} = \text{Lie}(\{i G_k\})$.
  - Rigorous proof of sub-exponential dimension $\dim(\mathfrak{g}_{\text{EQNN}}) \sim \mathcal{O}(\text{poly}(N))$ vs $\dim(\mathfrak{g}_{\text{HEA}}) = 4^N - 1$.
- **Extended Quantum Physics Models**:
  - **Fermi-Hubbard Chain (`FermiHubbardChain`)**: $SU(2)_{\text{spin}} \times SU(2)_{\text{charge}}$ pseudospin symmetry.
  - **Quantum Graph Neural Networks (`QuantumGraphNeuralNetwork`)**: Permutation-equivariant graph convolutional layers on arbitrary molecular adjacency matrices $A_{ij}$.
  - **OpenFermion / Qiskit Nature Exporter (`export_heisenberg_to_dict`)**.
- **Google Sycamore Hardware Compilation**:
  - Native `cirq.PhasedFSimGate` + 1Q rotations mapped to planar grid topologies.

---

## 🚀 Quickstart

```python
from geometric_qml import (
    GradientVarianceAnalyzer,
    DynamicalLieAlgebraAnalyzer,
    FermiHubbardChain,
    QuantumGraphNeuralNetwork,
    SycamoreEquivariantTranspiler,
    EquivariantVQE,
)

# 1. Analyze Dynamical Lie Algebra Dimension
model = FermiHubbardChain(n_sites=2)
print(f"Fermi-Hubbard Qubits: {model.n_qubits}")

# 2. Transpile onto Google Sycamore Grid
transpiler = SycamoreEquivariantTranspiler(n_qubits=4)
circuit = transpiler.transpile_circuit(n_layers=3, params=[0.2, 0.4, 0.6])
print(circuit)

# 3. Equivariant VQE Ground State Finding
vqe = EquivariantVQE(n_qubits=4, n_layers=3)
res = vqe.train(steps=40)
print(f"Ground Energy: {res['final_energy']:.4f} (Exact: {res['exact_ground_energy']:.4f})")
```

---

## 🧪 Testing & Benchmarks

```bash
pytest -v tests/
python benchmarks/run_barren_plateau_benchmark.py
```

---

## 📄 Citation & License

Developed by **Jasper Sands** under the **Apache-2.0 License**.
