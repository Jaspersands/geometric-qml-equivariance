# Geometric QML: SU(2) Equivariant Quantum Circuits vs. Barren Plateaus

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/Differentiable-PennyLane-purple.svg)](https://pennylane.ai/)
[![Cirq Sycamore](https://img.shields.io/badge/Hardware-Google%20Sycamore%20FSim-teal.svg)](https://quantumai.google/cirq)

> **Equivariant Quantum Neural Networks (EQNNs) preserving global $SU(2)$ spin-rotation and spatial permutation symmetries for Heisenberg spin lattices in PennyLane, demonstrating rigorous Barren Plateau evasion alongside native Google Sycamore $\text{PhasedFSim}$ hardware transpilation.**

---

## 🌌 Overview & Theoretical Foundations

Variational Quantum Algorithms (VQAs) utilizing standard Hardware-Efficient Ansatzes (HEAs) suffer from **Barren Plateaus**—an exponential vanishing of cost function gradient variances with respect to the number of qubits $N$:

$$\text{Var}_{\vec{\theta}}\left[ \partial_{\theta_k} \langle H \rangle \right] \sim \mathcal{O}\left(\frac{1}{2^N}\right)$$

### 1. Lie Algebra & $SU(2)$ Equivariance

By constructing quantum circuits equivariant under the global symmetry group $\mathcal{G} = SU(2)$, where every generator commutes with the total angular momentum:

$$\left[ U(\vec{\theta}), S^\alpha \right] = 0, \quad S^\alpha = \frac{1}{2} \sum_{i=1}^N \sigma_i^\alpha, \quad \forall \alpha \in \{x, y, z\}$$

the dynamics are confined to the **singlet irrep subspace** ($\mathcal{S}_{\text{tot}} = 0$).

### 2. Catalan Dimension Scaling & Barren Plateau Evasion

By Schur-Weyl duality, the dimension of the singlet subspace is given by the **Catalan numbers**:

$$\dim(\mathcal{H}_{S=0}) = C_{N/2} = \frac{1}{N/2 + 1} \binom{N}{N/2} \ll 2^N$$

Because the Haar measure is integrated strictly over the commutant subspace, the gradient variance decays **polynomially**:

$$\text{Var}_{\vec{\theta}}\left[ \partial_{\theta_k} \langle H \rangle_{\text{EQNN}} \right] \sim \mathcal{O}\left(\frac{1}{\text{poly}(N)}\right)$$

| Qubits ($N$) | Singlet Dim $C_{N/2}$ | Full Hilbert Space $2^N$ | State Space Compression |
|:---:|:---:|:---:|:---:|
| 4 | 2 | 16 | 87.5% |
| 6 | 5 | 64 | 92.2% |
| 8 | 14 | 256 | 94.5% |
| 10 | 42 | 1,024 | 95.9% |
| 12 | 132 | 4,096 | 96.8% |
| 14 | 429 | 16,384 | 97.4% |
| 16 | 1,430 | 65,536 | **97.8%** |

---

## 🛠️ Google Sycamore Hardware Transpilation

The isotropic exchange interaction $\exp\left(-i \theta (X_i X_j + Y_i Y_j + Z_i Z_j)\right)$ decomposes into Google Sycamore native gates:

1. **Planar XY Interaction**: $\text{FSim}(2\theta, \phi=0) \equiv \exp\left(-i \theta (X_i X_j + Y_i Y_j)\right)$
2. **Longitudinal Z Interaction**: $\exp\left(-i \theta Z_i Z_j\right) = \text{CNOT} \cdot R_z(2\theta) \cdot \text{CNOT}$

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/Jaspersands/geometric-qml-equivariance.git
cd geometric-qml-equivariance
pip install -e .
```

### Python API Example

```python
from geometric_qml import (
    GradientVarianceAnalyzer,
    run_barren_plateau_scaling_study,
    SycamoreEquivariantTranspiler,
    EquivariantVQE,
)

# 1. Compare Gradient Variance Scaling
results = run_barren_plateau_scaling_study(qubit_list=[2, 4, 6, 8], n_layers=2)
print("EQNN Variances:", results["eqnn_variances"])
print("HEA Variances: ", results["hea_variances"])

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

Run the unit tests:
```bash
pytest -v tests/
```

Run the barren plateau scaling benchmark:
```bash
python benchmarks/run_barren_plateau_benchmark.py
```

---

## 🌐 Interactive Web Application

Launch `web/index.html` to explore:
- **Lattice Symmetry Graph Explorer**: Visualizes 1D chains, triangular, Kagome, and 2D grid graphs with real-time representation theory metrics.
- **Dynamic Barren Plateau Scaling Analyzer**: Real-time log-linear gradient variance decay comparator.
- **3D Variational Loss Surface**: Contrast the smooth convex funnel of the Equivariant Ansatz against the flat barren plateau of HEA.
- **Sycamore Hardware Circuit Decomposition Inspector**.

---

## 📄 Citation & License

Developed by **Jasper Sands** under the **Apache-2.0 License**.
