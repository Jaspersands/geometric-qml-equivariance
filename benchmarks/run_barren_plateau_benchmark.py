"""
Benchmark runner for Barren Plateau scaling in Geometric QML.
"""

import time
import numpy as np
from geometric_qml.models import HeisenbergSpinChain, compute_singlet_dimension
from geometric_qml.barren_plateau import run_barren_plateau_scaling_study
from geometric_qml.vqe import EquivariantVQE


def run_benchmark():
    print("=" * 65)
    print("GEOMETRIC QML: SU(2) EQUIVARIANCE VS BARREN PLATEAUS")
    print("=" * 65)

    print("\n1. Singlet Subspace Catalan Scaling vs Total Hilbert Space:")
    for n in [2, 4, 6, 8, 10, 12, 14, 16]:
        c_dim = compute_singlet_dimension(n)
        tot_dim = 2**n
        compression = (1.0 - c_dim / tot_dim) * 100.0
        print(f"   N={n:2d} qubits | Singlet Dim = {c_dim:5d} | Total Dim = {tot_dim:6d} | Space Reduction = {compression:.2f}%")

    print("\n2. Running Gradient Variance Scaling Study...")
    t0 = time.time()
    results = run_barren_plateau_scaling_study(
        qubit_list=[2, 4, 6, 8],
        n_layers=2,
        n_samples=25,
        seed=42,
    )
    t1 = time.time()
    print(f"   -> Sampling completed in {t1 - t0:.2f} s")

    print("\n   Scaling Comparison:")
    print("   --------------------------------------------------------")
    print("   Qubits (N)  |  Equivariant Var[grad]  |  HEA Var[grad]")
    print("   --------------------------------------------------------")
    for i, n in enumerate(results["qubit_counts"]):
        v_eqnn = results["eqnn_variances"][i]
        v_hea = results["hea_variances"][i]
        print(f"       {n:2d}      |        {v_eqnn:12.6f}     |    {v_hea:12.6f}")

    print("\n3. Testing Equivariant VQE Ground State Finding (N=4)...")
    vqe = EquivariantVQE(n_qubits=4, n_layers=3, learning_rate=0.1)
    res_vqe = vqe.train(steps=30)
    print(f"   -> Exact Ground Energy: {res_vqe['exact_ground_energy']:.4f}")
    print(f"   -> VQE Final Energy:    {res_vqe['final_energy']:.4f}")
    print(f"   -> Absolute Error:      {res_vqe['energy_error']:.2e}")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark()
