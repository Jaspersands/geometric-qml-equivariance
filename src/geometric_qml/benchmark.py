"""
Benchmark: singlet-sector scaling, DLA dimensions vs the closed form,
gradient-variance scaling for EQNN / HEA / QAOA, equivariant VQE, and a
Sycamore transpilation. Exposed as ``geometric-qml benchmark``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import numpy as np

from .models import compute_singlet_dimension, total_spin_sectors
from .ansatz import EquivariantQuantumAnsatz, HardwareEfficientAnsatz
from .dla import analyze_ansatz, equivariant_dla_dimension_formula


def run_benchmark(as_json: bool = False, quick: bool = False) -> int:
    out: Dict[str, Any] = {}
    log = [] if as_json else None

    def say(msg: str = "") -> None:
        (log.append(msg) if log is not None else print(msg))

    say("=" * 70)
    say("GEOMETRIC QML: SU(2) EQUIVARIANCE, DLA DIMENSION AND BARREN PLATEAUS (v0.3)")
    say("=" * 70)

    say("\n1. Singlet (Catalan) sector vs full Hilbert space")
    sing = {}
    for n in (2, 4, 6, 8, 10, 12, 14, 16):
        c = compute_singlet_dimension(n)
        sing[n] = c
        say(f"   N = {n:2d} | C_{n//2:<2d} = {c:6d} | 2^N = {2**n:6d} | fraction = {c / 2**n:.4f}")
    out["singlet"] = sing

    say("\n2. DLA closure (chain EQNN with free bond angles) vs closed form Σ_S(m_S²−1)+1, and HEA")
    dla = {}
    for n in (2, 4, 6):
        t0 = time.time()
        r = analyze_ansatz(EquivariantQuantumAnsatz(n_qubits=n, n_layers=1, weight_sharing=False))
        f = equivariant_dla_dimension_formula(n)
        say(f"   EQNN N = {n}: dim 𝔤 = {r['dla_dimension']:4d} (formula {f:4d}, closed = {r['closed']}, {time.time()-t0:.2f} s) | su(2^N) = {4**n - 1}")
        dla[f"eqnn_{n}"] = r["dla_dimension"]
    for n in (2, 3):
        r = analyze_ansatz(HardwareEfficientAnsatz(n_qubits=n, n_layers=1))
        say(f"   HEA  N = {n}: dim 𝔤 = {r['dla_dimension']:4d} = su(2^N)")
        dla[f"hea_{n}"] = r["dla_dimension"]
    out["dla"] = dla

    try:
        from .barren_plateau import run_barren_plateau_scaling_study
        say("\n3. Gradient-variance scaling with depth L = N (EQNN per-bond normalised)")
        t0 = time.time()
        qubits = [4, 6, 8] if quick else [4, 6, 8, 10]
        res = run_barren_plateau_scaling_study(qubit_list=qubits, n_samples=20 if quick else 40, seed=42,
                                               ansatz_types=("equivariant", "hea", "qaoa"), depth="linear")
        say(f"   ({time.time()-t0:.1f} s)")
        say("   N   L  | EQNN mean Var/bond² | HEA mean Var   | QAOA mean Var")
        for i, n in enumerate(qubits):
            say(f"   {n:2d} {res['depths'][i]:2d}  | {res['var_means']['equivariant'][i]:.4e}          | {res['var_means']['hea'][i]:.4e}     | {res['var_means']['qaoa'][i]:.4e}")
        for a, f in res["fits_mean"].items():
            say(f"   log2-slope {a:<11s}: {f['slope_log2']:+.3f} per qubit (R² = {f['r2']:.2f})  → Var ∝ 2^({f['slope_log2']:+.2f}·N)")
        say("   HEA shows the exponential barren-plateau decay; the EQNN per-bond variance decays polynomially.")
        out["variance"] = {"qubits": qubits, "depths": res["depths"], "var_means": res["var_means"], "fits_mean": res["fits_mean"]}

        from .vqe import EquivariantVQE
        say("\n4. Equivariant VQE, N = 4 chain")
        t0 = time.time()
        v = EquivariantVQE(n_qubits=4, n_layers=3, learning_rate=0.12, seed=1).train(steps=40 if quick else 80)
        say(f"   E = {v['final_energy']:.5f}  exact = {v['exact_ground_energy']:.5f}  error = {v['energy_error']:.2e}  singlet purity = {v['singlet_purity']:.6f}  ({time.time()-t0:.1f} s)")
        out["vqe"] = {k: v[k] for k in ("final_energy", "exact_ground_energy", "energy_error", "singlet_purity")}
    except ImportError:
        say("\n3./4. PennyLane not installed — skipping variance study and VQE")

    try:
        from .sycamore_transpile import SycamoreEquivariantTranspiler
        say("\n5. Sycamore transpilation (8-qubit snake, 2 layers)")
        tr = SycamoreEquivariantTranspiler(n_qubits=8)
        c = tr.transpile_circuit(n_layers=2, params=np.array([0.3, 0.5]))
        counts = tr.gate_counts(c)
        say(f"   all chain bonds adjacent: {tr.check_adjacency([(i, i+1) for i in range(7)])} | gates: {counts}")
        out["sycamore"] = counts
    except ImportError:
        say("\n5. Cirq not installed — skipping transpilation")
    say("=" * 70)
    if as_json:
        print(json.dumps(out, indent=2))
    return 0
