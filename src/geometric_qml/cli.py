"""
Command-line interface: ``geometric-qml <command>``.

variance         gradient-variance sampling for an ansatz (barren-plateau probe)
variance-theory  loss variance of a deep circuit: Lie-algebraic prediction against sampling
dla              Dynamical Lie Algebra closure for an ansatz on N qubits
sectors          total-spin irrep multiplicities and the singlet (Catalan) dimension
transpile        equivariant circuit on a Sycamore snake grid
vqe              equivariant VQE ground-state search
benchmark        the full benchmark suite
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from .models import HeisenbergSpinChain, compute_singlet_dimension, total_spin_sectors, LATTICES
from .ansatz import EquivariantQuantumAnsatz, HardwareEfficientAnsatz, QAOABaselineAnsatz
from .dla import analyze_ansatz, equivariant_dla_dimension_formula


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geometric-qml", description="Geometric QML: SU(2)-equivariant circuits, DLA, barren plateaus")
    sub = parser.add_subparsers(dest="command")

    v = sub.add_parser("variance", help="Sample gradient variance")
    v.add_argument("--qubits", type=int, default=6)
    v.add_argument("--ansatz", choices=["equivariant", "hea", "qaoa"], default="equivariant")
    v.add_argument("--layers", type=int, default=3)
    v.add_argument("--samples", type=int, default=20)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--json", action="store_true")

    vt = sub.add_parser("variance-theory", help="Deep-circuit loss variance: Lie-algebraic prediction against sampling")
    vt.add_argument("--ansatz", choices=["su2", "hea", "sn"], default="su2",
                    help="su2: exchange gates; hea: hardware-efficient; sn: permutation-equivariant")
    vt.add_argument("--qubits", type=int, default=6)
    vt.add_argument("--depth", type=int, default=None, help="layers (default: 4N, or 32 for sn)")
    vt.add_argument("--samples", type=int, default=2000)
    vt.add_argument("--seed", type=int, default=0)
    vt.add_argument("--json", action="store_true")

    d = sub.add_parser("dla", help="Dynamical Lie Algebra closure")
    d.add_argument("--ansatz", choices=["equivariant", "hea", "qaoa", "sn"], default="equivariant")
    d.add_argument("--qubits", type=int, default=4)
    d.add_argument("--lattice", choices=LATTICES, default="1d_chain")
    d.add_argument("--json", action="store_true")

    s = sub.add_parser("sectors", help="Total-spin sector multiplicities")
    s.add_argument("--qubits", type=int, default=6)
    s.add_argument("--json", action="store_true")

    t = sub.add_parser("transpile", help="Transpile an EQNN onto a Sycamore grid")
    t.add_argument("--qubits", type=int, default=4)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--phased", action="store_true", help="use PhasedFSimGate")
    t.add_argument("--json", action="store_true")

    q = sub.add_parser("vqe", help="Equivariant VQE")
    q.add_argument("--qubits", type=int, default=4)
    q.add_argument("--layers", type=int, default=3)
    q.add_argument("--steps", type=int, default=60)
    q.add_argument("--lattice", choices=LATTICES, default="1d_chain")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--json", action="store_true")

    b = sub.add_parser("benchmark", help="Run the benchmark suite")
    b.add_argument("--json", action="store_true")
    b.add_argument("--quick", action="store_true")
    return parser


def _emit(payload: Dict[str, Any], as_json: bool, lines: List[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        print("\n".join(lines))


def _make_ansatz(kind: str, n: int, layers: int, lattice: str = "1d_chain"):
    if kind == "equivariant":
        from .models import lattice_edges
        return EquivariantQuantumAnsatz(n_qubits=n, n_layers=layers, weight_sharing=False, couplings=lattice_edges(n, lattice))
    if kind == "hea":
        return HardwareEfficientAnsatz(n_qubits=n, n_layers=layers)
    return QAOABaselineAnsatz(n_qubits=n, n_layers=layers)


def cmd_variance(args) -> int:
    from .barren_plateau import GradientVarianceAnalyzer
    an = GradientVarianceAnalyzer(n_qubits=args.qubits, ansatz_type=args.ansatz, n_layers=args.layers)
    res = an.compute_gradient_variance(n_samples=args.samples, seed=args.seed)
    _emit(res, args.json, [
        f"[*] {args.ansatz} on {args.qubits} qubits, L = {args.layers}, {args.samples} samples",
        f"[+] Var[∂θ₀⟨H⟩] = {res['var_param_0']:.4e}   mean Var = {res['var_mean']:.4e}   mean |∇| = {res['mean_norm']:.4f}   params = {res['n_params']}",
    ])
    return 0


def cmd_variance_theory(args) -> int:
    from .variance_theory import compare_prediction, deep_depth, default_setup, predict_loss

    n = args.qubits
    if n % 2 or n < 2:
        raise SystemExit("variance-theory uses the singlet product (hea, su2) or Néel (sn) input; "
                         "give an even number of qubits, at least 2")
    L = args.depth if args.depth is not None else deep_depth(args.ansatz, n)
    psi, O, label = default_setup(args.ansatz, n)
    r = compare_prediction(args.ansatz, n, L, args.samples, args.seed, psi, O)
    pred = predict_loss(args.ansatz, n, psi, O)
    ideals = pred["purities_obs"].get("dimensions", {})
    r["ideal_dimensions"] = {str(k): v for k, v in ideals.items()} if isinstance(ideals, dict) else {}
    r["setup"] = label
    zm = (r["measured_mean"] - r["predicted_mean"]) / r["measured_mean_se"] if not r["zero_variance"] else 0.0
    lines = [
        f"[*] {args.ansatz} circuit on {n} qubits, L = {L}, {args.samples} random parameter vectors",
        f"[*] input and observable: {label}",
        f"[+] dim 𝔤 = {r['dla_dimension']}" + (f"; simple ideals (keyed by S): {r['ideal_dimensions']}" if r["ideal_dimensions"] else ""),
        (f"[+] Var[ℓ]: predicted 0 and measured {r['measured']:.1e}: the loss is constant (round-off only), z set to 0"
         if r["zero_variance"] else
         f"[+] Var[ℓ]: predicted {r['predicted']:.5e}   measured {r['measured']:.5e} ± {r['measured_se']:.1e}   z = {r['z']:+.2f}"),
        f"[+] E[ℓ]:   predicted {r['predicted_mean']:+.5f}   measured {r['measured_mean']:+.5f} ± {r['measured_mean_se']:.1e}   z = {zm:+.2f}",
    ]
    _emit(r, args.json, lines)
    return 0


def cmd_dla(args) -> int:
    if args.ansatz == "sn":
        from .dla import permutation_dla_dimension_by_closure, permutation_dla_dimension_formula
        res = permutation_dla_dimension_by_closure(args.qubits)
        payload = {k: v for k, v in res.items() if k != "basis"}
        payload["closed_form"] = permutation_dla_dimension_formula(args.qubits)
        _emit(payload, args.json, [
            f"[*] DLA closure for the S_N-equivariant generators ΣX, ΣY, ΣZZ on {args.qubits} qubits "
            f"(spin blocks, exact arithmetic mod {' and '.join(str(q) for q in res['primes'])})",
            f"[+] dim 𝔤 = {res['dla_dimension']}  (su(2^N) has {res['full_su_dimension']}), closed = {res['closed']}",
            f"[+] closed form C(N+3, 3) − ⌊N/2⌋ = {payload['closed_form']}",
        ])
        return 0
    ansatz = _make_ansatz(args.ansatz, args.qubits, 1, args.lattice)
    if args.ansatz == "equivariant" and args.qubits > 6:
        from .dla import equivariant_dla_dimension_by_closure
        res = equivariant_dla_dimension_by_closure(ansatz.couplings, args.qubits)   # SU(2)-reduced representation
    else:
        res = analyze_ansatz(ansatz)
    payload = {k: v for k, v in res.items() if k != "basis"}
    lines = [
        f"[*] DLA closure for {args.ansatz} on {args.qubits} qubits ({args.lattice})",
        f"[+] dim 𝔤 = {res['dla_dimension']}  (su(2^N) has {res['full_su_dimension']}), closed = {res['closed']}, rounds = {res['iteration_dimensions']}",
    ]
    if args.ansatz == "equivariant" and args.qubits % 2 == 0:
        formula = equivariant_dla_dimension_formula(args.qubits)
        payload["closed_form"] = formula
        lines.append(f"[+] closed form Σ_S(m_S²−1)+1 = {formula}")
    _emit(payload, args.json, lines)
    return 0


def cmd_sectors(args) -> int:
    sectors = total_spin_sectors(args.qubits)
    payload = {"n_qubits": args.qubits, "sectors": {str(k): v for k, v in sectors.items()},
               "singlet_dimension": compute_singlet_dimension(args.qubits), "hilbert_dimension": 2**args.qubits}
    lines = [f"[*] Total-spin sectors of {args.qubits} spin-½ (Hilbert dim {2**args.qubits})"]
    for S, m in sorted(sectors.items()):
        lines.append(f"    S = {S:<4} multiplicity {m:4d}   (× {int(2*S+1)} states)")
    lines.append(f"[+] singlet dimension C_{args.qubits//2} = {payload['singlet_dimension']}")
    _emit(payload, args.json, lines)
    return 0


def cmd_transpile(args) -> int:
    from .sycamore_transpile import SycamoreEquivariantTranspiler
    tr = SycamoreEquivariantTranspiler(n_qubits=args.qubits, use_phased_fsim=args.phased)
    c = tr.transpile_circuit(n_layers=args.layers, params=np.linspace(0.1, 0.5, args.layers))
    counts = tr.gate_counts(c)
    _emit({"counts": counts, "qubits": [str(q) for q in tr.grid_qubits], "circuit": str(c)}, args.json,
          [f"[*] {args.qubits}-qubit EQNN, {args.layers} layers on Sycamore snake grid", f"[+] gate counts: {counts}", str(c)])
    return 0


def cmd_vqe(args) -> int:
    from .vqe import EquivariantVQE
    res = EquivariantVQE(n_qubits=args.qubits, n_layers=args.layers, seed=args.seed, lattice_type=args.lattice).train(steps=args.steps)
    payload = {k: v for k, v in res.items() if k != "energy_history"}
    _emit(payload, args.json, [
        f"[*] Equivariant VQE on {args.qubits}-qubit {args.lattice}, L = {args.layers}, {args.steps} Adam steps",
        f"[+] E = {res['final_energy']:.6f}  exact = {res['exact_ground_energy']:.6f}  error = {res['energy_error']:.2e}  singlet purity = {res['singlet_purity']:.6f}",
    ])
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "benchmark":
        from .benchmark import run_benchmark
        return run_benchmark(as_json=args.json, quick=args.quick)
    return {"variance": cmd_variance, "variance-theory": cmd_variance_theory, "dla": cmd_dla, "sectors": cmd_sectors,
            "transpile": cmd_transpile, "vqe": cmd_vqe}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
