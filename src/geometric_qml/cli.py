"""
Command-Line Interface for geometric_qml.
"""

from __future__ import annotations
import argparse
import sys
import json
import numpy as np
from .models import HeisenbergSpinChain, compute_singlet_dimension
from .barren_plateau import GradientVarianceAnalyzer, run_barren_plateau_scaling_study
from .dla import DynamicalLieAlgebraAnalyzer
from .fermi_hubbard import FermiHubbardChain
from .sycamore_transpile import SycamoreEquivariantTranspiler
from .export_openfermion import save_hamiltonian_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geometric-qml",
        description="Geometric QML & SU(2) Equivariant Quantum Circuits CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: variance
    var_parser = subparsers.add_parser("variance", help="Sample gradient variance to benchmark barren plateaus")
    var_parser.add_argument("--qubits", type=int, default=6, help="Number of qubits (even integer)")
    var_parser.add_argument("--ansatz", type=str, choices=["equivariant", "hea", "qaoa"], default="equivariant")
    var_parser.add_argument("--layers", type=int, default=3, help="Circuit layers")
    var_parser.add_argument("--samples", type=int, default=20, help="Number of Haar parameter samples")

    # Command: dla
    dla_parser = subparsers.add_parser("dla", help="Analyze Dynamical Lie Algebra dimension")
    dla_parser.add_argument("--sites", type=int, default=2, help="Fermi-Hubbard sites (qubits = 2 * sites)")

    # Command: transpile
    tr_parser = subparsers.add_parser("transpile", help="Transpile equivariant circuit onto Google Sycamore grid")
    tr_parser.add_argument("--qubits", type=int, default=4, help="Number of qubits")
    tr_parser.add_argument("--layers", type=int, default=2, help="Circuit layers")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "variance":
        print(f"[*] Analyzing gradient variance for {args.ansatz.upper()} on {args.qubits} qubits (L={args.layers})...")
        analyzer = GradientVarianceAnalyzer(n_qubits=args.qubits, ansatz_type=args.ansatz, n_layers=args.layers)
        res = analyzer.compute_gradient_variance(n_samples=args.samples)
        print(f"[+] Empirical Var[grad_0]: {res['var_param_0']:.6e} (Mean Norm: {res['mean_norm']:.4f})")
        print(f"[+] Total Parameters: {analyzer.ansatz.num_params()}")
        return 0

    elif args.command == "dla":
        print(f"[*] Building Fermi-Hubbard chain ({args.sites} sites -> {2*args.sites} qubits)...")
        fh = FermiHubbardChain(n_sites=args.sites)
        print(f"[+] Qubits: {fh.n_qubits}, Singlet Irrep Dim: {compute_singlet_dimension(fh.n_qubits)}")
        return 0

    elif args.command == "transpile":
        print(f"[*] Transpiling {args.qubits}-qubit EQNN ({args.layers} layers) to Google Sycamore PhasedFSim...")
        transpiler = SycamoreEquivariantTranspiler(n_qubits=args.qubits)
        c = transpiler.transpile_circuit(n_layers=args.layers, params=list(np.linspace(0.1, 0.5, args.layers)))
        print(f"[+] Transpiled Circuit:\n{c}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
