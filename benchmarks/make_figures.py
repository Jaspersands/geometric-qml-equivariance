"""
python benchmarks/make_figures.py [--quick] [--plot-only] [--lines] [--recheck]

Runs the comparison of measured loss variance with the Lie-algebraic
prediction, the depth scans, g-sim runs of the S_N circuit beyond statevector
size, and the expressivity comparison; writes figures/*.png and
figures/results.json. A full run keeps any other keys already in results.json
(such as ``su2_recheck``).

``--plot-only`` redraws from an existing results.json. ``--lines`` recomputes
only the prediction lines (N = 4 to 24, a few seconds) into it. ``--recheck``
adds an 8000-sample run of the exchange circuit at L = 4N and 16N (N = 6, 8, 10).
The full run takes about ten minutes on a laptop, the recheck about four.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometric_qml.expressivity import expressivity_comparison, sn_energy_spread  # noqa: E402
from geometric_qml.gsim import permutation_equivariant_sequence, permutation_equivariant_simulator  # noqa: E402
from geometric_qml.variance_theory import (agreement, depth_scan, singlet_input_prediction, variance_study,  # noqa: E402
                                           variance_with_error)

FIG = ROOT / "figures"
COLORS = {"hea": "#c25a3a", "su2": "#5f4b9c", "sn": "#1a9472"}      # validated: CVD ΔE ≥ 8, contrast ≥ 3:1
MARKERS = {"hea": "s", "su2": "o", "sn": "^"}
NAMES = {"hea": "hardware-efficient", "su2": "SU(2)-equivariant", "sn": "$S_N$-equivariant"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"


def compute(quick: bool) -> dict:
    out: dict = {"quick": quick}
    t0 = time.time()
    qubits = (4, 6, 8) if quick else (4, 6, 8, 10, 12)
    study = variance_study(qubits=qubits, n_samples=1000 if quick else 2000, seed=0)
    for r in study["rows"]:
        print(f"  {r['kind']:4s} N={r['n_qubits']:2d} L={r['depth']:3d}  predicted {r['predicted']:.4e}  "
              f"measured {r['measured']:.4e} ± {r['measured_se']:.1e}  z = {r['z']:+.2f}", flush=True)
    out["study"] = study
    print(f"[study] {time.time() - t0:.0f} s, agreement {study['agreement']}", flush=True)

    t0 = time.time()
    scans = []
    for kind in ("hea", "su2", "sn"):
        for n in ((6, 8) if quick else (6, 8, 10)):
            scans.append(depth_scan(kind, n, depths=(1, 2, 4, 8, 16, 32, 64), n_samples=600 if quick else 1000, seed=7))
    out["depth_scans"] = scans
    print(f"[depth scans] {time.time() - t0:.0f} s", flush=True)

    # S_N circuit beyond statevector size: g-sim in the adjoint representation
    t0 = time.time()
    gsim_rows = []
    for n in ((14, 16) if quick else (14, 16, 20, 24)):
        t = time.time()
        sim = permutation_equivariant_simulator(n)
        build = time.time() - t
        rng = np.random.default_rng(100 + n)
        th = rng.uniform(0, 2 * np.pi, (1000 if quick else 2000, 3 * 32))
        t = time.time()
        x = sim.expectation(permutation_equivariant_sequence(32), th)
        run = time.time() - t
        v, se = variance_with_error(x)
        pred = sim.predicted_variance()
        gsim_rows.append({"kind": "sn", "n_qubits": n, "depth": 32, "predicted": pred, "measured": v, "measured_se": se,
                          "z": (v - pred) / se, "obs_scale": n * (n - 1) / 2, "adjoint_dimension": sim.dimension + 1,
                          "build_seconds": build, "run_seconds": run, "n_samples": int(x.size)})
        print(f"  g-sim N={n}: predicted {pred:.4e} measured {v:.4e} ± {se:.1e} z = {(v - pred) / se:+.2f} "
              f"(dim {sim.dimension + 1}, build {build:.1f} s, {x.size} circuits {run:.1f} s)", flush=True)
    out["gsim"] = gsim_rows
    out["agreement_all"] = agreement(study["rows"] + gsim_rows)
    out["prediction_lines"] = prediction_lines()
    print(f"[g-sim] {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    expr = {}
    for lattice in ("1d_chain", "ring"):
        expr[lattice] = expressivity_comparison(n_qubits=6 if quick else 8, lattice=lattice, budget=112,
                                                n_restarts=2 if quick else 4, seed=0)
        expr[lattice]["sn_spread_singlet"] = sn_energy_spread(6 if quick else 8, lattice, initial="singlet")
        expr[lattice]["sn_spread_neel"] = sn_energy_spread(6 if quick else 8, lattice, initial="neel")
        e = expr[lattice]
        print(f"  {lattice}: E0 = {e['exact_energy']:.6f}; " + "; ".join(
            f"{k} ({e[k]['n_params']} params, L = {e[k]['depth']}): error {e[k]['error']:.3e}, fidelity {e[k]['fidelity']:.6f}"
            for k in ("hea", "su2", "sn")), flush=True)
    out["expressivity"] = expr
    print(f"[expressivity] {time.time() - t0:.0f} s", flush=True)
    return out


def prediction_lines(n_max: int = 24) -> dict:
    """
    Normalised deep-circuit predictions, Var[ℓ] / (Tr O² / 2^N), for N = 4, 6, …, n_max.
    hea: (1 − 2^−N) Tr O² / (4^N − 1) divided by Tr O² / 2^N, i.e. 1/(2^N + 1).
    su2: Var_spec(H₀)/(C + 1) from the S_z-sector traces (``singlet_input_prediction``).
    sn: g-sim's Σ_S P_S(ρ) P_S(O) / dim 𝔤_S, no 2^N object needed.
    """
    lines = {"hea": {}, "su2": {}, "sn": {}, "crossover": None}
    for n in range(4, n_max + 1, 2):
        lines["hea"][n] = 1.0 / (2**n + 1)
        r = singlet_input_prediction(n)
        lines["su2"][n] = r["variance"] / r["obs_scale"]
        lines["sn"][n] = permutation_equivariant_simulator(n, dynamics=False).predicted_variance() / (n * (n - 1) / 2)
    below = [n for n in sorted(lines["su2"]) if lines["su2"][n] < lines["sn"][n]]
    if below:
        lines["crossover"] = {"last_su2_above": below[0] - 2, "first_su2_below": below[0]}
    print("  su2 / sn normalised: " + ", ".join(f"{n}: {lines['su2'][n]:.3g} / {lines['sn'][n]:.3g}" for n in sorted(lines["su2"])),
          flush=True)
    return lines


def recheck() -> dict:
    """High-statistics check that L = 4N is deep enough for the exchange circuit."""
    from geometric_qml.variance_theory import default_setup, measure_loss_variance, predict_loss

    rows = []
    for n in (6, 8, 10):
        psi, O, _ = default_setup("su2", n)
        pred = predict_loss("su2", n, psi, O)["variance"]
        for L in (4 * n, 16 * n):
            m = measure_loss_variance("su2", n, L, 8000, seed=1234 + L, psi0=psi, obs=O)
            rows.append({"n_qubits": n, "depth": L, "predicted": pred, "measured": m["variance"],
                         "measured_se": m["variance_se"], "z": (m["variance"] - pred) / m["variance_se"]})
            print(f"  su2 N={n:2d} L={L:3d}: measured/predicted {m['variance'] / pred:.4f} ± {m['variance_se'] / pred:.4f}", flush=True)
    return {"rows": rows, "agreement": agreement(rows)}


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.labelsize": 9.5, "axes.titlesize": 10, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5, "font.family": "DejaVu Sans",
        "lines.linewidth": 1.6, "lines.markersize": 6, "errorbar.capsize": 0,
    })


def _norm_rows(rows, kind):
    rs = sorted([r for r in rows if r["kind"] == kind], key=lambda r: r["n_qubits"])
    n = np.array([r["n_qubits"] for r in rs])
    s = np.array([r["obs_scale"] for r in rs])
    return n, np.array([r["predicted"] for r in rs]) / s, np.array([r["measured"] for r in rs]) / s, \
        np.array([r["measured_se"] for r in rs]) / s


def variance_axes(ax, res, labels=True, narrow=False):
    rows = res["study"]["rows"]
    lines = res["prediction_lines"]
    for kind in ("hea", "su2", "sn"):
        c = COLORS[kind]
        n, pred, meas, se = _norm_rows(rows, kind)
        ln = np.array(sorted(int(k) for k in lines[kind]))
        lp = np.array([lines[kind][str(k)] if str(k) in lines[kind] else lines[kind][k] for k in ln])
        # solid where there are measurements, dashed where the line is the formula alone
        n_meas = max([int(x) for x in n] + [r["n_qubits"] for r in res.get("gsim", []) if r["kind"] == kind])
        inside = ln <= n_meas
        ax.plot(ln[inside], lp[inside], color=c, lw=1.4, zorder=2)
        if np.any(~inside):
            k0 = int(np.sum(inside)) - 1
            ax.plot(ln[k0:], lp[k0:], color=c, lw=1.3, ls=(0, (3.5, 2.5)), zorder=2)
        ax.errorbar(n, meas, yerr=se, fmt=MARKERS[kind], color=c, mfc=c, mec="white", mew=0.8, ms=6.5, elinewidth=1.0,
                    zorder=3)
        if kind == "sn" and res.get("gsim"):
            g = sorted(res["gsim"], key=lambda r: r["n_qubits"])
            gn = np.array([r["n_qubits"] for r in g])
            gs = np.array([r["obs_scale"] for r in g])
            ax.errorbar(gn, np.array([r["measured"] for r in g]) / gs, yerr=np.array([r["measured_se"] for r in g]) / gs,
                        fmt=MARKERS[kind], color=c, mfc="white", mec=c, mew=1.2, ms=6.5, elinewidth=1.0, zorder=3)
        if labels:
            text = {"hea": "hardware-efficient\n$\\dim\\mathfrak{g} = 4^N - 1$",
                    "su2": "SU(2)-equivariant\n$\\dim\\mathfrak{g} = \\sum_S (m_S^2 - 1) + 1$",
                    "sn": "$S_N$-equivariant\n$\\dim\\mathfrak{g} = \\binom{N+3}{3} - \\lfloor N/2 \\rfloor$"}[kind]
            # positions in data coordinates, clear of the other curves
            xy = {"hea": (8.5, 2.5e-6) if narrow else (15.6, 1.1e-7), "su2": (9.0, 0.22), "sn": (24.0, 1.2e-2)}[kind]
            ax.text(*xy, text, fontsize=8.2, color=INK, ha="right" if kind == "sn" else "left", va="center")
    ax.set_yscale("log")
    ax.set_xlabel("qubits $N$")
    ax.set_ylabel("Var[ℓ] / (Tr $O^2$ / $2^N$)")
    ax.set_xticks(range(4, 25, 4))
    ax.set_xlim(3, 25.5)


def fig_variance(res):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    variance_axes(ax, res)
    handles = [Line2D([], [], color=MUTED, lw=1.4, label="prediction (deep circuit, Haar on $e^{\\mathfrak{g}}$)"),
               Line2D([], [], color=MUTED, lw=1.3, ls=(0, (3.5, 2.5)), label="prediction, no measurement at these sizes"),
               Line2D([], [], color=MUTED, marker="o", lw=0, mec="white", label="measured, statevector"),
               Line2D([], [], color=MUTED, marker="o", lw=0, mfc="white", mec=MUTED, mew=1.2,
                      label="measured, g-sim (adjoint representation)")]
    ax.legend(handles=handles, loc="lower left")
    ax.set_title("Loss variance against the Lie-algebraic prediction")
    ag = res["agreement_all"]
    ax.text(0.99, 0.99, f"χ² = {ag['chi2']:.1f} over {ag['dof']} points (p = {ag['p_value']:.2f})", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "variance_vs_theory.png", dpi=150)
    plt.close(fig)


def fig_depth(res):
    import matplotlib.pyplot as plt

    scans = res["depth_scans"]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.2), sharey=True)
    shades = {6: 0.45, 8: 0.72, 10: 1.0}
    for ax, kind in zip(axes, ("hea", "su2", "sn")):
        for s in [s for s in scans if s["kind"] == kind]:
            n = s["n_qubits"]
            d = np.array(s["depths"])
            ratio = np.array(s["measured"]) / s["predicted"]
            err = np.array(s["measured_se"]) / s["predicted"]
            col = _blend(COLORS[kind], shades.get(n, 1.0))
            ax.errorbar(d, ratio, yerr=err, fmt=MARKERS[kind] + "-", color=col, mec="white", mew=0.6, ms=5, lw=1.2,
                        elinewidth=0.9, label=f"N = {n}")
        ax.axhline(1.0, color="#9a9a9a", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.set_ylim(0.6, 90)
        ax.set_xlabel("layers $L$")
        ax.set_title(NAMES[kind], fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7.8, handlelength=1.6)
    axes[0].set_ylabel("measured / predicted variance")
    fig.tight_layout()
    fig.savefig(FIG / "depth_convergence.png", dpi=150)
    plt.close(fig)


def _blend(hex_color, t):
    """Mix a colour with white: t = 1 gives the colour, t = 0 white."""
    h = hex_color.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return tuple(1 - t * (1 - rgb))


def expressivity_axes(ax, res, compact=False):
    expr = res["expressivity"]
    kinds = ("hea", "su2", "sn")
    floor = 1e-13
    for i, kind in enumerate(kinds):
        y = len(kinds) - 1 - i
        for lattice, mk, fill in (("1d_chain", "o", True), ("ring", "D", False)):
            e = expr[lattice]
            rel = max(e[kind]["error"] / abs(e["exact_energy"]), floor)
            ax.plot([rel], [y + (0.12 if lattice == "1d_chain" else -0.12)], mk, color=COLORS[kind],
                    mfc=COLORS[kind] if fill else "white", mec=COLORS[kind], mew=1.3, ms=7 if mk == "o" else 6, zorder=3)
            if not compact:
                lab = "below $10^{-13}$" if rel <= floor else f"{rel:.1e}"
                ax.annotate(lab, (rel, y + (0.12 if lattice == "1d_chain" else -0.12)), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=7.8, color=MUTED)
    e, f = expr["1d_chain"], expr["ring"]

    def depth_label(k):
        if e[k]["depth"] == f[k]["depth"]:
            return f"L = {e[k]['depth']}"
        return f"L = {e[k]['depth']} (chain), {f[k]['depth']} (ring)"

    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([f"{NAMES[k]}\n{max(e[k]['n_params'], f[k]['n_params'])} parameters, {depth_label(k)}"
                        for k in reversed(kinds)], fontsize=8.2)
    ax.set_xscale("log")
    ax.set_xlim(3e-14, 3)
    ax.set_ylim(-0.6, len(kinds) - 0.4)
    ax.grid(False)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("relative energy error $(E - E_0)/|E_0|$")
    ax.plot([], [], "o", color=MUTED, label="open chain")
    ax.plot([], [], "D", color=MUTED, mfc="white", mew=1.3, label="ring")
    ax.legend(loc="upper center", fontsize=8)


def fig_expressivity(res):
    import matplotlib.pyplot as plt

    n = res["expressivity"]["1d_chain"]["n_qubits"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    expressivity_axes(ax, res)
    ax.set_title(f"Ground-state search at N = {n}, best of {len(res['expressivity']['1d_chain']['hea']['energies'])} starts")
    fig.tight_layout()
    fig.savefig(FIG / "expressivity.png", dpi=150)
    plt.close(fig)


def fig_tradeoff(res):
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    variance_axes(a, res, narrow=True)
    a.set_title("Trainability: loss variance of deep circuits")
    expressivity_axes(b, res)
    n = res["expressivity"]["1d_chain"]["n_qubits"]
    b.set_title(f"Expressivity: ground-state search at N = {n}")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.png", dpi=150)
    plt.close(fig)


def plot(res):
    style()
    FIG.mkdir(exist_ok=True)
    fig_variance(res)
    fig_depth(res)
    fig_expressivity(res)
    fig_tradeoff(res)
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}: {p.stat().st_size / 1024:.0f} KB")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if k not in ("samples", "best_params")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main(argv):
    import matplotlib

    matplotlib.use("Agg")
    path = FIG / "results.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    if "--plot-only" in argv or "--recheck" in argv or "--lines" in argv:
        res = old
        if "--lines" in argv:
            res["prediction_lines"] = _jsonable(prediction_lines())
        if "--recheck" in argv:
            res["su2_recheck"] = _jsonable(recheck())
        if "--lines" in argv or "--recheck" in argv:
            path.write_text(json.dumps(res, indent=1))
    else:
        res = _jsonable(compute(quick="--quick" in argv))
        for k, v in old.items():                          # keep results this run does not produce (su2_recheck)
            res.setdefault(k, v)
        FIG.mkdir(exist_ok=True)
        path.write_text(json.dumps(res, indent=1))
    plot(res)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
