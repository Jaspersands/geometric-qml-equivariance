"""
Tests for gradient-variance scaling and the equivariant VQE.
"""

import numpy as np
import pytest

pytest.importorskip("pennylane")

from geometric_qml.barren_plateau import (
    GradientVarianceAnalyzer,
    run_barren_plateau_scaling_study,
    fit_variance_scaling,
)
from geometric_qml.vqe import EquivariantVQE


def test_fit_variance_scaling_recovers_slope():
    qubits = [2, 4, 6, 8]
    variances = [0.5 * 2.0 ** (-0.75 * n) for n in qubits]
    fit = fit_variance_scaling(qubits, variances)
    assert np.isclose(fit["slope_log2"], -0.75, atol=1e-9)
    assert fit["r2"] > 0.999


def test_gradient_variance_sampling_shapes_and_positivity():
    an = GradientVarianceAnalyzer(n_qubits=4, ansatz_type="equivariant", n_layers=2)
    grads = an.sample_gradients(n_samples=6, seed=3)
    assert grads.shape == (6, 2)
    res = an.compute_gradient_variance(n_samples=6, seed=3)
    assert res["var_param_0"] > 0
    assert res["var_mean"] > 0
    assert res["n_params"] == 2
    with pytest.raises(ValueError):
        GradientVarianceAnalyzer(n_qubits=4, ansatz_type="nonsense")


def test_scaling_study_hea_plateaus_while_eqnn_decays_polynomially():
    # Depth L = N (the barren-plateau regime). HEA's mean gradient variance must
    # fall exponentially (log2 slope clearly negative); the EQNN per-bond
    # variance decays only polynomially, i.e. much more slowly.
    res = run_barren_plateau_scaling_study(qubit_list=[4, 6, 8], n_samples=25, seed=42, depth="linear")
    assert res["depths"] == [4, 6, 8]
    assert set(res["variances"]) == {"equivariant", "hea"}
    hea_slope = res["fits_mean"]["hea"]["slope_log2"]
    eqnn_slope = res["fits_mean"]["equivariant"]["slope_log2"]
    assert hea_slope < -0.4
    assert eqnn_slope > hea_slope + 0.2
    assert len(res["samples"]["hea"]) == 3


def test_scaling_study_fixed_depth_option():
    res = run_barren_plateau_scaling_study(qubit_list=[4, 6], n_layers=2, n_samples=5, seed=1, depth="fixed")
    assert res["depths"] == [2, 2]


def test_vqe_is_deterministic_and_converges_with_singlet_purity():
    r1 = EquivariantVQE(n_qubits=4, n_layers=3, learning_rate=0.15, seed=7).train(steps=60)
    r2 = EquivariantVQE(n_qubits=4, n_layers=3, learning_rate=0.15, seed=7).train(steps=60)
    assert np.isclose(r1["final_energy"], r2["final_energy"])
    assert r1["energy_error"] < 5e-2
    assert r1["singlet_purity"] > 0.999
    assert len(r1["energy_history"]) == 60
    assert r1["exact_ground_energy"] == pytest.approx(-6.464, abs=1e-3)  # 4-site open XXX chain
