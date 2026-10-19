"""
Tests for the Lie-algebraic variance prediction, the S_N-equivariant ansatz,
the batched statevector simulator, g-sim and the expressivity comparison.
"""

import json
import math

import numpy as np
import pytest

from geometric_qml.ansatz import EquivariantQuantumAnsatz, HardwareEfficientAnsatz, PermutationEquivariantAnsatz
from geometric_qml import dla as dla_module
from geometric_qml.dla import (DynamicalLieAlgebraAnalyzer, _lie_closure_mod_p, analyze_ansatz,
                               permutation_dla_dimension_by_closure, permutation_dla_dimension_formula,
                               permutation_integer_generators, spin_matrices)
from geometric_qml.models import (HeisenbergSpinChain, create_singlet_product_state_vector, get_casimir_operator,
                                  pauli_string_sparse)
from geometric_qml.qgnn import permutation_matrix
from geometric_qml.statevector import (SzSector, energy_and_gradient, hardware_efficient_circuit, num_params,
                                       permutation_equivariant_circuit, run_circuit, run_circuit_sector, sample_losses,
                                       su2_equivariant_circuit, expectation)
from geometric_qml.variance_theory import (SchurBasis, compare_prediction, default_setup, g_purities,
                                           measure_loss_variance, neel_state_vector, predict_loss,
                                           reductive_decomposition, schur_purities, singlet_input_prediction,
                                           variance_with_error)
from geometric_qml.cli import main as cli_main


def _random_hermitian(n, rng):
    a = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    return a + a.conj().T


def _random_state(n, rng):
    v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def su2_decomposition_4():
    a = EquivariantQuantumAnsatz(n_qubits=4, n_layers=1, weight_sharing=False)
    return reductive_decomposition(analyze_ansatz(a)["basis"], a.generators())


@pytest.fixture(scope="module")
def sn_decomposition_4():
    a = PermutationEquivariantAnsatz(n_qubits=4, n_layers=1)
    return reductive_decomposition(DynamicalLieAlgebraAnalyzer(a.generators()).compute_dla_dimension()["basis"],
                                   a.generators())


# ----------------------------------------------------------------------------- #
# g-purities
# ----------------------------------------------------------------------------- #
def test_ideal_structure_found_from_dense_basis(su2_decomposition_4, sn_decomposition_4):
    # exchange circuits, N = 4: su(2) on W_1 (m = 3 → 8) and su(2) on W_0 (m = 2 → 3), plus a 1-d centre
    assert su2_decomposition_4["ideal_dimensions"] == [3, 8]
    assert su2_decomposition_4["center"].shape[1] == 1
    # S_N circuit, N = 4: su(3) on V_1 and su(5) on V_2, plus a 1-d centre
    assert sn_decomposition_4["ideal_dimensions"] == [8, 24]
    assert sn_decomposition_4["center"].shape[1] == 1
    # hardware-efficient: su(2^N) is simple, no centre
    h = HardwareEfficientAnsatz(n_qubits=2, n_layers=1)
    dec = reductive_decomposition(analyze_ansatz(h)["basis"], h.generators())
    assert dec["ideal_dimensions"] == [15] and dec["center"].shape[1] == 0


def test_purity_ignores_identity_and_is_full_norm_inside_the_algebra(su2_decomposition_4):
    rng = np.random.default_rng(0)
    X = _random_hermitian(4, rng)
    p = g_purities(X, su2_decomposition_4)
    q = g_purities(X + 3.7 * np.eye(16), su2_decomposition_4)
    assert np.allclose(p["ideals"], q["ideals"]) and np.isclose(p["center"], q["center"])
    s1 = schur_purities(X, 4, "su2")
    s2 = schur_purities(X + 3.7 * np.eye(16), 4, "su2")
    assert np.allclose(list(s1["ideals"].values()), list(s2["ideals"].values())) and np.isclose(s1["center"], s2["center"])
    # the chain Hamiltonian is a sum of generators: its purities exhaust Tr H² (it is traceless)
    H = HeisenbergSpinChain(4).get_matrix()
    s = schur_purities(H, 4, "su2")
    assert s["total"] == pytest.approx(np.trace(H @ H).real, rel=1e-10)
    # a generic operator is not in i𝔤: its projection is strictly smaller than its traceless norm
    X0 = X - np.trace(X) / 16 * np.eye(16)
    assert s1["total"] < np.trace(X0 @ X0).real


@pytest.mark.parametrize("n", [4, 6])
def test_schur_purities_match_dense_projection_su2(n):
    a = EquivariantQuantumAnsatz(n_qubits=n, n_layers=1, weight_sharing=False)
    dec = reductive_decomposition(analyze_ansatz(a)["basis"], a.generators())
    rng = np.random.default_rng(n)
    for X in (_random_hermitian(n, rng), _random_state(n, rng), neel_state_vector(n), create_singlet_product_state_vector(n)):
        g = g_purities(X, dec)
        s = schur_purities(X, n, "su2")
        assert sorted(g["ideals"]) == pytest.approx(sorted(s["ideals"].values()), rel=1e-9, abs=1e-12)
        assert g["center"] == pytest.approx(s["center"], rel=1e-9, abs=1e-12)
        # ideals + centre = the full projection onto 𝔤 (from the orthonormal DLA basis)
        assert sum(s["ideals"].values()) + s["center"] == pytest.approx(g["total"], rel=1e-9)


def test_schur_purities_match_dense_projection_sn(sn_decomposition_4):
    rng = np.random.default_rng(1)
    _, O, _ = default_setup("sn", 4)
    for X in (_random_hermitian(4, rng), _random_state(4, rng), neel_state_vector(4), O):
        g = g_purities(X, sn_decomposition_4)
        s = schur_purities(X, 4, "sn")
        assert sorted(g["ideals"]) == pytest.approx(sorted(s["ideals"].values()), rel=1e-9, abs=1e-12)
        assert sum(s["ideals"].values()) + s["center"] == pytest.approx(g["total"], rel=1e-9)


def test_hea_prediction_is_the_haar_formula():
    n = 3
    d = 2**n
    H = HeisenbergSpinChain(4).get_matrix()[:d, :d]                     # any Hermitian operator on 3 qubits
    psi = _random_state(n, np.random.default_rng(2))
    r = predict_loss("hea", n, psi, H)
    expected = (1 - 1 / d) * (np.trace(H @ H).real - np.trace(H).real ** 2 / d) / (d * d - 1)
    assert r["variance"] == pytest.approx(expected, rel=1e-12)
    h = HardwareEfficientAnsatz(n_qubits=n, n_layers=1)
    dec = reductive_decomposition(analyze_ansatz(h)["basis"], h.generators())
    pr, po = g_purities(psi, dec), g_purities(H, dec)
    assert pr["ideals"][0] * po["ideals"][0] / dec["ideal_dimensions"][0] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("n", [4, 6, 8])
def test_singlet_prediction_equals_haar_in_the_singlet_sector(n):
    """With the input in S = 0, the deep exchange circuit is Haar on SU(C_{N/2}) there: Var = Var_spec(H₀)/(C + 1)."""
    psi, H, _ = default_setup("su2", n)
    sb = SchurBasis(n)
    idx, V = sb.iso(0.0, 0.0)
    H0 = V.T @ (H[idx][:, idx] @ V)
    ev = np.linalg.eigvalsh(H0)
    C = len(ev)
    r = predict_loss("su2", n, psi, H, schur=sb)
    assert r["variance"] == pytest.approx(np.var(ev) / (C + 1), rel=1e-10)
    assert r["mean"] == pytest.approx(-3.0, abs=1e-10)                   # E[ℓ] = Tr[ρ_𝔷 O_𝔷] = −3 for every N
    assert r["obs_in_algebra"]


def test_singlet_prediction_from_sz_sectors_matches_schur_route():
    """Tr_{S=0} X = Tr_{S_z=0} X − Tr_{S_z=1} X for SU(2)-invariant X gives the same prediction without the Schur basis."""
    for n in (2, 4, 6, 8, 10):
        psi, H, _ = default_setup("su2", n)
        r = singlet_input_prediction(n)
        assert r["variance"] == pytest.approx(predict_loss("su2", n, psi, H)["variance"], rel=1e-12, abs=1e-14)
        assert r["mean"] == pytest.approx(-3.0, abs=1e-12)
        assert r["obs_scale"] == pytest.approx(np.real(H.multiply(H.conj()).sum()) / 2**n, rel=1e-12)
    ring = HeisenbergSpinChain(8, lattice_type="ring")
    psi = create_singlet_product_state_vector(8)
    assert singlet_input_prediction(8, ring.couplings)["variance"] == pytest.approx(
        predict_loss("su2", 8, psi, ring.get_sparse_matrix())["variance"], rel=1e-12)


def test_casimir_from_schur_blocks():
    n = 4
    C2 = get_casimir_operator(n)
    s = schur_purities(C2, n, "su2")
    C0 = C2 - np.trace(C2) / 16 * np.eye(16)
    assert sum(s["ideals"].values()) == pytest.approx(0.0, abs=1e-10)  # central: no semisimple part
    assert s["center"] == pytest.approx(np.trace(C0 @ C0).real, rel=1e-10)


# ----------------------------------------------------------------------------- #
# prediction against measurement
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,n", [("su2", 4), ("su2", 6), ("hea", 4), ("sn", 4), ("sn", 6)])
def test_prediction_matches_measured_variance(kind, n):
    r = compare_prediction(kind, n, depth=4 * n if kind != "sn" else 32, n_samples=1500, seed=11)
    assert r["obs_in_algebra"]
    assert abs(r["z"]) < 3.0
    assert abs(r["measured_mean"] - r["predicted_mean"]) < 4 * r["measured_mean_se"]


@pytest.mark.slow
def test_prediction_matches_measured_variance_eight_qubits_and_neel_input():
    for kind in ("su2", "hea", "sn"):
        assert abs(compare_prediction(kind, 8, depth=32, n_samples=1500, seed=5)["z"]) < 3.0
    # Néel input: weight in several spin sectors, so several ideals contribute
    n = 6
    psi, H = neel_state_vector(n), HeisenbergSpinChain(n).get_sparse_matrix()
    pr = predict_loss("su2", n, psi, H)
    assert sum(v > 1e-12 for v in pr["purities_rho"]["ideals"].values()) == 3
    assert abs(compare_prediction("su2", n, depth=24, n_samples=1500, seed=6, psi0=psi, obs=H)["z"]) < 3.0


def test_single_bond_observable_reaches_several_ideals():
    """O = σ_1·σ_2 on one bond: in i𝔤 (a generator), with weight in every ideal; singlet and Néel inputs."""
    n = 6
    G12 = sum(pauli_string_sparse({0: p, 1: p}, n) for p in "XYZ").tocsr()
    for psi, seed in ((create_singlet_product_state_vector(n), 21), (neel_state_vector(n), 22)):
        pr = predict_loss("su2", n, psi, G12)
        assert pr["obs_in_algebra"]
        assert sum(v > 1e-12 for v in pr["purities_obs"]["ideals"].values()) == 3       # S = 0, 1, 2 (S = 3 has m = 1)
        r = compare_prediction("su2", n, depth=24, n_samples=4000, seed=seed, psi0=psi, obs=G12)
        assert abs(r["z"]) < 3.0
        assert abs(r["measured_mean"] - r["predicted_mean"]) < 4 * r["measured_mean_se"]


def test_shallow_circuits_have_not_reached_the_deep_circuit_value():
    r = compare_prediction("su2", 6, depth=1, n_samples=1500, seed=3)
    assert r["z"] > 5.0                                                  # one layer is far from Haar on e^𝔤


@pytest.mark.parametrize("dist", ["laplace", "uniform"])
def test_variance_standard_error(dist):
    """The SE uses the fourth moment: check it on a heavy-tailed and a light-tailed distribution."""
    rng = np.random.default_rng(0)
    n, reps = 400, 600
    if dist == "laplace":                                                # scale 1: σ² = 2, μ₄ = 24 (kurtosis 6)
        draw, s2, m4 = (lambda: rng.laplace(size=n)), 2.0, 24.0
    else:                                                                # U(0, 1): σ² = 1/12, μ₄ = 1/80 (kurtosis 1.8)
        draw, s2, m4 = (lambda: rng.uniform(size=n)), 1 / 12, 1 / 80
    exact = np.sqrt((m4 - s2**2 * (n - 3) / (n - 1)) / n)
    gaussian = s2 * np.sqrt(2 / (n - 1))
    assert abs(exact / gaussian - 1) > 0.3                               # the Gaussian formula would fail here
    est = np.array([variance_with_error(draw()) for _ in range(reps)])
    assert np.mean(est[:, 0]) == pytest.approx(s2, rel=0.02)             # unbiased
    assert np.std(est[:, 0], ddof=1) == pytest.approx(exact, rel=0.1)
    assert np.mean(est[:, 1]) == pytest.approx(exact, rel=0.1)


# ----------------------------------------------------------------------------- #
# S_N-equivariant ansatz
# ----------------------------------------------------------------------------- #
def test_sn_dla_dimensions_match_formula():
    assert [permutation_dla_dimension_formula(n) for n in range(2, 9)] == [9, 19, 33, 54, 81, 117, 161]
    for n in range(2, 15):                                               # exact closure mod p
        res = permutation_dla_dimension_by_closure(n)
        assert res["closed"] and res["dla_dimension"] == permutation_dla_dimension_formula(n)
    for n in range(2, 17):                                               # floating point, generators scaled to unit norm
        res = permutation_dla_dimension_by_closure(n, method="float")
        assert res["closed"] and res["dla_dimension"] == permutation_dla_dimension_formula(n), n
    for n in (2, 3, 4):                                                  # dense closure on 4^N-dimensional operators
        res = DynamicalLieAlgebraAnalyzer(PermutationEquivariantAnsatz(n, 1).generators()).compute_dla_dimension()
        assert res["dla_dimension"] == permutation_dla_dimension_formula(n)


@pytest.mark.slow
def test_sn_dla_exact_closure_fifteen_and_sixteen_qubits():
    for n in (15, 16):
        assert permutation_dla_dimension_by_closure(n)["dla_dimension"] == permutation_dla_dimension_formula(n)


def test_modular_closure_finds_smaller_algebras_and_guards_the_bound(monkeypatch):
    sp_, sm, zz = permutation_integer_generators(6)
    assert _lie_closure_mod_p([sp_, sm], 1048573)[0] == 3                # S_+, S_− alone: one copy of sl(2)
    lin = [np.diag((b.shape[0] - 1 - 2 * np.arange(b.shape[0])).astype(float)) for b in zz]          # 2S_z
    assert _lie_closure_mod_p([sp_, sm, lin], 1048573)[0] == 3           # adding 2S_z changes nothing
    monkeypatch.setattr(dla_module, "_lie_closure_mod_p", lambda gens, p, **kw: (10**6, [10**6], True))
    with pytest.raises(ValueError):                                      # more than C(N+3, 3) directions is impossible
        permutation_dla_dimension_by_closure(6)


def test_spin_matrices_satisfy_su2():
    for S in (0.5, 1.0, 2.5):
        sx, sy, sz = spin_matrices(S)
        assert np.allclose(sx @ sy - sy @ sx, 1j * sz)
        assert np.allclose(sx @ sx + sy @ sy + sz @ sz, S * (S + 1) * np.eye(int(2 * S + 1)))


def test_sn_circuit_stays_permutation_invariant_when_any_generator_is_varied():
    n = 4
    a = PermutationEquivariantAnsatz(n_qubits=n, n_layers=2)
    rng = np.random.default_rng(4)
    base = rng.uniform(0, 2 * np.pi, a.num_params())
    perms = [permutation_matrix(p, n) for p in ([1, 0, 2, 3], [0, 2, 1, 3], [1, 2, 3, 0], [3, 2, 1, 0])]
    for k in range(a.num_params()):                                      # every angle, i.e. every generator
        for delta in (0.37, -1.9):
            p = base.copy()
            p[k] += delta
            U = a.unitary(p)
            for P in perms:
                assert np.linalg.norm(P @ U @ P.T - U) < 1e-12
    # the exchange circuit, by contrast, is not permutation invariant
    e = EquivariantQuantumAnsatz(n_qubits=n, n_layers=2, weight_sharing=False)
    U = e.unitary(rng.uniform(0, 2 * np.pi, e.num_params()))
    assert np.linalg.norm(perms[0] @ U @ perms[0].T - U) > 0.1


@pytest.mark.parametrize("lattice", ["1d_chain", "ring"])
def test_sn_circuit_never_changes_an_su2_invariant_energy(lattice):
    n = 6
    H = HeisenbergSpinChain(n, lattice_type=lattice).get_sparse_matrix()
    rng = np.random.default_rng(0)
    for psi0 in (create_singlet_product_state_vector(n), neel_state_vector(n), _random_state(n, rng)):
        th = rng.uniform(0, 2 * np.pi, (50, 3 * 6))
        e = expectation(run_circuit(permutation_equivariant_circuit(n, 6), n, th, psi0), H)
        assert np.max(np.abs(e - np.real(np.vdot(psi0, H @ psi0)))) < 1e-10
        # and the gradient is identically zero
        _, g = energy_and_gradient(permutation_equivariant_circuit(n, 3), n, th[0, :9], psi0, H)
        assert np.max(np.abs(g)) < 1e-10


def test_singlet_input_gives_zero_variance_for_sn_circuit():
    n = 6
    psi = create_singlet_product_state_vector(n)
    _, O, _ = default_setup("sn", n)
    assert predict_loss("sn", n, psi, O)["variance"] == pytest.approx(0.0, abs=1e-12)


def test_permutation_ansatz_pennylane_matches_numpy():
    qml = pytest.importorskip("pennylane")
    n = 4
    a = PermutationEquivariantAnsatz(n_qubits=n, n_layers=2)
    p = np.random.default_rng(5).uniform(0, 2 * np.pi, a.num_params())
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circ(x):
        a.apply_circuit(x)
        return qml.state()

    psi_np = a.unitary(p) @ a.initial_state_vector()
    assert abs(np.vdot(np.asarray(circ(p)), psi_np)) == pytest.approx(1.0, abs=1e-10)
    psi_sv = run_circuit(permutation_equivariant_circuit(n, 2), n, p[None, :], a.initial_state_vector())[0]
    assert abs(np.vdot(psi_sv, psi_np)) == pytest.approx(1.0, abs=1e-12)


# ----------------------------------------------------------------------------- #
# statevector simulator, g-sim, expressivity, CLI
# ----------------------------------------------------------------------------- #
def test_batched_statevector_matches_dense_unitaries_and_gradients():
    n = 4
    rng = np.random.default_rng(6)
    psi0 = create_singlet_product_state_vector(n)
    e = EquivariantQuantumAnsatz(n, 3, weight_sharing=False)
    p = rng.uniform(0, 2 * np.pi, e.num_params())
    assert np.allclose(run_circuit(su2_equivariant_circuit(n, 3), n, p[None], psi0)[0], e.unitary(p) @ psi0, atol=1e-12)
    h = HardwareEfficientAnsatz(n, 2)
    p = rng.uniform(0, 2 * np.pi, h.num_params())
    assert np.allclose(run_circuit(hardware_efficient_circuit(n, 2), n, p[None], psi0)[0], h.unitary(p) @ psi0, atol=1e-12)
    H = HeisenbergSpinChain(n).get_sparse_matrix()
    for gates in (su2_equivariant_circuit(n, 2), hardware_efficient_circuit(n, 1), permutation_equivariant_circuit(n, 2)):
        P = sum(1 for g in gates if g[0] != "cnot")
        x = rng.uniform(0, 2 * np.pi, P)
        _, g = energy_and_gradient(gates, n, x, psi0, H)
        fd = [(energy_and_gradient(gates, n, x + 1e-6 * np.eye(P)[k], psi0, H)[0]
               - energy_and_gradient(gates, n, x - 1e-6 * np.eye(P)[k], psi0, H)[0]) / 2e-6 for k in range(P)]
        assert np.allclose(g, fd, atol=1e-6)


@pytest.mark.parametrize("n", [4, 6, 12])
def test_sz_sector_simulator_matches_full_statevector(n):
    """The sector path (used for the exchange circuit at N = 14, 16) against the full statevector."""
    rng = np.random.default_rng(40 + n)
    psi, H, _ = default_setup("su2", n)
    sec = SzSector.containing(psi, n)
    assert sec.n_ones == n // 2 and sec.dim == math.comb(n, n // 2)
    gates = su2_equivariant_circuit(n, 4 * n)
    th = rng.uniform(0, 2 * np.pi, (3, num_params(gates)))
    full = run_circuit(gates, n, th, psi)
    assert np.abs(full - sec.embed(run_circuit_sector(gates, sec, th, sec.restrict(psi)))).max() < 1e-12
    # same seed, same angles: the sampled losses agree sample by sample
    a = sample_losses(gates, n, psi, H, 40, seed=3, batch=16)
    b = sample_losses(gates, n, psi, H, 40, seed=3, batch=16, sector=True)
    assert np.abs(a - b).max() < 1e-11
    if n == 4:
        m = [measure_loss_variance("su2", n, 16, 200, seed=8, sector=s)["variance"] for s in (False, True)]
        assert m[0] == pytest.approx(m[1], rel=1e-10)
    # the other S_z-conserving gates, on a random state with one qubit in |1⟩ and a random observable
    mixed = [("exch", (0, n - 1)), ("rz", (1,)), ("sumzz", ()), ("exch", (1, 2)), ("rz", (n - 1,))]
    sec1 = SzSector(n, 1)
    phi = rng.normal(size=sec1.dim) + 1j * rng.normal(size=sec1.dim)
    psi1 = sec1.embed(phi / np.linalg.norm(phi))
    th = rng.uniform(0, 2 * np.pi, (4, len(mixed)))
    full = run_circuit(mixed, n, th, psi1)
    part = run_circuit_sector(mixed, SzSector.containing(psi1, n), th, sec1.restrict(psi1))
    assert np.abs(full - sec1.embed(part)).max() < 1e-12
    if n <= 6:
        O = _random_hermitian(n, rng)
        assert np.allclose(expectation(full, O), expectation(part, sec1.block(O)), atol=1e-10)


def test_sz_sector_rejects_mixed_states_and_nonconserving_gates():
    n = 4
    sec = SzSector(n, 2)
    assert np.array_equal(sec.swap_index(0, 3)[sec.swap_index(0, 3)], np.arange(sec.dim))   # SWAP is an involution
    with pytest.raises(ValueError):
        SzSector.containing(_random_state(n, np.random.default_rng(0)), n)
    with pytest.raises(ValueError):
        run_circuit_sector([("ry", (0,))], sec, np.zeros((1, 1)), np.ones(sec.dim) / math.sqrt(sec.dim))
    psi, H, _ = default_setup("hea", n)
    with pytest.raises(ValueError):
        sample_losses(hardware_efficient_circuit(n, 1), n, psi, H, 4, seed=0, sector=True)


def test_gsim_matches_statevector_and_schur_prediction():
    from geometric_qml.gsim import permutation_equivariant_sequence, permutation_equivariant_simulator

    rng = np.random.default_rng(7)
    for n in (6, 8, 14):                                                 # N = 14: the first size of the g-sim points
        sim = permutation_equivariant_simulator(n)
        th = rng.uniform(0, 2 * np.pi, (4, 3 * 5))
        psi, O, _ = default_setup("sn", n)
        sv = expectation(run_circuit(permutation_equivariant_circuit(n, 5), n, th, psi), O)
        assert np.allclose(sim.expectation(permutation_equivariant_sequence(5), th), sv, atol=1e-11)
        if n < 14:
            assert sim.predicted_variance() == pytest.approx(predict_loss("sn", n, psi, O)["variance"], rel=1e-10)


@pytest.mark.slow
def test_gsim_prediction_matches_schur_route_at_fourteen_qubits():
    """At N ≥ 14 the figures use g-sim for both points and line; check the line against the Schur route once."""
    from geometric_qml.gsim import permutation_equivariant_simulator

    psi, O, _ = default_setup("sn", 14)
    assert permutation_equivariant_simulator(14, dynamics=False).predicted_variance() == pytest.approx(
        predict_loss("sn", 14, psi, O)["variance"], rel=1e-10)


@pytest.mark.slow
def test_expressivity_at_fixed_budget_small_chain():
    from geometric_qml.expressivity import expressivity_comparison

    r = expressivity_comparison(n_qubits=4, lattice="1d_chain", budget=24, n_restarts=2, seed=0, maxiter=500)
    assert r["su2"]["error"] < 1e-8 and r["su2"]["fidelity"] > 1 - 1e-8
    assert r["sn"]["error"] == pytest.approx(r["initial_energy"] - r["exact_energy"], abs=1e-9)   # never moves
    assert r["hea"]["n_params"] <= 24 and r["su2"]["n_params"] <= 24 and r["sn"]["n_params"] <= 24


def test_cli_variance_theory(capsys):
    assert cli_main(["variance-theory", "--ansatz", "su2", "--qubits", "4", "--samples", "400", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["predicted"] == pytest.approx(4.0) and out["dla_dimension"] == 12 and abs(out["z"]) < 4
    assert cli_main(["dla", "--ansatz", "sn", "--qubits", "6", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dla_dimension"] == out["closed_form"] == 81
    # two qubits: the loss is constant, so both variances are round-off and z is reported as 0
    assert cli_main(["variance-theory", "--ansatz", "su2", "--qubits", "2", "--samples", "100", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["zero_variance"] and out["z"] == 0.0
    for kind in ("hea", "su2", "sn"):                                    # the default inputs need an even N
        with pytest.raises(SystemExit):
            cli_main(["variance-theory", "--ansatz", kind, "--qubits", "5"])
