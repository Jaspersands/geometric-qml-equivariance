# Changelog

## 0.5.0 — 2026-10-19

The gradient-variance study of 0.3 and 0.4 fitted slopes but never compared them with a theory. This release predicts the deep-circuit loss variance from the dynamical Lie algebra and checks it against sampling, and adds a permutation-equivariant ansatz to show the trade-off between trainability and expressivity.

### Added
- `variance_theory`: the Lie-algebraic variance formula of Ragone et al. (2024), Var[ℓ] = Σ_j P_j(ρ) P_j(O) / dim 𝔤_j, and the mean from the centre. Purities come either from any dense DLA basis (`reductive_decomposition` finds the centre and the simple ideals) or from the Schur–Weyl blocks (`SchurBasis`, `schur_purities`), which reach N = 12. Measured variances with fourth-moment standard errors, depth scans and a χ² summary (`compare_prediction`, `variance_study`, `depth_scan`). `singlet_input_prediction` gives the exchange-circuit prediction for the singlet input from S_z-sector traces, Tr_{S=0} X = Tr_{S_z=0} X − Tr_{S_z=1} X, up to N = 24.
- `PermutationEquivariantAnsatz` (generators ΣX, ΣY, Σ ZZ, after Schatzki et al. 2024), its DLA on the spin blocks (`permutation_reduced_generators`, `permutation_dla_dimension_by_closure`) and the closed form C(N+3, 3) − ⌊N/2⌋. The closure runs in exact integer arithmetic modulo two primes (`_lie_closure_mod_p`, `permutation_integer_generators`) and matches the formula for every N from 2 to 24 and at N = 28; the tests run N = 2 to 16.
- `statevector`: NumPy simulation of the three circuit families, batched over parameter samples, with adjoint gradients. No PennyLane needed. Circuits of S_z-conserving gates (exchange, RZ, Σ ZZ) can also run inside one S_z sector (`SzSector`, `run_circuit_sector`, `sample_losses(..., sector=True)`): 12 870 amplitudes at N = 16 for the singlet product input instead of 65 536. The tests check it against the full statevector at N = 4, 6 and 12 (states to 1e-12, sampled losses to 1e-11, with the same angles).
- `gsim`: simulation in the adjoint representation (Goh et al.) for circuits whose observable lies in the DLA; for the S_N circuit it agrees with the statevector to within 5e-13 at N = 6, 8 and 14 with 32 layers (the tests require 1e-11) and runs at N = 24 (dimension 2 913).
- `expressivity`: ground-state searches for the Heisenberg chain and ring at a fixed parameter budget.
- `benchmarks/make_figures.py` and `figures/`: prediction against measurement, depth convergence, expressivity, and the combined trade-off figure; `figures/results.json` holds the numbers. `--extend` measures the exchange and hardware-efficient circuits at N = 14 and 16 with the same protocol, and the exchange circuit again at L = 16N.
- CLI: `geometric-qml variance-theory`, and `geometric-qml dla --ansatz sn`.

### Findings
- Measured and predicted loss variance agree within sampling error for the hardware-efficient, SU(2)-equivariant and S_N-equivariant circuits at N = 4 to 12, for the first two also at N = 14 and 16, and for the S_N circuit to N = 24 with g-sim (χ² = 15.6 over 23 points). In depth scans at N = 6, 8 and 10 the first depth within two standard errors of the prediction is at most 3.2N layers for the first two circuits and 8 layers for the third.
- Up to N = 12 the exchange circuit has the larger normalised variance, each circuit on its own loss. At N = 14 and 16 the S_N circuit's is larger by factors of 2.1 and 5.9, measured for both circuits, so the crossover lies between 12 and 14 qubits as the formula predicts.
- At N = 16 the exchange circuit's mean loss at L = 4N is −3.023 ± 0.004, 5.5 standard errors from the deep-circuit value −3; at L = 16N it is −3.001 ± 0.004. The variance agrees with the prediction at both depths.
- An S_N-equivariant circuit never changes the expectation value of an SU(2)-invariant Hamiltonian (chain, ring or any other graph), from any input state: its generators ΣX = 2S_x, ΣY = 2S_y and Σ ZZ = 2S_z² − N/2 commute with it. Its gradient is identically zero.

### Changed
- README leads with the variance result and the trade-off; the fitted-slope claim is replaced by the comparison with theory.
- `DynamicalLieAlgebraAnalyzer` scales block-diagonal generators to unit operator norm. The span is unchanged; without it, round-off in the S_N closure grew with the generator norm (about N²/2 for Σ ZZ) and the floating-point count was wrong from N = 13. With it the count is right up to N = 16, and `permutation_dla_dimension_by_closure` raises an error for any count above C(N+3, 3).
- `variance-theory` in the CLI rejects odd N for every circuit (all default inputs need an even N) and reports z = 0 when prediction and measurement are both zero (N = 2), instead of a ratio of round-off errors.
- `benchmarks/make_figures.py` keeps keys it does not produce (such as `su2_recheck`) when it rewrites `results.json`, and has a `--lines` option.
- Tests: from 61 to 100; a `slow` marker is registered.

## 0.4.0 — 2026-09-23

Changes from an adversarial review of 0.3.

### Fixed
- `kagome` was a chain of triangles joined by single bonds. It is now the kagome lattice on `shape=(Lx, Ly)` unit cells (`kagome_edges`, `kagome_positions`); on the torus every site has coordination 4 and lies in two triangles, and the 12-site cluster reproduces E₀/N = −0.45374.
- Fermi–Hubbard operators are sparse (a dense L = 8 build needed about 70 GB per operator); ground energies use Lanczos in fixed (N, S_z) sectors. `get_matrix` refuses dense output beyond 14 qubits.

### Added
- Lie algebra closure that commutes generators with the newest elements only, batched with pivoted QR, plus an SU(2)-reduced representation (`su2_reduced_generators`, `equivariant_dla_dimension_by_closure`) that verifies the closed form at N = 8 (1426).
- `spin_multiplicities` in closed form (replaces a dense Casimir diagonalisation).
- Sparse Heisenberg models (`get_sparse_matrix`, `pauli_string_sparse`) with Lanczos ground states beyond 10 qubits.
- `QuantumGraphNeuralNetwork(trotter_steps=r)`: second-order Trotterisation into native exchange gates (IsingXX·IsingYY·IsingZZ), exact SU(2) equivariance, measured permutation error (`equivariance_error`, `su2_equivariance_error`); optional distance-2 exchange terms.

### Changed
- Tests: 49 → 61.

## 0.3.0 — 2026-09-21

### Fixed (correctness)
- `geometric-qml dla` now runs the Lie-algebra closure (v0.2 printed a Catalan number).
- Sycamore placement is a snake; v0.2 placed qubits in rows of four so chain bonds (3,4) were not adjacent while the README claimed planar routing. Non-adjacent couplings now raise.
- `triangular` / `kagome` lattices are real geometries (v0.2 built the same NN+NNN chain for both).
- QGNN layers are single exponentials of the graph Hamiltonian; the v0.2 product of bond gates was not permutation-equivariant (bond gates do not commute).
- Fermi–Hubbard on-site term includes its constant; matrix verified against PennyLane.
- VQE is seeded.
- README claim of a polynomial EQNN DLA withdrawn; replaced by the verified closed form Σ_S (m_S² − 1) + 1.
- Barren-plateau study uses depth L = N and per-bond normalisation; fixed shallow depth showed no plateau for any ansatz and N = 2 is degenerate for the EQNN.

### Added
- `lattice_edges`, `total_spin_sectors`, `create_singlet_product_state_vector`, `pauli_string_matrix`, PennyLane-free `get_matrix`.
- `generators()` and dense `unitary()` on every ansatz.
- DLA analyzer rewrite (traceless generators, incremental closure, `closed` flag, basis returned) and `equivariant_dla_dimension_formula`.
- `fit_variance_scaling`, multi-ansatz study with fits and raw samples, `singlet_purity`.
- Fermi–Hubbard spin and η-pairing charge SU(2) generators, number operator, particle-number-sector ground energy, particle-hole-symmetric form.
- `permutation_matrix`, `is_permutation_equivariant`.
- `PhasedFSimGate` output, `check_adjacency`, `gate_counts`.
- OpenFermion / Qiskit string exporters.
- CLI: `sectors`, `vqe`, `benchmark`, `--json`, `--lattice`.
- 49 tests (from 17); browser demo with a real statevector simulator.

## 0.2.0
- DLA analyzer, Fermi–Hubbard chain, QGNN, OpenFermion exporter, CLI, CI, notebook.

## 0.1.0
- Initial release.
