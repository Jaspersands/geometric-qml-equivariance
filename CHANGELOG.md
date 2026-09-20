# Changelog

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
