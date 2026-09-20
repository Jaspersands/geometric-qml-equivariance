# Contributing to geometric-qml-equivariance

Thank you for contributing to **geometric-qml-equivariance**!

## Development Setup

```bash
git clone https://github.com/Jaspersands/geometric-qml-equivariance.git
cd geometric-qml-equivariance
pip install -e ".[dev]"
pytest -v tests/
```

## Pull Request Guidelines
- All unit tests must pass.
- Preserve symmetry theorems and Lie algebraic proofs.

## Conventions
- Qubit 0 is the most significant bit in dense matrices (matches PennyLane's `qml.matrix` wire order).
- Every stochastic routine takes a `seed`.
- New physics claims need a test that checks the number (e.g. DLA dimensions against the closed form).
- Run `pytest -q tests/` and `python benchmarks/run_barren_plateau_benchmark.py --quick` before opening a PR.
