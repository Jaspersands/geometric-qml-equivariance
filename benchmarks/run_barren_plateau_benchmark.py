"""python benchmarks/run_barren_plateau_benchmark.py [--json] [--quick]  (== geometric-qml benchmark)"""

import sys

from geometric_qml.benchmark import run_benchmark

if __name__ == "__main__":
    sys.exit(run_benchmark(as_json="--json" in sys.argv, quick="--quick" in sys.argv))
