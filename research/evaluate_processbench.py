"""Run the fixed PONR heuristic against every official ProcessBench split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import PROCESSBENCH_SPLITS, evaluate_processbench  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("learned", "heuristic", "both"), default="both")
    args = parser.parse_args()
    detectors = ("learned", "heuristic") if args.detector == "both" else (args.detector,)
    results = [evaluate_processbench(split, detector) for detector in detectors
               for split in sorted(PROCESSBENCH_SPLITS)]
    print(json.dumps({"test_policy": "problem-grouped held-out test only; thresholds selected on validation",
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
