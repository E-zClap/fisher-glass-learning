"""Refit the auxiliary logistic predictors from recovered donor rows; no neural training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))

import numpy as np
import scipy
import sklearn
from predictive_validity import analyse_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "refit_verification.json")
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()
    archived_payload = (HERE / "predictive_validity.json").read_bytes()
    expected = json.loads(archived_payload)
    provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    if hashlib.sha256(archived_payload).hexdigest() != provenance["result_sha256"]:
        raise ValueError("Original result JSON hash mismatch")
    for relative, digest in provenance["recovered_files_sha256"].items():
        if hashlib.sha256((HERE / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input/source hash mismatch: {relative}")
    settings = SimpleNamespace(folds=expected["args"]["folds"], seed=expected["args"]["seed"])
    results, differences = [], []
    for archived in expected["runs"]:
        result = analyse_run(archived["tag"], HERE / "inputs", settings, selected_gid=3)
        if result is None:
            raise ValueError(f"No results for {archived['tag']}")
        results.append(result)
        for variant, metrics in archived["variants"].items():
            for metric, value in metrics.items():
                actual = result["variants"][variant][metric]
                difference = abs(actual - value)
                differences.append((difference, archived["tag"], variant, metric))
                if not math.isclose(actual, value, rel_tol=1e-8, abs_tol=args.atol):
                    raise ValueError(f"Archived result mismatch: {archived['tag']} / {variant} / {metric}: {actual} != {value}")
    report = {
        "scope": "Refit only auxiliary flip predictors from recovered original donor rows; no neural training or image scoring",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "absolute_tolerance": args.atol,
        "relative_tolerance": 1e-8,
        "all_archived_metrics_matched": True,
        "largest_absolute_difference": max(differences),
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified all 15 run/variant outputs against the archive; maximum absolute difference {max(differences)[0]:.3g}.")


if __name__ == "__main__":
    main()
