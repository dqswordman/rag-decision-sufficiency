"""Verify the release inputs and regenerate the numerical paper artifacts on CPU."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import platform
import time


def verify_inputs(root):
    manifest = root / "data/SHA256SUMS"
    lines = manifest.read_text(encoding="ascii").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}")
    print(f"Verified {len(lines)} archived inputs and reference files.", flush=True)
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-inputs", action="store_true", help="Check archived file integrity and exit.")
    args = parser.parse_args()
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    count = verify_inputs(root)
    if args.verify_inputs:
        return

    # The small matrix operations in the replay run faster with one BLAS thread.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    import pandas as pd
    import calibration_audit
    import decision_diagnostics
    import extension_replay
    import generation_metrics
    import paper_tables
    import replay
    import scale_diagnostics

    for name in ("results", "source", "figures"):
        (root / name).mkdir(exist_ok=True)
    report_path = replay.RESULTS / "reproduction_report.json"
    report_path.unlink(missing_ok=True)
    calibration_audit.main()
    replay.replay()
    selections = pd.read_csv(replay.RESULTS / "m8_decisions.csv", dtype={"query_id": str}, float_precision="round_trip")
    intervals = replay.paired(selections, ["normalized_regret", "raw_regret"], ["budget", *replay.CELL])
    assert (intervals.n == 60).all() and (intervals.clusters == 60).all()
    intervals.to_csv(replay.RESULTS / "selection_paired_by_cell.csv", index=False)
    decision_diagnostics.main()
    generation_metrics.main()
    extension_replay.main()
    scale_diagnostics.main()
    paper_tables.materialize_tables()
    paper_tables.plot_regret_distribution()

    reference_csvs = sorted((root / "reference/results").glob("*.csv"))
    for reference in reference_csvs:
        expected = pd.read_csv(reference, float_precision="round_trip")
        actual = pd.read_csv(replay.RESULTS / reference.name, float_precision="round_trip")
        try:
            pd.testing.assert_frame_equal(actual, expected, atol=1e-12, rtol=0, check_exact=False)
        except AssertionError as error:
            raise AssertionError(f"Reference mismatch in {reference.name}\n{error}") from error
    reference_tables = sorted((root / "reference/tables").glob("*.tex"))
    for reference in reference_tables:
        actual = (root / "source" / reference.name).read_text(encoding="ascii")
        if actual != reference.read_text(encoding="ascii"):
            raise AssertionError(f"Paper table differs from its archived rows: {reference.name}")

    report = {"status": "pass", "device": "CPU", "inputs_verified": count,
              "reference_csvs_matched": [file.name for file in reference_csvs],
              "tex_tables_matched": [file.name for file in reference_tables],
              "csv_absolute_tolerance": 1e-12,
              "regret_plot": "figures/regret_distribution.pdf", "seconds": round(time.perf_counter() - start, 2),
              "python": platform.python_version(),
              "dependencies": {name: version(name) for name in ("numpy", "pandas", "PyYAML", "matplotlib")}}
    replay.write_json(report_path, report)
    print(f"Matched {len(reference_csvs)} reference CSVs and {len(reference_tables)} paper tables in {report['seconds']} s.")
    print(f"Verification report: {report_path.relative_to(root)}")


if __name__ == "__main__":
    main()
