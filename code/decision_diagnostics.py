"""Summarize the truncation, checked proposals, and same-probe comparisons."""
import json

import numpy as np
import pandas as pd

from replay import KEY, CELL, RESULTS, write_json


def main():
    walsh = pd.read_csv(RESULTS / "walsh_audit.csv", dtype={"query_id": str})
    data = pd.read_csv(RESULTS / "m8_decisions.csv", dtype={"query_id": str})
    overall = pd.read_csv(RESULTS / "selection_overall.csv")
    audit = {}
    for b in [3, 4]:
        selected = walsh[walsh.budget == b]
        assert (selected.legacy_p5.astype(int) & 128 == 0).all()
        assert (selected.swapped_legacy_p5.astype(int) & 1 == 0).all()
        by_position = {method: [int(((selected[method].astype(int) & (1 << p)) != 0).sum()) for p in range(8)]
                       for method in ["p1", "legacy_p5", "swapped_legacy_p5"]}
        wide = data[data.budget == b].pivot(index=KEY, columns="method", values=["normalized_regret", "selected_mask", "probe_count"])
        cell = wide.groupby(level=CELL).mean()["normalized_regret"]
        p1_minus_bo = wide["normalized_regret", "P1"] - wide["normalized_regret", "BO9"]
        check_improvement = wide["normalized_regret", "BO9"] - wide["normalized_regret", "P1-check"]
        adapt_vs_bo = wide["normalized_regret", "Adaptive"] - wide["normalized_regret", "BO-adaptive"]
        adapt = overall[(overall.budget == b) & (overall.method == "Adaptive")].iloc[0]
        audit[f"b{b}"] = dict(n=len(selected), walsh_exact_error=selected.full_walsh_max_error.max(),
                             walsh_ls_error=selected.ls_max_error.max(), candidate_inclusions=by_position,
                             permutation_changes_selection=int((selected.legacy_p5 != selected.swapped_legacy_p5).sum()),
                             p1_vs_bo=dict(better=int((p1_minus_bo < 0).sum()), worse=int((p1_minus_bo > 0).sum()),
                                           tied=int((p1_minus_bo == 0).sum())),
                             adapt_vs_bo=dict(better=int((adapt_vs_bo < 0).sum()), worse=int((adapt_vs_bo > 0).sum()),
                                              tied=int((adapt_vs_bo == 0).sum())),
                             one_check_new_count=int((wide["probe_count", "P1-check"] == 10).sum()),
                             one_check_strict_gain_count=int((check_improvement > 0).sum()),
                             one_check_material_gain_count=int((check_improvement >= .05).sum()),
                             p1_better_than_allpairs_cells=int((cell.P1 < cell["All-pairs"]).sum()),
                             p1_better_than_source_and_p5_cells=int(((cell.P1 < cell["Fixed source"]) & (cell.P1 < cell["Truncated Walsh"])).sum()),
                             adaptive_better_than_p1_cells=int((cell.Adaptive < cell.P1).sum()),
                             adaptive_better_than_both_same_cost_controls=int(((cell.Adaptive < cell["Random pairs"]) & (cell.Adaptive < cell["Hashed subsets"])).sum()),
                             adaptive_actual_max=int(adapt.max_probes), adaptive_combinatorial_upper_bound=21)
    write_json(RESULTS / "decision_diagnostics.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
