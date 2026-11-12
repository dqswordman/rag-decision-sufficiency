"""Paired pool-width changes and feasible-domain geometry for the extension."""
from itertools import combinations

import numpy as np
import pandas as pd

from replay import DATA, KEY, RESULTS, cluster_ci, scales


def main():
    changes = []
    for design, name, pairs, metrics in [
        ("historical_prefix", "nested_decisions.csv", [(6, 4), (8, 6), (8, 4)], ["normalized_regret", "probe_count"]),
        ("fresh_extension", "extension_decisions.csv", [(12, 8)], ["normalized_regret", "probe_count", "token_f1", "exact_match"]),
    ]:
        rows = pd.read_csv(RESULTS / name, dtype={"query_id": str})
        for method, data in rows.groupby("method"):
            scores = data.pivot(index=KEY, columns="m", values=metrics)
            for wider, narrower in pairs:
                for metric in metrics:
                    delta = (scores[metric, wider] - scores[metric, narrower]).rename("delta").reset_index()
                    assert delta.delta.notna().all()
                    changes.append(dict(design=design, method=method, wider=wider, narrower=narrower,
                                        metric=metric, **cluster_ci(delta, "delta")))
    pd.DataFrame(changes).to_csv(RESULTS / "pool_width_changes.csv", index=False)

    table = scales().query("dataset_key == 'musique' and universe_name == 'B_m8'").set_index("model_key")
    raw = pd.read_csv(DATA / "extension_raw_mean_logprobs.csv", dtype={"query_id": str}, float_precision="round_trip")
    geometry = []
    for (model, qid), frame in raw.groupby(["model_key", "query_id"]):
        assert len(frame) == 299
        epsilon = table.loc[model, "epsilon"]
        for m in (8, 12):
            u = dict(zip(frame["mask"], frame["mean_logprob"]))
            marginals, second = [], []
            for base in (mask for mask in u if mask < 1 << m):
                flags = [1 << bit for bit in range(m) if not base & (1 << bit)]
                size = bin(base).count("1")
                if size <= 2:
                    marginals.extend(u[base | bit] - u[base] for bit in flags)
                if size <= 1:
                    second.extend(u[base | i | j] - u[base | i] - u[base | j] + u[base] for i, j in combinations(flags, 2))
            geometry.append(dict(dataset_key="musique", query_id=qid, model_key=model, m=m,
                                 material_negative_marginal_rate=(np.asarray(marginals) < -epsilon).mean(),
                                 material_positive_hessian_rate=(np.asarray(second) > epsilon).mean()))
    rows = pd.DataFrame(geometry)
    assert len(rows) == 80
    rows.to_csv(RESULTS / "extension_geometry_rows.csv", index=False)
    output = []
    for m, frame in rows.groupby("m"):
        for metric in ("material_negative_marginal_rate", "material_positive_hessian_rate"):
            output.append(dict(m=m, metric=metric, **cluster_ci(frame, metric)))
    pd.DataFrame(output).to_csv(RESULTS / "extension_geometry_summary.csv", index=False)


if __name__ == "__main__":
    main()
