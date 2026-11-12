"""Score observed-only policies against frozen utility arrays, then write paired results."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
sys.path.insert(0, str(ROOT / "frozen"))
import historical_baselines as frozen

PROTOCOL = json.loads((DATA / "revision_protocol.json").read_text(encoding="utf-8"))
KEY = ["dataset_key", "query_id", "universe_name", "model_key"]
CELL = ["dataset_key", "universe_name", "model_key"]
CLUSTER = KEY[:2]
GEN_ROLES = PROTOCOL["generation"]["roles"]
CONTRASTS = [
    ("P1", "BO9"), ("Adaptive", "BO-adaptive"), ("P1-check", "BO9"),
    ("P1-check", "P1"), ("P1", "All-pairs"), ("P1", "Source ensemble"),
    ("Adaptive", "P1"), ("P1-check", "Adaptive"), ("Adaptive-check", "Adaptive"),
    ("Adaptive", "Random pairs"), ("Adaptive", "Hashed subsets"),
    ("P1", "Fixed source"), ("P1", "Truncated Walsh"), ("P1-check", "All-pairs"),
    ("BO-adaptive", "BO9"), ("BO-adaptive", "P1-check"), ("Adaptive-check", "BO-adaptive"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scales():
    path = ROOT / "frozen" / "protocol" / "calibration_scales.yaml"
    assert digest(path) == "b14d187bc1c9673a34760a1f5d86bb356f7bd2ddee66bbeefe29deb9594a427e"
    old = yaml.safe_load(path.read_text(encoding="utf-8"))["cells"]
    current = yaml.safe_load((path.parent / "v4_confirmation_protocol.yaml").read_text(encoding="utf-8"))
    trigger = current["methods"]["B_contender_k4_q50"]["trigger"]
    assert float(trigger.split(">=")[1]) == PROTOCOL["adaptive"]["risk_threshold"]
    rows = [dict(dataset_key=r["dataset_key"], model_key=r["model_key"],
                 universe_name=r["universe_name"].replace("m6", "m8"),
                 scale=r["robust_iqr_scale"], epsilon=r["epsilon_material"],
                 origin="R02 calibration IQR") for r in old]
    for r in current["normalization"]["phi_scales"]:
        donor = next(d for d in rows if d["dataset_key"] == r["dataset_key"] and d["universe_name"] == r["track"]
                     and d["model_key"] == "qwen")
        assert (r["scale"], r["epsilon"]) == (donor["scale"], donor["epsilon"])
        rows.append(dict(donor, model_key="phi", origin="V4 fixed values, numerically identical to Qwen"))
    assert len(rows) == 24 and all(np.isclose(r["epsilon"], .05 * r["scale"]) for r in rows)
    return pd.DataFrame(rows).sort_values(CELL)


def design(m, budget):
    masks = np.array([sum(1 << bit for bit in items) for n in range(budget + 1)
                      for items in combinations(range(m), n)], dtype=np.int64)
    masks.sort()
    singleton = np.array([1 << i for i in range(m)], dtype=np.int64)
    pairs = np.array([sum(pair) for pair in combinations(singleton.tolist(), 2)], dtype=np.int64)
    incidence = ((masks[:, None] & singleton[None, :]) != 0).astype(float)
    pair_incidence = (masks[:, None] & pairs[None, :]) == pairs[None, :]
    return masks, singleton, pairs, incidence, pair_incidence


def best(observed, budget):
    return min((mask for mask in observed if bin(mask).count("1") <= budget),
               key=lambda mask: (-observed[mask], mask))


def pick(scores, masks):
    # design() sorts masks, so argmax implements the exact tie rule.
    return int(masks[int(np.argmax(scores))])


def base_policies(read, m, budget, risk, threshold=None):
    """The read callable exposes only requested scores. Evaluation is external."""
    masks, singletons, all_pairs, incidence, pair_incidence = design(m, budget)
    base = {int(mask): read(int(mask)) for mask in [0, *singletons]}
    singleton_values = np.array([base[mask] - base[0] for mask in singletons])
    additive = base[0] + incidence @ singleton_values
    p1 = pick(additive, masks)
    checked = dict(base)
    if p1 not in checked:
        checked[p1] = read(p1)
    if threshold is None:
        threshold = PROTOCOL["adaptive"]["risk_threshold"]
    triggered = risk >= threshold
    # The adaptive contender search uses b=3 even for the b=4 robustness analysis.
    contender_masks, _, _, contender_incidence, _ = design(m, 3)
    contender_scores = base[0] + contender_incidence @ singleton_values
    order = np.lexsort((contender_masks, -contender_scores))[:PROTOCOL["adaptive"]["contender_count"]]
    contenders = contender_masks[order]
    adaptive_pairs = [int(pair) for pair in all_pairs if triggered and np.any((contenders & pair) == pair)]

    def quadratic(probes):
        observed = dict(base)
        coefficients = np.zeros(len(all_pairs))
        for idx, pair in enumerate(all_pairs):
            pair = int(pair)
            if pair in probes:
                observed[pair] = read(pair)
                left = pair & -pair
                coefficients[idx] = observed[pair] - base[left] - base[pair ^ left] + base[0]
        return pick(additive + pair_incidence @ coefficients, masks), observed

    adaptive, measured = quadratic(set(adaptive_pairs))
    adaptive_checked = dict(measured)
    if adaptive not in adaptive_checked:
        adaptive_checked[adaptive] = read(adaptive)
    all_pair, complete = quadratic(set(all_pairs))
    choices = {
        "BO9": (best(base, budget), base), "P1": (p1, base),
        "P1-check": (best(checked, budget), checked), "BO-adaptive": (best(measured, budget), measured),
        "Adaptive": (adaptive, measured), "Adaptive-check": (best(adaptive_checked, budget), adaptive_checked),
        "All-pairs": (all_pair, complete),
    }
    return choices, adaptive_pairs, triggered, quadratic


@lru_cache(maxsize=None)
def stencil(budget):
    marginals, hessians = [], []
    for base in range(256):
        available = [1 << bit for bit in range(8) if not base & (1 << bit)]
        size = bin(base).count("1")
        if size <= budget - 1:
            marginals.extend((base | bit, base) for bit in available)
        if size <= budget - 2:
            hessians.extend((base | i | j, base | i, base | j, base) for i, j in combinations(available, 2))
    return np.asarray(marginals).T, np.asarray(hessians).T


def geometries(values, epsilon, scale, budget):
    (hi, low), (ab, a, b, base) = stencil(budget)
    marginal = values[hi] - values[low]
    hessian = values[ab] - values[a] - values[b] + values[base]
    return dict(marginal_count=len(marginal), hessian_count=len(hessian),
                negative_marginal_rate=(marginal < 0).mean(), positive_hessian_rate=(hessian > 0).mean(),
                material_negative_marginal_rate=(marginal < -epsilon).mean(),
                material_positive_hessian_rate=(hessian > epsilon).mean(),
                normalized_negative_mass=np.maximum(-marginal, 0).sum() / scale,
                normalized_positive_hessian_mass=np.maximum(hessian, 0).sum() / scale)


def walsh_audit(values, historical, budget):
    masks, singleton, _, _, _ = design(8, budget)
    terms = [0, *singleton]
    differences = values[singleton] - values[0]
    coefficients = np.array([values[0] + differences.sum() / 2, *(-differences / 2)])
    design_matrix = frozen.WALSH[:, terms]
    fitted = design_matrix @ coefficients
    fitted_ls = design_matrix @ np.linalg.lstsq(design_matrix[terms], values[terms], rcond=None)[0]
    additive = frozen.degree_projection(values, 1)
    np.testing.assert_allclose(fitted, additive, rtol=0, atol=1e-12)
    assert pick(fitted[masks], masks) == pick(additive[masks], masks)
    assert np.array_equal(historical[:128], historical[128:])
    # Swap labels 1 and 8 in the indexing alone. Values still belong to the original physical contexts.
    permutation = np.array([(mask & 126) | ((mask & 1) << 7) | ((mask & 128) >> 7) for mask in range(256)])
    swapped = frozen.hierarchical_omp_prediction(values[permutation], np.array(terms))[permutation]
    assert np.array_equal(swapped[::2], swapped[1::2])
    return (float(np.max(np.abs(fitted - additive))), float(np.max(np.abs(fitted_ls - additive))),
            pick(historical[masks], masks), pick(swapped[masks], masks))


def row_metrics(base, values, mask, probes, oracle, scale, epsilon, budget):
    gap = float(values[oracle] - values[mask])
    assert gap >= 0 and bin(mask).count("1") <= budget
    return dict(**base, budget=budget, selected_mask=mask, selected_size=bin(mask).count("1"),
                probe_count=len(probes), probe_masks=" ".join(map(str, sorted(probes))),
                selected_utility=float(values[mask]), oracle_mask=oracle, oracle_utility=float(values[oracle]),
                raw_regret=gap, normalized_regret=gap / scale, epsilon_optimal=int(gap <= epsilon + 1e-12),
                exact_set_match=int(mask == oracle))


def replay():
    raw = pd.read_csv(DATA / "m8_raw_mean_logprobs.csv", dtype={"query_id": str}, float_precision="round_trip")
    assert len(raw) == 1440 and not raw.duplicated(KEY).any()
    scale_table = scales()
    scale_table.to_csv(RESULTS / "scales.csv", index=False)
    raw = raw.merge(scale_table, on=CELL, validate="many_to_one")
    records, geometry, nested, audit, risks = [], [], [], [], []
    cols = [f"mask_{mask}" for mask in range(256)]
    for identity, group in raw.groupby(KEY[:3], sort=True):
        assert group.universe_hash.nunique() == 1 and len(group) == 4
        centered = {row.model_key: values - values[0]
                    for row, values in zip(group.itertuples(), group[cols].to_numpy())}
        per_model_scale = dict(zip(group.model_key, group.scale))
        for row, raw_values in zip(group.itertuples(), group[cols].to_numpy()):
            model = row.model_key
            key = (*identity, model)
            base = dict(zip(KEY, key))
            values, scale, epsilon = centered[model], row.scale, row.epsilon
            assert np.isfinite(values).all()
            risk = -raw_values[0]
            historical_source = centered[frozen.SOURCE_MAPPING[model]]
            ensemble = np.mean([centered[g] / per_model_scale[g] for g in sorted(centered) if g != model], axis=0)
            probes9 = frozen.generic_probe_masks(9, key)
            old = frozen.hierarchical_omp_prediction(values, probes9)
            for budget in (3, 4):
                masks, _, _, _, _ = design(8, budget)
                oracle = pick(values[masks], masks)
                read = lambda mask: float(values[mask])
                choices, pairs, trigger, quadratic = base_policies(read, 8, budget, risk)
                for left, right in [("P1-check", "P1"), ("P1-check", "BO9"), ("Adaptive-check", "Adaptive")]:
                    assert values[choices[left][0]] >= values[choices[right][0]]
                if budget == 3:
                    risks.append(dict(**base, no_context_nll=risk, trigger=int(trigger), pair_count=len(pairs),
                                      changes_selection=int(choices["Adaptive"][0] != choices["P1"][0])))
                original_pairs = [(i, j) for i, j in combinations(range(8), 2)]
                hashed = sorted(original_pairs, key=lambda pair: hashlib.sha256(
                    f"{'|'.join(key)}|{pair}|RANDOM".encode()).hexdigest())[:len(pairs)]
                generic = frozen.generic_probe_masks(9 + len(pairs), (*key, "B_contender_k4_q50", "samecost"))
                generic_pred = frozen.hierarchical_omp_prediction(values, generic)
                error, ls_error, p5, swap_p5 = walsh_audit(values, old, budget)
                audit.append(dict(**base, budget=budget, p1=choices["P1"][0], legacy_p5=p5,
                                  swapped_legacy_p5=swap_p5, full_walsh_max_error=error, ls_max_error=ls_error))
                assert choices["P1"][0] == frozen.exact(frozen.degree_projection(values, 1), budget)[0]
                assert choices["Adaptive"][0] == frozen.exact(frozen.quadratic_from_pairs(
                    values, frozen.contender_pairs(frozen.degree_projection(values, 1)) if trigger else []), budget)[0]
                choices.update({"Oracle": (oracle, list(range(256))),
                                "Fixed source": (pick(historical_source[masks], masks), []),
                                "Source ensemble": (pick(ensemble[masks], masks), []),
                                "Truncated Walsh": (p5, probes9),
                                "Random pairs": quadratic({(1 << i) | (1 << j) for i, j in hashed}),
                                "Hashed subsets": (pick(generic_pred[masks], masks), generic)})
                for method, (mask, probes) in choices.items():
                    records.append(row_metrics(dict(**base, method=method), values, mask, probes, oracle, scale, epsilon, budget))
            for domain in PROTOCOL["geometry"]["domains"]:
                geometry.append(dict(**base, domain_budget=domain, **geometries(values, epsilon, scale, domain)))
            for m in PROTOCOL["nested"]["m"]:
                masks, _, _, _, _ = design(m, 3)
                oracle = pick(values[masks], masks)
                methods, _, _, _ = base_policies(read, m, 3, risk)
                for method, (mask, probes) in methods.items():
                    nested.append(row_metrics(dict(**base, m=m, method=method), values, mask, probes, oracle, scale, epsilon, 3))
    frame = pd.DataFrame(records)
    frame.to_csv(RESULTS / "m8_decisions.csv", index=False)
    pd.DataFrame(audit).to_csv(RESULTS / "walsh_audit.csv", index=False)
    pd.DataFrame(risks).to_csv(RESULTS / "risk_rows.csv", index=False)
    geom = pd.DataFrame(geometry)
    geom.to_csv(RESULTS / "geometry_rows.csv", index=False)
    for cols, suffix in [(["domain_budget"], "overall"), (["domain_budget", *CELL], "by_cell")]:
        geometry_summary(geom, cols).to_csv(RESULTS / f"geometry_{suffix}.csv", index=False)
    nd = pd.DataFrame(nested)
    nd.to_csv(RESULTS / "nested_decisions.csv", index=False)
    summarize(nd, ["m", "method"]).to_csv(RESULTS / "nested_summary.csv", index=False)
    for cols, suffix in [(["budget", "method"], "overall"), (["budget", "method", *CELL], "by_cell"),
                         (["budget", "method", "model_key"], "by_model")]:
        summarize(frame, cols).to_csv(RESULTS / f"selection_{suffix}.csv", index=False)
    paired(frame, ["normalized_regret", "raw_regret", "epsilon_optimal", "exact_set_match", "probe_count", "selected_size"],
           ["budget"]).to_csv(RESULTS / "selection_paired.csv", index=False)
    write_generation_plan(frame)
    write_risk_summary(frame, pd.DataFrame(risks))
    write_examples(frame)
    print(f"Replayed {len(raw)} complete cubes and wrote {len(frame)} decision rows.", flush=True)


def summarize(frame, grouping):
    return frame.groupby(grouping).agg(
        n=("selected_mask", "size"), mean_regret=("normalized_regret", "mean"),
        median_regret=("normalized_regret", "median"), p90_regret=("normalized_regret", lambda s: s.quantile(.9)),
        mean_raw_gap=("raw_regret", "mean"), median_raw_gap=("raw_regret", "median"),
        p90_raw_gap=("raw_regret", lambda s: s.quantile(.9)), epsilon_rate=("epsilon_optimal", "mean"),
        exact_set_rate=("exact_set_match", "mean"), mean_probes=("probe_count", "mean"),
        p90_probes=("probe_count", lambda s: s.quantile(.9)), max_probes=("probe_count", "max"),
        mean_size=("selected_size", "mean"), size0=("selected_size", lambda s: (s == 0).mean()),
        size1=("selected_size", lambda s: (s == 1).mean()), size2=("selected_size", lambda s: (s == 2).mean()),
        size3=("selected_size", lambda s: (s == 3).mean()), size4=("selected_size", lambda s: (s == 4).mean()),
    ).reset_index()


@lru_cache(maxsize=None)
def resample_indices(n):
    spec = PROTOCOL["bootstrap"]
    return np.random.default_rng(spec["seed"]).integers(n, size=(spec["resamples"], n))


def cluster_ci(frame, value):
    clustered = frame.groupby(CLUSTER)[value].agg(["sum", "count"])
    i = resample_indices(len(clustered))
    draws = clustered["sum"].to_numpy()[i].sum(axis=1) / clustered["count"].to_numpy()[i].sum(axis=1)
    lo, hi = np.quantile(draws, PROTOCOL["bootstrap"]["quantiles"])
    return dict(n=len(frame), clusters=len(clustered), mean=frame[value].mean(), ci_lo=lo, ci_hi=hi)


def paired(frame, metrics, grouping, contrasts=CONTRASTS):
    rows = []
    index = [k for k in KEY if k not in grouping]
    for values, group in frame.groupby(grouping[0] if len(grouping) == 1 else grouping):
        labels = dict(zip(grouping, values if isinstance(values, tuple) else (values,)))
        wide = group.pivot(index=index, columns="method", values=metrics)
        for left, right in contrasts:
            for metric in metrics:
                if (metric, left) not in wide or (metric, right) not in wide:
                    continue
                differences = (wide[metric, left] - wide[metric, right]).rename("difference").reset_index()
                assert differences["difference"].notna().all()
                for k in CLUSTER:
                    if k not in differences:
                        differences[k] = labels[k]
                rows.append(dict(**labels, left=left, right=right, metric=metric,
                                 median=differences.difference.median(), p90=differences.difference.quantile(.9),
                                 **cluster_ci(differences, "difference")))
    return pd.DataFrame(rows)


def geometry_summary(frame, grouping):
    rows = []
    for values, part in frame.groupby(grouping[0] if len(grouping) == 1 else grouping):
        for metric in ("material_negative_marginal_rate", "material_positive_hessian_rate",
                       "negative_marginal_rate", "positive_hessian_rate"):
            rows.append(dict(zip(grouping, values if isinstance(values, tuple) else (values,)), metric=metric,
                             **cluster_ci(part, metric)))
    return pd.DataFrame(rows)


def write_generation_plan(frame):
    plan = frame[(frame.budget == PROTOCOL["generation"]["budget"]) & frame.method.isin(GEN_ROLES)]
    old = pd.read_csv(DATA / "v4_generated_selected_sets.csv", dtype={"query_id": str})
    mapping = {"P0_historical_fixed_source": "Fixed source", "P1_mplus1": "P1",
               "P5_generic_hierarchical_omp_n9": "Truncated Walsh", "P2_all_pairs": "All-pairs",
               "B_contender_k4_q50": "Adaptive", "exact_target_optimum": "Oracle"}
    old = old.rename(columns={"target_model": "model_key", "subset_bitmask": "selected_mask"})
    old["method"] = old.selection_role.map(mapping)
    cols = KEY + ["method", "selected_mask"]
    test = old[cols].merge(plan[cols], how="left", on=cols, indicator=True)
    assert (test._merge == "both").all(), test[test._merge != "both"].head()
    plan[cols].to_csv(RESULTS / "generation_roles.csv", index=False)
    plan.groupby(KEY + ["selected_mask"]).method.agg("|".join).reset_index().to_csv(RESULTS / "generation_plan.csv", index=False)


def write_risk_summary(frame, risks):
    wide = frame[frame.budget == 3].pivot(index=KEY, columns="method", values=["normalized_regret", "probe_count"])
    part = risks.set_index(KEY)
    part["p1_regret"] = wide["normalized_regret", "P1"]
    part["adaptive_regret"] = wide["normalized_regret", "Adaptive"]
    part["check_regret"] = wide["normalized_regret", "P1-check"]
    part["adaptive_probes"] = wide["probe_count", "Adaptive"]
    part["check_probes"] = wide["probe_count", "P1-check"]
    part["adaptive_gain"] = part.p1_regret - part.adaptive_regret
    part = part.reset_index()
    measures = ["trigger", "changes_selection", "p1_regret", "adaptive_regret", "check_regret", "adaptive_gain",
                "adaptive_probes", "check_probes"]
    for cols, name in [(["model_key"], "by_model"), (CELL, "by_cell"), (["trigger"], "by_trigger")]:
        table = part.groupby(cols)[[metric for metric in measures if metric not in cols]].mean()
        table["n"] = part.groupby(cols).size()
        table.to_csv(RESULTS / f"risk_{name}.csv")


def write_examples(frame):
    scope = frame[(frame.budget == 3) & (frame.universe_name == "B_m8")]
    wide = scope.pivot(index=KEY, columns="method", values=["normalized_regret", "selected_size"])
    gains = wide["normalized_regret", "BO9"] - wide["normalized_regret", "P1"]
    masks = wide["selected_size", "P1"] >= 2
    raw = pd.read_csv(DATA / "m8_raw_mean_logprobs.csv", dtype={"query_id": str}, float_precision="round_trip").set_index(KEY)
    examples = []
    for name, eligible in [("benefit", (gains >= .05) & masks), ("counterexample", (gains <= -.05) & masks)]:
        stratum = gains[eligible]
        if stratum.empty:
            examples.append(dict(name=name, stratum_n=0, scope_n=len(wide)))
            continue
        median = float(stratum.median())
        key = min(stratum.index, key=lambda k: (abs(stratum[k] - median), k))
        result = scope.set_index(KEY).loc[[key]].drop(columns="budget").to_dict(orient="records")
        raw_row = raw.loc[key]
        examples.append(dict(name=name, stratum_n=int(eligible.sum()), scope_n=len(wide),
                             stratum_median_gain=median, key=dict(zip(KEY, key)),
                             raw_nine={str(i): float(raw_row[f"mask_{i}"]) for i in [0] + [1 << b for b in range(8)]},
                             decisions=result))
    write_json(RESULTS / "worked_examples.json", examples)
