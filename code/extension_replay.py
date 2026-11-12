"""Replay the paired MuSiQue candidate-pool extension from archived scores."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from replay import DATA, KEY, RESULTS, base_policies, cluster_ci, design, pick, row_metrics, scales, summarize


EXTENSION_CONTRASTS = [("P1", "BO9"), ("P1-check", "BO9"), ("BO-adaptive", "Adaptive"), ("P1-check", "Retrieval top-b")]
METRICS = ["normalized_regret", "token_f1", "exact_match"]


def choices_for_pool(raw, ranks, model, qid, widths, budget, calibration):
    values = {mask: value - raw[0] for mask, value in raw.items()}
    retrieval_mask = sum(1 << row.canonical_position for row in ranks.itertuples() if row.rrf_rank <= budget)
    for width in widths:
        masks, _, _, _, _ = design(width, budget)
        oracle = pick(np.array([values[mask] for mask in masks]), masks)
        choices, _, _, _ = base_policies(values.__getitem__, width, budget, -raw[0])
        choices["Retrieval top-b"] = (retrieval_mask, [])
        choices["Oracle"] = (oracle, masks)
        for method, (mask, probes) in choices.items():
            base = dict(dataset_key="musique", query_id=qid, universe_name="B_m8", model_key=model, m=width, method=method)
            yield row_metrics(base, values, mask, probes, oracle, calibration.scale, calibration.epsilon, budget)


def summarize_extension(frame):
    for grouping, file in [(["m", "method"], "summary"), (["m", "method", "model_key"], "by_model")]:
        answers = frame.groupby(grouping, as_index=False)[METRICS[1:]].mean()
        summary = summarize(frame, grouping).merge(answers, on=grouping, validate="one_to_one")
        summary.to_csv(RESULTS / f"extension_{file}.csv", index=False)
    comparisons = []
    for width, part in frame.groupby("m"):
        wide = part.pivot(index=KEY, columns="method", values=METRICS)
        for left, right in EXTENSION_CONTRASTS:
            for metric in METRICS:
                delta = (wide[metric, left] - wide[metric, right]).rename("delta").reset_index()
                comparisons.append(dict(m=width, left=left, right=right, metric=metric, **cluster_ci(delta, "delta")))
    pd.DataFrame(comparisons).to_csv(RESULTS / "extension_paired.csv", index=False)


def main():
    spec = json.loads((DATA / "extension_protocol.json").read_text(encoding="utf-8"))
    freeze = json.loads((DATA / "extension_freeze.json").read_text(encoding="utf-8"))
    raw_frame = pd.read_csv(DATA / "extension_raw_mean_logprobs.csv", dtype={"query_id": str}, float_precision="round_trip")
    answers = pd.read_csv(DATA / "extension_context_scores.csv", dtype={"query_id": str}, float_precision="round_trip")
    answers = answers.set_index(["model_key", "query_id", "mask"], verify_integrity=True)
    ranks = pd.read_csv(DATA / "extension_candidate_ranks.csv", dtype={"query_id": str})
    by_query = {query: group for query, group in ranks.groupby("query_id")}
    calibration = scales().query("dataset_key == 'musique' and universe_name == 'B_m8'").set_index("model_key")
    expected_keys = {(model, query) for model in spec["models"] for query in freeze["selected_query_ids"]}
    groups = raw_frame.groupby(["model_key", "query_id"])
    assert len(expected_keys) == 40 and set(groups.groups) == expected_keys
    assert set(by_query) == set(freeze["selected_query_ids"])
    widest = max(spec["pool_sizes"])
    budget = spec["budget"]
    valid_masks = set(design(widest, budget)[0])
    for pool in by_query.values():
        assert sorted(pool.canonical_position) == list(range(widest))
        assert sorted(pool.rrf_rank) == list(range(1, widest + 1))

    records, selected_contexts = [], set()
    for (model, qid), group in groups:
        raw = dict(zip(group["mask"], group["mean_logprob"]))
        assert len(raw) == len(group) == 299 and raw.keys() == valid_masks and np.isfinite(group.mean_logprob).all()
        for decision in choices_for_pool(raw, by_query[qid], model, qid, spec["pool_sizes"], budget, calibration.loc[model]):
            context = (model, qid, decision["selected_mask"])
            selected_contexts.add(context)
            score = answers.loc[context]
            records.append(dict(**decision, token_f1=score.token_f1, exact_match=score.exact_match))
    assert selected_contexts == set(answers.index)
    frame = pd.DataFrame(records)
    assert len(frame) == 720
    frame.to_csv(RESULTS / "extension_decisions.csv", index=False)
    summarize_extension(frame)
    print(f"Replayed {len(frame):,} decisions across the two matched extension widths.", flush=True)
