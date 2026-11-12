"""Replay generated-answer comparisons from one archived score per context."""
from __future__ import annotations

import json

import pandas as pd

from replay import CELL, DATA, KEY, RESULTS, cluster_ci, paired, write_json


CONTEXT = [*KEY, "selected_mask"]
METRICS = ["token_f1", "exact_match", "generated_token_count"]


def read_csv(path):
    return pd.read_csv(path, dtype={"query_id": str}, float_precision="round_trip")


def compare_decoding(answers):
    legacy = read_csv(DATA / "legacy_context_scores.csv")
    history = answers.merge(legacy, on=CONTEXT, validate="one_to_one")
    assert len(history) == len(legacy)
    shifts = []
    for model, group in history.groupby("model_key"):
        shifts.append(dict(model_key=model, identical_contexts=len(group),
                           legacy_f1=group.old_token_f1.mean(), revised_f1=group.token_f1.mean(),
                           legacy_em=group.old_exact_match.mean(), revised_em=group.exact_match.mean(),
                           legacy_64_rate=(group.old_generated_token_count == 64).mean(),
                           revised_64_rate=(group.generated_token_count == 64).mean(),
                           mean_old_tokens=group.old_generated_token_count.mean(),
                           mean_revised_tokens=group.generated_token_count.mean()))
    pd.DataFrame(shifts).to_csv(RESULTS / "decode_audit_summary.csv", index=False)


def attach_example_scores(answers):
    examples = json.loads((RESULTS / "worked_examples.json").read_text(encoding="utf-8"))
    lookup = answers.set_index(CONTEXT)
    for example in examples:
        if not example["stratum_n"]:
            continue
        identity = tuple(example["key"][name] for name in KEY)
        for row in example["decisions"]:
            context = (*identity, row["selected_mask"])
            if context in lookup.index:
                score = lookup.loc[context]
                row["generated"] = {"token_f1": float(score.token_f1), "exact_match": int(score.exact_match)}
    write_json(RESULTS / "worked_examples_with_generation.json", examples)


def main():
    plans = read_csv(RESULTS / "generation_plan.csv")
    roles = read_csv(RESULTS / "generation_roles.csv")
    answers = read_csv(DATA / "generation_context_scores.csv")
    answers = plans[CONTEXT].merge(answers, on=CONTEXT, validate="one_to_one", how="outer", indicator=True)
    assert len(answers) == 8106 and (answers._merge == "both").all()
    assert answers.token_f1.between(0, 1).all() and answers.exact_match.isin([0, 1]).all()
    merged = roles.merge(answers[CONTEXT + METRICS + ["origin"]], on=CONTEXT, validate="many_to_one")
    assert len(merged) == len(roles) == 15840 and (merged.groupby("method").size() == 1440).all()
    merged.to_csv(RESULTS / "generation_scores_by_role.csv", index=False)
    merged["budget"] = 3
    for grouping, label in [(["method"], "overall"), (["method", "model_key"], "by_model"), (["method", *CELL], "by_cell")]:
        stats = merged.groupby(grouping).agg(n=("token_f1", "size"), token_f1=("token_f1", "mean"),
                                             exact_match=("exact_match", "mean"), mean_new_tokens=("generated_token_count", "mean"))
        stats.reset_index().to_csv(RESULTS / f"generation_{label}.csv", index=False)
    for grouping, name in [(["budget"], "paired"), (["model_key"], "paired_by_model"), (CELL, "paired_by_cell")]:
        paired(merged, METRICS[:2], grouping).to_csv(RESULTS / f"generation_{name}.csv", index=False)

    risk = read_csv(RESULTS / "risk_rows.csv")
    changed = risk[(risk.trigger == 1) & (risk.changes_selection == 1)][KEY]
    contrast = merged[merged.method.isin(["Adaptive", "P1"])].pivot(index=KEY, columns="method", values="token_f1")
    subset = (contrast["Adaptive"] - contrast["P1"]).rename("difference").reset_index().merge(changed, on=KEY, validate="one_to_one")
    write_json(RESULTS / "generation_exploratory_triggered_change.json", cluster_ci(subset, "difference"))
    compare_decoding(answers)
    attach_example_scores(answers)
    print(f"Matched {len(answers):,} archived context scores to {len(merged):,} method outcomes.", flush=True)
