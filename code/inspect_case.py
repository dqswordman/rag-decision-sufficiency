"""Follow one published decision from archived scores to the selected context."""

import argparse
import csv
from itertools import combinations
import json
from math import fsum
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"
CASES = {
    "king-arthur": ("2hop__10620_49084", "mistral"),
    "ornans": ("2hop__107905_110222", "qwen"),
}


def records(filename, query, model, model_column="model_key"):
    with (DATA / filename).open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if (
                row["dataset_key"] == "musique"
                and row["query_id"] == query
                and row["universe_name"] == "B_m8"
                and row[model_column] == model
            ):
                yield row


def positions(mask):
    return "{" + ", ".join(str(bit + 1) for bit in range(8) if mask & (1 << bit)) + "}"


def inspect_case(case):
    query, model = CASES[case]
    (scores,) = records("m8_raw_mean_logprobs.csv", query, model)
    empty = float(scores["mask_0"])
    singletons = [1 << index for index in range(8)]
    initial = [0, *singletons]

    def utility(mask):
        return float(scores[f"mask_{mask}"]) - empty

    gains = {mask: utility(mask) for mask in singletons}
    candidates = {
        sum(group): fsum(gains[mask] for mask in group)
        for size in range(4)
        for group in combinations(singletons, size)
    }
    proposal = max(candidates, key=lambda mask: (candidates[mask], -mask))
    (historical,) = (
        row for row in records("v4_generated_selected_sets.csv", query, model, "target_model")
        if row["selection_role"] == "P1_mplus1"
    )
    if proposal != int(historical["subset_bitmask"]):
        raise ValueError("The proposal differs from the frozen P1 selection.")

    best_initial = max(initial, key=lambda mask: (utility(mask), -mask))
    checked = max((best_initial, proposal), key=lambda mask: (utility(mask), -mask))
    answers = {
        int(row["selected_mask"]): row
        for row in records("generation_context_scores.csv", query, model)
    }
    policies = []
    for name, mask, calls in (
        ("Best measured (BO9)", best_initial, 9),
        ("Additive (P1)", proposal, 9),
        ("P1 + one check", checked, 9 + int(proposal not in initial)),
    ):
        answer = answers[mask]
        policies.append({
            "name": name, "calls": calls, "context": positions(mask), "utility": utility(mask),
            "token_f1": float(answer["token_f1"]), "exact_match": int(float(answer["exact_match"])),
        })
    return {
        "dataset": "MuSiQue", "pool": "B_m8", "model": model, "query": query,
        "proposal_utility": candidates[proposal], "policies": policies,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="king-arthur")
    parser.add_argument("--json", action="store_true", help="Print the archived decision as JSON.")
    args = parser.parse_args()
    case = inspect_case(args.case)
    if args.json:
        print(json.dumps(case, allow_nan=False))
        return

    print("{dataset} / {pool} / {model} / {query}".format(**case))
    print(f"P1 estimates utility {case['proposal_utility']:.3f} for {case['policies'][1]['context']}.")
    print("Policy                Calls  Set           Measured U  Token F1  Answer EM")
    for policy in case["policies"]:
        print(
            "{name:21} {calls:5}  {context:12} {utility:11.3f}  {token_f1:8.3f}  {exact_match:9d}".format(**policy)
        )


if __name__ == "__main__":
    main()
