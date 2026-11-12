"""Follow one published decision from archived scores to the selected context."""

import argparse
import csv
from itertools import combinations
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="king-arthur")
    args = parser.parse_args()
    query, model = CASES[args.case]
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
    print(f"MuSiQue / B_m8 / {model} / {query}")
    print(f"P1 estimates utility {candidates[proposal]:.3f} for {positions(proposal)}.")
    print("Policy                Calls  Set           Measured U  Token F1  Answer EM")
    for name, mask, calls in (
        ("Best measured (BO9)", best_initial, 9),
        ("Additive (P1)", proposal, 9),
        ("P1 + one check", checked, 9 + int(proposal not in initial)),
    ):
        answer = answers[mask]
        print(
            f"{name:21} {calls:5}  {positions(mask):12} {utility(mask):11.3f}"
            f"  {float(answer['token_f1']):8.3f}  {int(float(answer['exact_match'])):9d}"
        )


if __name__ == "__main__":
    main()
