"""Deterministic Boolean-lattice and generator-utility primitives for EUG.

The functions in this module do not choose scientific thresholds or inspect
final outcomes. They implement the protocol's fixed mathematical operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def enumerate_bitmasks(m: int) -> Tuple[int, ...]:
    """Return every subset bitmask in deterministic integer order."""
    if not isinstance(m, int) or isinstance(m, bool) or m < 0:
        raise ValueError("m must be a nonnegative integer")
    return tuple(range(1 << m))


def members_from_bitmask(bitmask: int, m: int) -> Tuple[int, ...]:
    """Decode a bitmask into ascending zero-based candidate indices."""
    _validate_bitmask(bitmask, m)
    return tuple(i for i in range(m) if bitmask & (1 << i))


def bitmask_from_members(members: Iterable[int], m: int) -> int:
    """Encode unique zero-based candidate indices as a subset bitmask."""
    if not isinstance(m, int) or isinstance(m, bool) or m < 0:
        raise ValueError("m must be a nonnegative integer")
    result = 0
    for item in members:
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError("candidate indices must be integers")
        if item < 0 or item >= m:
            raise ValueError("candidate index outside [0, m)")
        flag = 1 << item
        if result & flag:
            raise ValueError("candidate indices must be unique")
        result |= flag
    return result


def popcount(bitmask: int) -> int:
    if not isinstance(bitmask, int) or isinstance(bitmask, bool) or bitmask < 0:
        raise ValueError("bitmask must be a nonnegative integer")
    # int.bit_count is unavailable in the repository's required Python 3.9
    # environment. This exact operation is equivalent for nonnegative masks.
    return bin(bitmask).count("1")


def _validate_bitmask(bitmask: int, m: int) -> None:
    if not isinstance(m, int) or isinstance(m, bool) or m < 0:
        raise ValueError("m must be a nonnegative integer")
    if not isinstance(bitmask, int) or isinstance(bitmask, bool):
        raise ValueError("bitmask must be an integer")
    if bitmask < 0 or bitmask >= (1 << m):
        raise ValueError("bitmask outside the m-candidate universe")


def canonical_subset_records(
    records: Sequence[Mapping[str, Any]],
    bitmask: int,
    *,
    id_key: str = "candidate_id",
    order_key: str = "canonical_position",
) -> Tuple[Mapping[str, Any], ...]:
    """Select records and order them by the frozen canonical position.

    Input row order is intentionally ignored. Canonical positions must be the
    exact permutation 0..m-1 and candidate identifiers must be unique.
    """
    m = len(records)
    _validate_bitmask(bitmask, m)
    positions = [record[order_key] for record in records]
    if sorted(positions) != list(range(m)):
        raise ValueError("canonical positions must be a permutation of 0..m-1")
    identifiers = [str(record[id_key]) for record in records]
    if len(set(identifiers)) != m:
        raise ValueError("candidate identifiers must be unique")
    selected = [
        record
        for record in records
        if bitmask & (1 << int(record[order_key]))
    ]
    selected.sort(key=lambda record: int(record[order_key]))
    return tuple(selected)


def canonical_serialize(
    records: Sequence[Mapping[str, Any]],
    bitmask: int,
    *,
    text_key: str = "fixed_text",
) -> str:
    """Serialize selected fixed representations without token reallocation."""
    selected = canonical_subset_records(records, bitmask)
    blocks = []
    for record in selected:
        text = str(record[text_key])
        blocks.append(
            "[EVIDENCE {position}]\n{content}\n[/EVIDENCE {position}]".format(
                position=int(record["canonical_position"]) + 1,
                content=text,
            )
        )
    return "\n\n".join(blocks)


def serialize_subset_with_global_order(
    records: Sequence[Mapping[str, Any]],
    bitmask: int,
    global_order: Sequence[int],
    *,
    text_key: str = "fixed_text",
) -> str:
    """Serialize a subset under a frozen global candidate permutation.

    Evidence block labels remain tied to primary canonical positions so the
    order control changes only block order, not text, labels, or membership.
    """
    m = len(records)
    _validate_bitmask(bitmask, m)
    order = tuple(global_order)
    if len(order) != m or set(order) != set(range(m)):
        raise ValueError("global_order must be a permutation of 0..m-1")
    by_position = {int(record["canonical_position"]): record for record in records}
    if set(by_position) != set(range(m)):
        raise ValueError("canonical positions must be a permutation of 0..m-1")
    identifiers = [str(record["candidate_id"]) for record in records]
    if len(set(identifiers)) != m:
        raise ValueError("candidate identifiers must be unique")
    blocks = []
    for position in order:
        if not (bitmask & (1 << position)):
            continue
        record = by_position[position]
        blocks.append(
            "[EVIDENCE {label}]\n{content}\n[/EVIDENCE {label}]".format(
                label=position + 1,
                content=str(record[text_key]),
            )
        )
    return "\n\n".join(blocks)


def stable_json_hash(value: Any) -> str:
    """SHA-256 of canonical UTF-8 JSON."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_identity(
    *,
    dataset: str,
    query_id: str,
    candidate_universe_hash: str,
    subset_bitmask: int,
    generator_revision: str,
    tokenizer_revision: str,
    prompt_hash: str,
    serialization_hash: str,
) -> str:
    """Return the immutable raw-evaluation cache key required by protocol."""
    fields = {
        "candidate_universe_hash": candidate_universe_hash,
        "dataset": dataset,
        "generator_revision": generator_revision,
        "prompt_hash": prompt_hash,
        "query_id": query_id,
        "serialization_hash": serialization_hash,
        "subset_bitmask": subset_bitmask,
        "tokenizer_revision": tokenizer_revision,
    }
    return stable_json_hash(fields)


def mean_target_logprob_from_logits(
    logits: Any,
    input_ids: Any,
    target_start: int,
) -> Tuple[float, int]:
    """Score target tokens from causal-LM logits using teacher forcing.

    `target_start` is the first input position belonging to the fixed target.
    The logit at position t-1 predicts the token at position t. The batch size
    must be one so that each raw record maps to one deterministic target.
    """
    import torch

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [1,T,V] and input_ids [1,T]")
    if logits.shape[0] != 1 or input_ids.shape[0] != 1:
        raise ValueError("batch size must be one")
    if logits.shape[1] != input_ids.shape[1]:
        raise ValueError("logit and input sequence lengths differ")
    sequence_length = int(input_ids.shape[1])
    if target_start <= 0 or target_start >= sequence_length:
        raise ValueError("target_start must identify at least one predictable target token")
    prediction_logits = logits[:, target_start - 1 : sequence_length - 1, :]
    target_ids = input_ids[:, target_start:sequence_length]
    token_logprobs = torch.log_softmax(prediction_logits.float(), dim=-1)
    gathered = token_logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(gathered).all():
        raise FloatingPointError("non-finite target token log probability")
    return float(gathered.mean().item()), int(target_ids.numel())


def score_teacher_forced_ids(
    model: Any,
    prefix_ids: Sequence[int],
    target_ids: Sequence[int],
    *,
    device: Any,
) -> Tuple[float, int]:
    """Run a causal LM on explicitly separated prefix and target token IDs."""
    import torch

    if not prefix_ids:
        raise ValueError("prefix_ids cannot be empty")
    if not target_ids:
        raise ValueError("target_ids cannot be empty")
    input_ids = torch.tensor(
        [list(prefix_ids) + list(target_ids)], dtype=torch.long, device=device
    )
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)
    return mean_target_logprob_from_logits(output.logits, input_ids, len(prefix_ids))


def center_utilities(raw_scores: Mapping[int, float], m: int) -> dict[int, float]:
    """Center all subset scores against the empty-evidence score."""
    expected = set(enumerate_bitmasks(m))
    actual = set(raw_scores)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"incomplete utility cube; missing={missing}, extra={extra}")
    values = np.asarray([raw_scores[mask] for mask in range(1 << m)], dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError("utility cube contains NaN or infinite values")
    empty = float(raw_scores[0])
    centered = {mask: float(raw_scores[mask] - empty) for mask in expected}
    centered[0] = 0.0
    return centered


@dataclass(frozen=True)
class MarginalRecord:
    base_bitmask: int
    candidate_index: int
    augmented_bitmask: int
    marginal: float


def compute_marginals(
    utilities: Mapping[int, float], m: int
) -> Tuple[MarginalRecord, ...]:
    _validate_complete_cube(utilities, m)
    result = []
    for base in enumerate_bitmasks(m):
        for candidate in range(m):
            flag = 1 << candidate
            if base & flag:
                continue
            augmented = base | flag
            result.append(
                MarginalRecord(
                    base_bitmask=base,
                    candidate_index=candidate,
                    augmented_bitmask=augmented,
                    marginal=float(utilities[augmented] - utilities[base]),
                )
            )
    return tuple(result)


@dataclass(frozen=True)
class HessianRecord:
    base_bitmask: int
    candidate_i: int
    candidate_j: int
    pair_bitmask: int
    hessian: float


def compute_hessians(
    utilities: Mapping[int, float], m: int
) -> Tuple[HessianRecord, ...]:
    """Compute each unordered pair's exact discrete Hessian on every base set."""
    _validate_complete_cube(utilities, m)
    result = []
    for base in enumerate_bitmasks(m):
        for i in range(m):
            i_flag = 1 << i
            if base & i_flag:
                continue
            for j in range(i + 1, m):
                j_flag = 1 << j
                if base & j_flag:
                    continue
                hessian = (
                    utilities[base | i_flag | j_flag]
                    - utilities[base | i_flag]
                    - utilities[base | j_flag]
                    + utilities[base]
                )
                result.append(
                    HessianRecord(
                        base_bitmask=base,
                        candidate_i=i,
                        candidate_j=j,
                        pair_bitmask=i_flag | j_flag,
                        hessian=float(hessian),
                    )
                )
    return tuple(result)


def _validate_complete_cube(utilities: Mapping[int, float], m: int) -> None:
    expected = set(enumerate_bitmasks(m))
    actual = set(utilities)
    if actual != expected:
        raise ValueError("utilities must contain every and only valid subset bitmask")
    values = np.asarray([utilities[mask] for mask in range(1 << m)], dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError("utilities contain NaN or infinite values")


@dataclass(frozen=True)
class SelectionResult:
    bitmask: int
    utility: float
    selected_indices: Tuple[int, ...]
    trajectory: Tuple[int, ...]


def exact_optimum(
    utilities: Mapping[int, float],
    m: int,
    budget: int,
    *,
    at_most: bool = True,
) -> SelectionResult:
    """Exact deterministic cardinality-constrained optimum.

    Ties are resolved by the numerically smaller bitmask, a rule frozen before
    data. With `at_most=True`, the empty set is a valid candidate.
    """
    _validate_complete_cube(utilities, m)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0 or budget > m:
        raise ValueError("budget must be an integer in [0,m]")
    eligible = [
        mask
        for mask in enumerate_bitmasks(m)
        if (popcount(mask) <= budget if at_most else popcount(mask) == budget)
    ]
    best = min(eligible, key=lambda mask: (-float(utilities[mask]), mask))
    return SelectionResult(
        bitmask=best,
        utility=float(utilities[best]),
        selected_indices=members_from_bitmask(best, m),
        trajectory=(best,),
    )


def true_marginal_greedy(
    utilities: Mapping[int, float],
    m: int,
    budget: int,
    *,
    stop_on_nonpositive: bool,
) -> SelectionResult:
    """Oracle greedy using observed true set-function marginals."""
    _validate_complete_cube(utilities, m)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0 or budget > m:
        raise ValueError("budget must be an integer in [0,m]")
    current = 0
    trajectory = [0]
    for _ in range(budget):
        options = []
        for candidate in range(m):
            flag = 1 << candidate
            if current & flag:
                continue
            augmented = current | flag
            marginal = float(utilities[augmented] - utilities[current])
            options.append((marginal, candidate, augmented))
        if not options:
            break
        marginal, _, augmented = min(options, key=lambda item: (-item[0], item[1]))
        if stop_on_nonpositive and marginal <= 0.0:
            break
        current = augmented
        trajectory.append(current)
    return SelectionResult(
        bitmask=current,
        utility=float(utilities[current]),
        selected_indices=members_from_bitmask(current, m),
        trajectory=tuple(trajectory),
    )


def singleton_top_b(
    utilities: Mapping[int, float], m: int, budget: int
) -> SelectionResult:
    """Select the b candidates with largest centered singleton utility."""
    _validate_complete_cube(utilities, m)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0 or budget > m:
        raise ValueError("budget must be an integer in [0,m]")
    ranking = sorted(range(m), key=lambda i: (-float(utilities[1 << i]), i))
    current = 0
    trajectory = [0]
    for candidate in ranking[:budget]:
        current |= 1 << candidate
        trajectory.append(current)
    return SelectionResult(
        bitmask=current,
        utility=float(utilities[current]),
        selected_indices=members_from_bitmask(current, m),
        trajectory=tuple(trajectory),
    )


def robust_iqr_scale(values: Sequence[float], *, floor: float = 1e-6) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot estimate a scale from no values")
    if not np.isfinite(array).all():
        raise FloatingPointError("scale values contain NaN or infinity")
    if floor <= 0 or not np.isfinite(floor):
        raise ValueError("floor must be finite and positive")
    q25, q75 = np.quantile(array, [0.25, 0.75])
    return float(max(float(q75 - q25), floor))


def calibration_materiality_threshold(scale: float) -> float:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return float(max(1e-3, 0.05 * scale))


def geometry_summary(
    utilities: Mapping[int, float],
    m: int,
    *,
    scale: float,
    epsilon_material: float,
) -> dict[str, float]:
    """Threshold-free and calibrated-materiality per-query geometry summary."""
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    if epsilon_material < 0 or not np.isfinite(epsilon_material):
        raise ValueError("epsilon_material must be finite and nonnegative")
    marginals = np.asarray([record.marginal for record in compute_marginals(utilities, m)])
    hessians = np.asarray([record.hessian for record in compute_hessians(utilities, m)])
    negative = np.maximum(-marginals, 0.0)
    positive_h = np.maximum(hessians, 0.0)
    return {
        "negative_marginal_rate": float(np.mean(marginals < 0.0)),
        "material_negative_marginal_rate": float(np.mean(marginals < -epsilon_material)),
        "mean_negative_marginal_magnitude": float(negative[marginals < 0.0].mean())
        if np.any(marginals < 0.0)
        else 0.0,
        "worst_negative_marginal": float(marginals.min()),
        "negative_marginal_mass": float(negative.sum()),
        "negative_marginal_mass_normalized": float(negative.sum() / scale),
        "positive_hessian_rate": float(np.mean(hessians > 0.0)),
        "material_positive_hessian_rate": float(np.mean(hessians > epsilon_material)),
        "mean_positive_hessian": float(positive_h[hessians > 0.0].mean())
        if np.any(hessians > 0.0)
        else 0.0,
        "max_positive_hessian": float(max(float(hessians.max()), 0.0)),
        "positive_hessian_mass": float(positive_h.sum()),
        "positive_hessian_mass_normalized": float(positive_h.sum() / scale),
    }


def oracle_regret(oracle: SelectionResult, method: SelectionResult, scale: float) -> dict[str, float]:
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and positive")
    regret = float(oracle.utility - method.utility)
    if regret < -1e-9:
        raise AssertionError("method utility exceeds claimed exact oracle")
    regret = max(regret, 0.0)
    return {"regret": regret, "normalized_regret": float(regret / scale)}


def normalize_answer(text: str) -> str:
    """Official-style lower/article/punctuation/whitespace normalization."""
    import re
    import string

    value = str(text).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_exact_match(prediction: str, answers: Sequence[str]) -> float:
    if not answers:
        raise ValueError("answers cannot be empty")
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(answer) for answer in answers))


def answer_token_f1(prediction: str, answers: Sequence[str]) -> float:
    """Maximum token F1 over frozen official aliases."""
    from collections import Counter

    if not answers:
        raise ValueError("answers cannot be empty")
    prediction_tokens = normalize_answer(prediction).split()
    scores = []
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not prediction_tokens and not answer_tokens:
            scores.append(1.0)
            continue
        if not prediction_tokens or not answer_tokens:
            scores.append(0.0)
            continue
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        scores.append(2.0 * precision * recall / (precision + recall))
    return float(max(scores))


def bootstrap_query_statistic(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    resamples: int = 10000,
    seed: int = 20270811,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap over query-level values only."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a nonempty one-dimensional query array")
    if not np.isfinite(array).all():
        raise FloatingPointError("query values contain NaN or infinity")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = array[rng.integers(0, array.size, size=array.size)]
        estimates[index] = float(statistic(sample))
    if not np.isfinite(estimates).all():
        raise FloatingPointError("bootstrap statistic produced NaN or infinity")
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": float(statistic(array)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "alpha": float(alpha),
        "resamples": int(resamples),
        "seed": int(seed),
        "sampling_unit": "query",
    }


def validate_disjoint_splits(
    calibration_ids: Iterable[str], final_ids: Iterable[str]
) -> dict[str, int]:
    calibration = [str(item) for item in calibration_ids]
    final = [str(item) for item in final_ids]
    if len(set(calibration)) != len(calibration):
        raise ValueError("duplicate query ID within calibration split")
    if len(set(final)) != len(final):
        raise ValueError("duplicate query ID within final split")
    overlap = sorted(set(calibration).intersection(final))
    if overlap:
        raise ValueError(f"calibration/final overlap: {overlap}")
    return {
        "calibration_count": len(calibration),
        "final_count": len(final),
        "overlap_count": 0,
    }
