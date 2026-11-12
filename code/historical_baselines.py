"""Frozen eight-passage baselines used in the historical comparison."""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from src.eug_core import popcount


M = 8
N = 1 << M
SOURCE_MAPPING = {"qwen": "mistral", "gemma": "mistral", "mistral": "gemma", "phi": "mistral"}


def exact(values: np.ndarray, budget: int) -> tuple[int, float, float]:
    allowed = [mask for mask in range(N) if popcount(mask) <= budget]
    ordered = sorted(allowed, key=lambda mask: (-float(values[mask]), int(mask)))
    return int(ordered[0]), float(values[ordered[0]]), float(values[ordered[0]] - values[ordered[1]])


def mobius_coefficients(values: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(values, dtype=float).copy()
    for bit in range(M):
        flag = 1 << bit
        for mask in range(N):
            if mask & flag:
                coefficients[mask] -= coefficients[mask ^ flag]
    return coefficients


def mobius_reconstruct(coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float).copy()
    for bit in range(M):
        flag = 1 << bit
        for mask in range(N):
            if mask & flag:
                values[mask] += values[mask ^ flag]
    return values


def degree_projection(values: np.ndarray, maximum_degree: int) -> np.ndarray:
    coefficients = mobius_coefficients(values)
    for mask in range(N):
        if popcount(mask) > maximum_degree:
            coefficients[mask] = 0.0
    return mobius_reconstruct(coefficients)


def quadratic_from_pairs(values: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    coefficients = mobius_coefficients(values)
    retained = {0} | {1 << bit for bit in range(M)} | {(1 << i) | (1 << j) for i, j in pairs}
    for mask in range(N):
        if mask not in retained:
            coefficients[mask] = 0.0
    return mobius_reconstruct(coefficients)


def walsh_design() -> np.ndarray:
    design = np.empty((N, N), dtype=float)
    for row in range(N):
        for column in range(N):
            design[row, column] = -1.0 if popcount(row & column) % 2 else 1.0
    return design


WALSH = walsh_design()


def generic_probe_masks(count: int, identity: Sequence[Any]) -> np.ndarray:
    mandatory = [0] + [1 << bit for bit in range(M)]
    remaining = [mask for mask in range(N) if mask not in mandatory]
    prefix = "|".join(map(str, identity))
    remaining.sort(key=lambda mask: hashlib.sha256(f"{prefix}|{mask}|P5".encode("utf-8")).hexdigest())
    return np.asarray((mandatory + remaining)[:count], dtype=int)


def hierarchical_omp_prediction(values: np.ndarray, probes: np.ndarray) -> np.ndarray:
    max_terms = min(max(M + 1, len(probes) // 4), 24, len(probes) - 1)
    selected = ([0] + [1 << bit for bit in range(M)])[:max_terms]
    coefficients = np.zeros(N, dtype=float)
    while True:
        coefficients[:] = 0.0
        coefficients[selected] = np.linalg.lstsq(WALSH[np.ix_(probes, selected)], values[probes], rcond=None)[0]
        if len(selected) >= max_terms:
            break
        residual = values[probes] - WALSH[probes] @ coefficients
        candidates = []
        for interaction in range(1, N):
            if interaction in selected:
                continue
            if popcount(interaction) >= 2:
                parents = [interaction ^ (1 << bit) for bit in range(M) if interaction & (1 << bit)]
                if not all(parent in selected for parent in parents):
                    continue
            score = abs(float(np.dot(WALSH[probes, interaction], residual)))
            candidates.append((score, -interaction, interaction))
        if not candidates:
            break
        selected.append(max(candidates)[2])
    return WALSH @ coefficients


def contender_pairs(additive: np.ndarray) -> list[tuple[int, int]]:
    feasible = [mask for mask in range(N) if popcount(mask) <= 3]
    top = sorted(feasible, key=lambda mask: (-float(additive[mask]), int(mask)))[:4]
    result: set[tuple[int, int]] = set()
    for mask in top:
        bits = [bit for bit in range(M) if mask & (1 << bit)]
        for i in range(len(bits)):
            for j in range(i + 1, len(bits)):
                result.add((bits[i], bits[j]))
    return sorted(result)
