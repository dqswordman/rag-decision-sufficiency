"""Recover the discovery IQR scales and the adaptive q50 from their original inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from replay import DATA, RESULTS, scales
from src.eug_core import calibration_materiality_threshold, robust_iqr_scale


GROUP = ["dataset_key", "model_key", "universe_name"]


def main():
    risk = pd.read_csv(DATA / "adaptive_discovery_risk.csv", float_precision="round_trip")
    assert (risk.primary_risk_score == -risk.raw_empty_mean_logprob).all()
    q50 = float(np.quantile(risk.primary_risk_score, .5))
    assert q50 == 7.288293838500977
    frozen = scales().set_index(GROUP)
    raw = pd.read_csv(DATA / "m6_calibration_raw_mean_logprobs.csv", float_precision="round_trip")
    records = []
    for key, rows in raw.groupby(GROUP):
        assert len(rows) == 60 and rows.query_id.nunique() == 60
        values = rows[[f"mask_{i}" for i in range(64)]].to_numpy()
        centered = (values - values[:, [0]]).ravel()
        scale = robust_iqr_scale(centered)
        epsilon = calibration_materiality_threshold(scale)
        match = frozen.loc[(key[0], key[1], key[2].replace("m6", "m8"))]
        assert abs(scale - match.scale) <= 1e-12 and abs(epsilon - match.epsilon) <= 1e-12
        records.append(dict(zip(GROUP, key), n=len(rows), scores=len(centered),
                            q25=np.quantile(centered, .25), q75=np.quantile(centered, .75), scale=scale, epsilon=epsilon))
    pd.DataFrame(records).to_csv(RESULTS / "m6_scale_reproduction.csv", index=False)
    print(f"Reproduced {len(records)} original cell scales, verified six fixed Phi values and recovered q50={q50} from {len(risk)} discovery rows.")
