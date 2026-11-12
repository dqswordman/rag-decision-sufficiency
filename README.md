# Complex Utility, Simple Calibration

**Decision Sufficiency in RAG Evidence Selection**\
Wenzhang Du · **WSDM 2027 Findings**

Replay an evidence-selection study from archived model scores on CPU. The tagged artifact regenerates the selections, paired comparisons, and paper tables.

[Technical report](https://dqswordman.github.io/technical-reports/decision-sufficiency/) · [中文技术报告](https://dqswordman.github.io/zh/technical-reports/decision-sufficiency/) · [CPU quickstart](#reproduce-the-results) · [Release v1.0.0](https://github.com/dqswordman/rag-decision-sufficiency/releases/tag/v1.0.0) · [Data provenance](DATA_PROVENANCE.md)

## Results at a glance

Nine measurements score the empty context and each singleton in an eight-passage pool. The additive policy (P1) proposes up to three passages. A targeted check scores that subset if needed and returns the highest-utility measured context.

[![Mean normalized regret across 1,440 cases. Best measured 0.213, additive 0.272, additive plus one check 0.162.](https://dqswordman.github.io/assets/reports/decision-sufficiency/mean-regret.png)](https://dqswordman.github.io/technical-reports/decision-sufficiency/#results)

| Policy | Mean target utility calls | Mean normalized regret ↓ | Token F1 ↑ | Answer EM ↑ |
| :--- | ---: | ---: | ---: | ---: |
| Best measured (BO9) | 9.00 | 0.213 | 0.586 | 0.444 |
| Additive (P1) | 9.00 | 0.272 | 0.554 | 0.422 |
| P1 + one check | **9.61** | **0.162** | **0.592** | **0.453** |

The primary comparison covers 1,440 question–pool–generator cases with a three-passage selection limit. Relative to P1, the one-check policy reduces mean normalized regret by 0.109 in the paired comparison (95% interval [0.085, 0.138]). [Main results table](reference/tables/table_primary.tex) · [Paired comparisons](reference/tables/table_paired.tex) · [Worked examples](https://dqswordman.github.io/technical-reports/decision-sufficiency/#worked-examples)

## Reproduce the results

The replay starts from archived model scores and runs locally on CPU with no model or benchmark downloads. The v1.0.0 verification used Python 3.9.21 and the versions in `requirements.txt`.

```sh
git clone --branch v1.0.0 https://github.com/dqswordman/rag-decision-sufficiency.git
cd rag-decision-sufficiency
python -m venv .venv
```

Activate the environment with `. .venv/bin/activate` on a POSIX shell or `.\.venv\Scripts\Activate.ps1` in PowerShell. Install the pinned dependencies and run the replay.

```sh
python -m pip install -r requirements.txt
python code/reproduce.py
```

The command verifies the archived inputs by SHA-256, recomputes the selections and paired bootstrap intervals, and checks 22 result CSVs and the exact typeset rows of all four paper tables. Use `python code/reproduce.py --verify-inputs` for an integrity check alone. A completed run writes `results/reproduction_report.json`. [Release verification record](https://github.com/dqswordman/rag-decision-sufficiency/releases/download/v1.0.0/reproduction_report.json)

<details>
<summary>Archive layout</summary>

| Directory | Contents |
| :--- | :--- |
| [`code/`](code/) | CPU replay, selection policies, and table generation |
| [`data/`](data/) | Archived context utilities, generation metrics, and input checksums |
| [`frozen/`](frozen/) | Frozen experiment protocols and manifests |
| [`reference/`](reference/) | Expected result CSVs and typeset rows |

</details>

## Trace the results

| Output | Contents |
| --- | --- |
| `results/selection_overall.csv` and `results/selection_paired.csv` | Main regret and utility-call results with paired intervals. [Archived references](reference/results/) |
| `results/generation_overall.csv` and `results/generation_paired.csv` | Token F1 and exact match from 8,106 distinct scored contexts, mapped to 15,840 method outcomes |
| `results/m8_decisions.csv` | Selected masks, observed probes, and regret for the eight-passage study |
| `results/geometry_*.csv`, `results/nested_*.csv`, `results/extension_*.csv` | Feasible-domain utility geometry and matched pool-width comparisons |
| `results/worked_examples_with_generation.json` | Selected query keys, utility probes, masks, and answer metrics for the two worked cases |
| `source/table_*.tex`, `figures/regret_distribution.{pdf,png}` | Typeset table rows and the Figure 2 regret distribution |

The archived model outputs consist of numeric measurements keyed by context. Dataset sources, exact model snapshots, input columns, and scope are documented in [Data provenance](DATA_PROVENANCE.md).

## Citation and reuse

Cite Wenzhang Du, *Complex Utility, Simple Calibration: Decision Sufficiency in RAG Evidence Selection*, WSDM 2027 Findings. The artifact citation is in [CITATION.cff](CITATION.cff). Use the [v1.0.0 release](https://github.com/dqswordman/rag-decision-sufficiency/releases/tag/v1.0.0) to identify this version.

The original code is available under the [MIT License](LICENSE). See [data terms](DATA_PROVENANCE.md#reuse) for the numerical exports and benchmark identifiers.
