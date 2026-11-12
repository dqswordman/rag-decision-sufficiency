# Complex Utility, Simple Calibration

**Decision Sufficiency in RAG Evidence Selection**\
Wenzhang Du · Independent Researcher, China · **WSDM 2027 Findings**

[Technical report](https://dqswordman.github.io/technical-reports/decision-sufficiency/) · [中文技术报告](https://dqswordman.github.io/zh/technical-reports/decision-sufficiency/) · [Reproduce the results](#reproduce-the-results) · [Release v1.0.0](https://github.com/dqswordman/rag-decision-sufficiency/releases/tag/v1.0.0) · [Data provenance](DATA_PROVENANCE.md)

## What one additional check changes

This artifact reproduces a labelled multi-hop RAG study with fixed eight-passage pools. The initial nine calls measure the empty context and each singleton. The additive policy (P1) uses these scores to propose a subset. The one-check policy measures that subset if needed and selects the best of the measured contexts.

| Policy | Mean target calls | Mean normalized regret ↓ | Token F1 ↑ |
| :--- | ---: | ---: | ---: |
| Best measured (base) | 9.00 | 0.213 | 0.586 |
| Additive (P1) | 9.00 | 0.272 | 0.554 |
| P1 + one check | **9.61** | **0.162** | **0.592** |

These are the primary-budget results with up to three selected passages. Relative to P1, the one-check policy reduces mean normalized regret by 0.109 in the paired comparison (95% interval [0.085, 0.138]). [Main results table](reference/tables/table_primary.tex) · [Paired comparisons](reference/tables/table_paired.tex)

## Reproduce the results

The replay starts from archived model scores. It runs locally on CPU with no model or benchmark downloads. The tested environment used Python 3.9.21 and the versions in `requirements.txt`.

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

The command verifies the archived inputs by SHA-256, recomputes the selections and paired bootstrap intervals, and checks 22 result CSVs and the exact typeset rows of all four paper tables. Use `python code/reproduce.py --verify-inputs` for an integrity check alone. A completed run writes `results/reproduction_report.json`.

## Artifact contents

| Directory | Contents |
| :--- | :--- |
| [`code/`](code/) | CPU replay, selection policies, and table generation |
| [`data/`](data/) | Archived context utilities, generation metrics, and input checksums |
| [`frozen/`](frozen/) | Frozen experiment protocols and manifests |
| [`reference/`](reference/) | Expected result CSVs and typeset rows |

## Outputs

| Output | Contents |
| --- | --- |
| `results/m8_decisions.csv` | Selected masks, observed probes, and regret for the eight-passage study |
| `results/selection_*.csv` | Regret and probe use by budget, method, and evaluation cell, with paired intervals |
| `results/generation_*.csv` | Token F1 and exact match from 8,106 distinct scored contexts, mapped to 15,840 method outcomes |
| `results/geometry_*.csv`, `results/nested_*.csv`, `results/extension_*.csv` | Feasible-domain utility geometry and matched pool-width comparisons |
| `results/worked_examples_with_generation.json` | Selected query keys, utility probes, masks, and answer metrics for the two worked cases |
| `source/table_*.tex`, `figures/regret_distribution.{pdf,png}` | Typeset table rows and the Figure 2 regret distribution |

The archived model outputs consist of numeric measurements keyed by context. Dataset sources, exact model snapshots, input columns, and scope are documented in [Data provenance](DATA_PROVENANCE.md).

## Citation and reuse

Cite Wenzhang Du, *Complex Utility, Simple Calibration: Decision Sufficiency in RAG Evidence Selection*, WSDM 2027 Findings. The artifact citation is in [CITATION.cff](CITATION.cff). Use the [v1.0.0 release](https://github.com/dqswordman/rag-decision-sufficiency/releases/tag/v1.0.0) to identify this version.

The original code is available under the [MIT License](LICENSE). See [data terms](DATA_PROVENANCE.md#reuse) for the numerical exports and benchmark identifiers.
