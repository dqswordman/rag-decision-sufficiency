# Complex Utility, Simple Calibration

**Decision Sufficiency in RAG Evidence Selection** · Wenzhang Du · Independent Researcher, China · WSDM 2027 Findings

This release reproduces the numerical results of a labelled multi-hop RAG study. With nine utility measurements per eight-passage pool, the best measured context has lower mean regret than an additive selector. Measuring the additive selector's proposal once more reduces mean regret from 0.213 to 0.162 at an average cost of 9.61 target calls. The artifact includes the policy code, complete utility arrays, and per-context answer metrics for auditing that comparison and the other reported experiments.

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
