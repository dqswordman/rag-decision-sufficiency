# Reproduce the WSDM 2027 Findings artifact

Wenzhang Du, *Complex Utility, Simple Calibration: Decision Sufficiency in RAG Evidence Selection*, author-final manuscript dated 11 November 2026.

The tagged artifact `v1.0.0` at commit `7b0f639a207b1826915227aaff2eef3addea499d` starts from archived scalar model scores. It reconstructs the selection policies, paired statistics, generation-score joins, four manuscript table bodies, and Figure 2. [Release](https://github.com/dqswordman/rag-decision-sufficiency/releases/tag/v1.0.0) · [Web companion](https://dqswordman.github.io/technical-reports/decision-sufficiency/)

## Try one archived decision

The default branch has a small example that uses only the Python standard library. With Python 3.9 or newer, run it from a fresh copy of the repository.

```sh
git clone --depth 1 https://github.com/dqswordman/rag-decision-sufficiency.git
cd rag-decision-sufficiency
python code/inspect_case.py
```

The default case is the MuSiQue King Arthur question in the support-blind pool with Mistral. The program reads the nine initial scores, recomputes the additive proposal, checks its mask against the frozen P1 selection, and looks up the recorded answer metrics. Passage positions are one-based.

| Policy | Context | Measured utility | Token F1 | Answer EM |
| --- | --- | ---: | ---: | ---: |
| Best measured (BO9) | {1} | 0.918 | 0.444 | 0 |
| Additive (P1) | {1, 5} | 0.206 | 1.000 | 1 |
| P1 + one check | {1} | 0.918 | 0.444 | 0 |

P1 predicts utility 1.574 for the pair. Measuring it gives 0.206, so the checked policy returns the singleton at 0.918. To see the archived case where a check keeps the proposal, run `python code/inspect_case.py --case ornans`. This path verifies the behavior of one archived decision per invocation.

## Replay the paper results

Use the tagged source snapshot for the full replay. The verified environment used Windows 11 x64, Python 3.9.21, NumPy 2.0.2, pandas 2.3.3, PyYAML 6.0.2, and Matplotlib 3.8.4. Installation uses the pinned packages in `requirements.txt`. The replay itself uses CPU and the included numerical archive.

```sh
git clone --branch v1.0.0 https://github.com/dqswordman/rag-decision-sufficiency.git rag-v1.0.0
cd rag-v1.0.0
python -m venv .venv
```

On Linux or macOS, run the following with a Python 3.9.21 environment.

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/reproduce.py
```

On Windows PowerShell, use the virtual environment's interpreter directly.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code/reproduce.py
```

The script first verifies the 41 input and reference hashes listed in `data/SHA256SUMS`. A successful run writes `results/reproduction_report.json` with `"status": "pass"`, lists the 22 matched CSVs and four matched TeX table bodies, and writes the Figure 2 regret distribution to `figures/regret_distribution.pdf` and `figures/regret_distribution.png`. CSV values are checked against the frozen references at absolute tolerance `1e-12`. The four TeX table bodies must match exactly. The plotted distribution is regenerated from the selected contexts and their regrets.

The public [source ZIP](https://github.com/dqswordman/rag-decision-sufficiency/archive/refs/tags/v1.0.0.zip) can be used instead of cloning. Extract it, open the extracted repository directory, and start at `python -m venv .venv`. The archive checked on 11 November 2026 was 4,292,790 bytes. The published [verification record](https://github.com/dqswordman/rag-decision-sufficiency/releases/download/v1.0.0/reproduction_report.json) accompanies the tag.

## Locate the evidence

| Manuscript | Generated output | What to inspect |
| --- | --- | --- |
| Table 1 | `source/table_primary.tex` and `results/selection_overall.csv` | Three- and four-passage regret, target calls, and budget-3 answer scores |
| Table 2 | `source/table_paired.tex`, `results/selection_paired.csv` and `results/generation_paired.csv` | First policy minus second with 95% query-cluster bootstrap intervals |
| Table 3 | `source/table_pool.tex`, `results/nested_summary.csv` and `results/extension_summary.csv` | Original pool prefixes and the matched eight-to-twelve-passage extension |
| Table 4 | `source/table_geometry.tex` and `results/geometry_overall.csv` | Material negative marginals and positive second differences inside the passage budget |
| Figure 2 | `figures/regret_distribution.png` and `results/m8_decisions.csv` | Cumulative regret curves for four selection policies |
| Worked examples in Section 5.1 | `results/worked_examples_with_generation.json` | Archived query keys, observed probes, selected masks, and normalized-alias answer metrics |

Use [the result map](result-map.csv) to connect each result to its frozen input and protocol. [The asset manifest](asset-manifest.csv) and [data provenance](../DATA_PROVENANCE.md) identify the original measurements, the benchmark identifiers, and the model revisions. The numerical replay starts after target scoring and answer scoring. It uses the published aggregate scores keyed by context.

If an input hash fails, obtain a clean `v1.0.0` copy and run `python code/reproduce.py --verify-inputs` in the configured environment before retrying. For a result discrepancy, record the tag or commit, Python and package versions, command, and the mismatched file or table in the GitHub issue.
