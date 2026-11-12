# Complex Utility, Simple Calibration: Decision Sufficiency in RAG Evidence Selection

Wenzhang Du · WSDM 2027 Findings · Accepted manuscript

The artifact replays how an additive model proposes a RAG evidence set and how a target score changes the final choice. It includes the numerical inputs, policy code, paired comparisons, and reference outputs for CPU reproduction.

[Paper companion](https://dqswordman.github.io/technical-reports/decision-sufficiency/) · [中文伴读](https://dqswordman.github.io/zh/technical-reports/decision-sufficiency/) · [Reproduction guide](docs/reproduction.md) · [Release v1.0.0](https://github.com/dqswordman/rag-decision-sufficiency/releases/tag/v1.0.0)

## Follow one archived decision

With Python 3.9 or newer, run this example from a fresh copy of the default branch. It uses the standard library and the included archive.

```sh
git clone --depth 1 https://github.com/dqswordman/rag-decision-sufficiency.git
cd rag-decision-sufficiency
python code/inspect_case.py
```

On the MuSiQue question about King Arthur, P1 estimates utility **1.574** for passages **{1, 5}**. The measured pair scores **0.206**. The check returns passage **{1}**, the best measured context at **0.918**. The program prints both the chosen contexts and their archived answer scores. The pair produced the exact short answer on this question, while the singleton scored 0.444 token F1. [Example details](docs/reproduction.md#try-one-archived-decision)

## Results at a glance

[![Mean normalized regret at the nine-call budget and after one optional check. BO9 0.213, P1 0.272, and P1 plus one check 0.162.](https://dqswordman.github.io/assets/reports/decision-sufficiency/mean-regret.png)](https://dqswordman.github.io/technical-reports/decision-sufficiency/#results)

The main comparison fixes the pool at eight candidates and allows up to three selected passages. Over **1,440 matched question–pool–generator settings**, the same nine target scores yield mean regret **0.213** when the policy keeps the best measured context and **0.272** when it commits to the additive proposal. Scoring that proposal when it is new lowers mean regret to **0.162** at **9.61** calls. The paired token F1 gain against P1 is **0.038** with a 95% interval of **[0.020, 0.056]**.

A second operating point uses selected pair measurements. At the same mean cost of **11.43** calls, keeping the best measured context reaches **0.122** regret and maximizing the fitted quadratic reaches **0.213**. [Table 1](reference/tables/table_primary.tex) · [Paired comparisons](reference/tables/table_paired.tex)

## Reproduce the results

Use [`v1.0.0`](https://github.com/dqswordman/rag-decision-sufficiency/releases/tag/v1.0.0) at `7b0f639a207b1826915227aaff2eef3addea499d` for the paper replay. A clean CPU run with Python 3.9.21 verified **41 input hashes**, matched **22 result CSVs** and **four TeX table bodies**, and regenerated the regret distribution. The replay reads archived scores. The original benchmark and model revisions are recorded in the provenance file.

[Run the tagged replay](docs/reproduction.md#replay-the-paper-results) · [Published verification record](https://github.com/dqswordman/rag-decision-sufficiency/releases/download/v1.0.0/reproduction_report.json)

The default branch is maintained for the example and documentation. Use the release tag when citing or replaying the frozen analysis.

## Trace the results

The result map connects the four manuscript tables, the regret distribution, and the two worked examples to the frozen settings and output files. The version manifest records the paper, code, and web companion versions. [Result map](docs/result-map.csv) · [Version manifest](docs/release-manifest.json)

## Citation and reuse

The original code is available under the [MIT License](LICENSE). Original numerical measurements are under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with benchmark identifiers governed by the upstream terms in [data provenance](DATA_PROVENANCE.md). The [asset manifest](docs/asset-manifest.csv) records sources, versions, and checksums.

Cite Wenzhang Du, *Complex Utility, Simple Calibration: Decision Sufficiency in RAG Evidence Selection*, WSDM 2027 Findings, for the research. Cite the [software artifact](CITATION.cff) at `v1.0.0` when using the archived inputs or CPU replay.
