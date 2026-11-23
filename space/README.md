---
title: Decision Sufficiency in RAG Evidence Selection
emoji: 📄
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
license: other
short_description: Trace how one more passage-set score changes a RAG decision
tags:
  - rag
  - reproducibility
  - evaluation
---

# Complex Utility, Simple Calibration

Wenzhang Du · WSDM 2027

[![GitHub](https://img.shields.io/badge/GitHub-Code-1B4D3E?style=flat-square&labelColor=263238)](https://github.com/dqswordman/rag-decision-sufficiency)
[![Paper companion](https://img.shields.io/badge/Paper_companion-263238?style=flat-square)](https://dqswordman.github.io/technical-reports/decision-sufficiency/)
[![中文伴读](https://img.shields.io/badge/%E4%B8%AD%E6%96%87%E4%BC%B4%E8%AF%BB-263238?style=flat-square)](https://dqswordman.github.io/zh/technical-reports/decision-sufficiency/)

Choose a saved question to follow an evidence decision. Compare the model’s predicted score for a passage combination with its measured score, then inspect the answer generated from each context. The two cases show why a better likelihood score need not produce a better answer.

The GitHub repository maintains the code and documentation. The Space mirrors its default branch and includes a separate ZIP of the frozen release for the full CPU replay. The [release manifest](docs/release-manifest.json) records the artifact version and the published verification. The [reproduction guide](docs/reproduction.md) gives the commands.

Original code is available under the [MIT License](LICENSE). Original numerical measurements are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Benchmark identifiers retain the upstream terms in [data provenance](DATA_PROVENANCE.md). See [CITATION.cff](CITATION.cff) for the software citation.
