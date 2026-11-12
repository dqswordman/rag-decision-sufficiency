# Data provenance

## Benchmarks and pool construction

The eight-passage study uses 60 held-out questions from each benchmark. Each question has two fixed candidate pools and four generator evaluations. Track A contains annotated support and fills the remaining positions using BGE. Track B fuses dataset-local BM25 and BGE ranks by reciprocal rank fusion. The 12-passage extension uses matched Track B pools for 20 of the MuSiQue questions and two generators. Each fixed passage representation was capped at 256 Qwen2.5 tokens.

| Benchmark | Release used | Upstream source and terms |
| --- | --- | --- |
| HotpotQA (Yang et al., 2018) | Distractor development split | [Dataset and CC BY-SA 4.0 terms](https://github.com/hotpotqa/hotpot#license) |
| 2WikiMultiHopQA (Ho et al., 2020) | Development split | [Dataset source](https://github.com/Alab-NII/2wikimultihop) and [Apache-2.0 repository license](https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE) |
| MuSiQue (Trivedi et al., 2022) | MuSiQue-Answerable development split | [Dataset source](https://github.com/StonyBrookNLP/musique) and [CC BY 4.0 license](https://github.com/StonyBrookNLP/musique/blob/main/LICENSE) |

## Model snapshots

The generator and tokenizer used the same pinned revision in each row. The original teacher-forced evaluation used bfloat16 and batch size one. The generator manifest records the prompt version. Generated answers used greedy decoding with the configured stop tokens and a cap of 64 new tokens. The replay reads the archived values on CPU.

| `model_key` | Generator | Pinned revision |
| --- | --- | --- |
| `qwen` | Qwen2.5-7B-Instruct | [`a09a35458c702b33eeacc393d103063234e8bc28`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28) |
| `gemma` | Gemma-3-4B-IT | [`093f9f388b31de276ce2de164bdc2081324b9767`](https://huggingface.co/google/gemma-3-4b-it/tree/093f9f388b31de276ce2de164bdc2081324b9767) |
| `mistral` | Mistral-7B-Instruct-v0.3 | [`c170c708c41dac9275d15a8fff4eca08d52bab71`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/c170c708c41dac9275d15a8fff4eca08d52bab71) |
| `phi` | Phi-4-Mini-Instruct | [`cfbefacb99257ffa30c83adab238a50856ac3083`](https://huggingface.co/microsoft/Phi-4-mini-instruct/tree/cfbefacb99257ffa30c83adab238a50856ac3083) |

The extension used the original BGE scoring revision for the appended ranks. The frozen extension records preserve the source digests. [BGE reranker revision](https://huggingface.co/BAAI/bge-reranker-v2-m3/tree/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e)

## Archive schema

The main context key is `(dataset_key, query_id, universe_name, model_key, selected_mask)`. `A_m8` and `B_m8` name the fixed pool constructions. A mask encodes membership using zero-based candidate positions. For example, mask `5` selects positions 0 and 2. Mask `0` is the empty context. Mean reference-answer log probability for a mask, minus the score for mask `0`, gives the centered utility. Equal utility is resolved by the smallest integer mask.

| Archive | Unit and fields |
| --- | --- |
| `data/m8_raw_mean_logprobs.csv` | 1,440 complete score arrays with `mask_0` through `mask_255`. `universe_hash` identifies the serialized pool and `raw_sha256` identifies the archived source cube. |
| `data/m6_calibration_raw_mean_logprobs.csv` | 1,080 calibration arrays of 64 mask scores. The 18 original cell IQRs and six fixed Phi values are verified during replay. |
| `data/adaptive_discovery_risk.csv` | 1,440 earlier no-context scores for recovering the median risk threshold. |
| `data/generation_context_scores.csv` | One score row per distinct evaluated context. `token_f1` and `exact_match` are the archived normalized-alias metrics. `generated_token_count` and `origin` record decoding length and whether the context output was reused. |
| `data/legacy_context_scores.csv` | Scores for the 6,390 identical context keys with an archived historical decode, used to audit changes to stopping behavior. |
| `data/v4_generated_selected_sets.csv` | Historical selected masks keyed by original method role. The replay checks these masks against the recomputed policies before joining metrics. |
| `data/extension_raw_mean_logprobs.csv` | 40 query–model pairs, each with all 299 feasible masks for 12 candidates and a budget of three. |
| `data/extension_candidate_ranks.csv` | The 12 canonical positions and RRF ranks per extension query. The first eight positions are shared by the two widths. |
| `data/extension_context_scores.csv` | F1 and exact match for 284 distinct extension contexts, keyed by `(model_key, query_id, mask)`. |

The original experimental settings are in `data/revision_protocol.json`, `data/extension_protocol.json`, and `frozen/protocol/`. The extension freeze records retain hashes from the original manifests and full inputs. The checksums in `data/SHA256SUMS` identify the files distributed here, including the selected-column exports. The query-cluster bootstrap uses 10,000 resamples with the seed and clustering unit specified in the revision protocol.

The numerical replay begins after neural scoring and answer scoring. It uses archived aggregate scores per context. The original benchmarks supply passages and reference answers, and the model publishers supply weights under their respective terms.

## Reuse

Wenzhang Du releases the original numerical measurements and annotations in this repository under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Benchmark identifiers retain the upstream terms in the table above. Original replay code is distributed under the [MIT License](LICENSE).
