# RouterBench-Mini V4: Confirmatory Evaluation

**English** | [简体中文](README.zh-CN.md) | [Latest version](../../../README.md)

V4 is not a new routing algorithm. It is a confirmatory evaluation of the feature choice made in V3. The published documentation is preserved at commit [`a9db8ef`](https://github.com/ken-ab/routerbench-mini/tree/a9db8ef).

## Protocol

- Development: the old 300 examples plus the 150-example A set, for 450 total.
- Confirmation: a fresh 150-example B set with 50 text, 50 vision, and 50 tool examples.
- Learned: frozen Text-only TF-IDF, `Ridge(alpha=0.1)`, five-fold threshold `0.02606`.
- Reflection: frozen response-only calibration with threshold `0.75`.
- Task-Aware: frozen V2 risk rules and threshold `2.0`.
- Decoding: `temperature=0.2`, `max_tokens=256`, thinking disabled.

B is used only after feature mode, estimator, and method selection are fixed. A and B have no exact fingerprint overlap, but A participates in V4 training; only B is the final untouched confirmation set.

## B-Set Results

| Method | Accuracy | 95% CI | Avg cost (CNY) | Avg latency | Strong rate |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 78.67% | [72.00, 84.68] | 0.00023762 | 1,178 ms | 0.00% |
| Always Strong | **83.33%** | [77.33, 89.33] | 0.00064448 | 2,619 ms | 100.00% |
| **Task-Aware** | **82.67%** | [76.67, 88.00] | 0.00050139 | **1,767 ms** | 66.00% |
| Learned Text-only | 80.00% | [73.33, 86.00] | **0.00043693** | 2,391 ms | 50.00% |
| Reflection | 80.00% | [73.33, 86.00] | 0.00047537 | 2,202 ms | 46.00% |

Task-Aware trails Always Strong by one answer while reducing average cost by 22.2%. Learned Text-only trails Strong by five answers, so V3's match does not reproduce. Reflection also reaches 80.00% but calls Cheap first and reviews 46% of requests.

## Interpretation

V3's 79.33% and V4's 80.00% use different tests and cannot show that 450 training examples beat 300. Always Strong itself moves from 79.33% on A to 83.33% on B.

The supported conclusion is narrower: **after freezing the method and feature representation, V3's Text-only match to Always Strong does not reproduce on B.**

Across A and B, unchanged baselines score 77.00% for Always Cheap, 81.33% for Always Strong, and 80.67% for Task-Aware. Task-Aware is 0.67 points below Strong while reducing cost by 22.5%, latency by 26.6%, and Strong usage to 65.67%.

## Lessons

1. Only 31 Strong-only and 21 Cheap-only labels exist among 450 development examples.
2. A small confirmation set cannot isolate training scale, test difficulty, and API variation.
3. Reflection still depends on noisy confidence and risky review corrections.
4. Legacy manifests lack source row IDs, pinned revisions, image IDs, and strict near-duplicate auditing.

V5 therefore rebuilds 3,200/800 examples from pinned revisions and adds Random/Oracle controls, Hard slices, and a strict freeze inventory.

## Artifacts

- [Main results](../../../results/qwen3.5-v4-study/test_summary.csv)
- [Study metadata](../../../results/qwen3.5-v4-study/study_metadata.json)
- [Research note](../../research_note.md)
- [Supervisor-style review](../../supervisor_review.zh-CN.md)

## Version Navigation

[V1](../v1/README.md) · [V2](../v2/README.md) · [V3](../v3/README.md) · [V4](README.md) · [V5](../v5/README.md)
