# RouterBench-Mini V3: Learned Quality-Gap Routing

**English** | [简体中文](README.zh-CN.md) | [Latest version](../../../README.md)

V3 adds the Learned Cost-Aware Router and changes model selection from handcrafted scores into supervised learning. The main implementation is preserved at commit [`2f0510c`](https://github.com/ken-ab/routerbench-mini/tree/2f0510c).

## Protocol

The old 300 examples become development data. A new non-overlapping A set contains 150 examples: 50 text, 50 vision, and 50 tool. All estimators and thresholds use development data only.

Models remain `qwen3.5-35b-a3b` and `qwen3.5-397b-a17b`, with `temperature=0.2`, `max_tokens=256`, and thinking disabled.

## Learned Router

1. Cheap and Strong answer all 300 development examples.
2. Deterministic scoring creates `y = Strong correctness - Cheap correctness`.
3. TF-IDF unigrams/bigrams and 13 observable structured features represent each request.
4. `Ridge(alpha=0.1)` predicts Strong's quality advantage.
5. Five-fold OOF predictions select the Combined threshold `0.04986`.
6. The final estimator is refit on all development examples before A-set routing.

Only 18 labels are `+1`, 14 are `-1`, and 268 are zero. Reflection uses response-only confidence, format, and self-check with threshold `0.65` and one Strong review.

## Main Results

| Method | Accuracy | 95% CI | Avg cost (CNY) | Avg latency | Strong rate |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 75.33% | [68.00, 82.00] | 0.00024400 | 692 ms | 0.00% |
| Always Strong | **79.33%** | [72.67, 85.33] | 0.00066199 | 1,568 ms | 100.00% |
| Task-Aware | 78.67% | [72.00, 85.33] | 0.00051066 | 1,305 ms | 65.33% |
| Learned Combined | 78.67% | [72.00, 85.33] | **0.00037586** | **1,031 ms** | 36.00% |
| Reflection | 74.00% | [66.67, 81.33] | 0.00063306 | 1,732 ms | 66.00% |

Learned Combined trails Always Strong by one answer while using Strong on 36%. Reflection escalates 99 examples, repairs five Cheap errors, and changes seven correct Cheap answers into errors.

## Feature Ablation

| Features | Accuracy | Avg cost (CNY) | Avg latency | Strong rate |
|---|---:|---:|---:|---:|
| Combined | 78.67% | **0.00037586** | **1,031 ms** | **36.00%** |
| Structured-only | 78.67% | 0.00039014 | 1,117 ms | 46.67% |
| **Text-only** | **79.33%** | 0.00044537 | 1,191 ms | 54.67% |

Text-only matches Always Strong on A while reducing cost by 32.7%. However, A was used to choose among feature variants and is no longer a fully untouched confirmation set.

## Lessons

1. Only 32 development labels carry nonzero routing information.
2. Text-only beats Combined by one answer and may reflect selection noise.
3. Strong review can both repair and regress Cheap answers.
4. A fresh confirmation set is required after selecting Text-only.

V4 freezes Text-only, adds A to development, and evaluates once on a new B set.

## Artifacts

- [Main results](../../../results/qwen3.5-v3-study/test_summary.csv)
- [Learned ablation](../../../results/qwen3.5-v3-ablation/test_summary.csv)
- [Study metadata](../../../results/qwen3.5-v3-study/study_metadata.json)
- [Paired comparisons](../../../results/qwen3.5-v3-study/paired_comparisons.csv)

## Version Navigation

[V1](../v1/README.md) · [V2](../v2/README.md) · [V3](README.md) · [V4](../v4/README.md) · [V5](../v5/README.md)
