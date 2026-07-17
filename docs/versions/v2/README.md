# RouterBench-Mini V2: Observable Features and Calibration

**English** | [简体中文](README.zh-CN.md) | [Latest version](../../../README.md)

V2 removes V1's dataset-label leakage and redesigns both visual coverage and Reflection. The main implementation is preserved at commit [`c5dc5cc`](https://github.com/ken-ab/routerbench-mini/tree/c5dc5cc).

## Data and Settings

V2 keeps 300 examples and the 60/240 split, but changes vision to ScienceQA 40, ChartQA 20, OCR-VQA 20, and MMMU 20. Text and tool tasks remain at 100 each.

- Models: `qwen3.5-35b-a3b` and `qwen3.5-397b-a17b`
- `temperature=0.2`, `max_tokens=256`, thinking disabled
- Task-Aware threshold: `2.0`
- Reflection calibration threshold: `0.5`

## Methods

Task-Aware uses only observable question length, numeric mentions, math/logic/chart/OCR cues, image presence, choices, tool count, required arguments, and schema depth.

Reflection Full combines raw confidence, format validity, self-check, and 13 request features. Logistic regression with three-fold Platt scaling estimates `P(Cheap is correct)`, and Strong performs review-and-correct below the threshold.

## Main Results

| Method | Accuracy | Avg cost (CNY) | Avg latency | Strong rate |
|---|---:|---:|---:|---:|
| Always Cheap | 76.67% | 0.00024165 | 1,141 ms | 0.00% |
| Always Strong | 77.92% | 0.00065225 | 1,783 ms | 100.00% |
| **Task-Aware** | **80.00%** | 0.00052408 | 1,610 ms | 68.33% |
| Reflection Full | 76.67% | **0.00025260** | **1,174 ms** | 2.08% |

Task-Aware exceeds Always Strong on this one 240-example split. Reflection reaches 95.00% on the 60-example calibration set but falls to 76.67% on test and escalates only 5/240 requests.

## Reflection Ablation

| Variant | Accuracy | Avg cost (CNY) | Avg latency | Strong rate |
|---|---:|---:|---:|---:|
| Format-only | 76.67% | **0.00024165** | **1,141 ms** | 0.00% |
| Raw confidence | 76.67% | 0.00024524 | 1,149 ms | 0.42% |
| **Calibrated response-only** | **79.17%** | 0.00056005 | 2,060 ms | 59.58% |
| Full: response + 13 features | 76.67% | 0.00025260 | 1,174 ms | 2.08% |

Response-only outperforms Full, indicating that 13 additional features increase overfitting on only 60 calibration examples.

## Lessons

1. The same 60 examples fit the calibrator and select its threshold, creating optimistic bias.
2. Predicting Cheap error is not the same as predicting Strong benefit.
3. Handcrafted Task-Aware scores need independent validation.

V3 converts all 300 old examples into development data and learns the direct quality gap `Strong correct - Cheap correct` before evaluation on a fresh 150-example A set.

## Artifacts

- [Main results](../../../results/qwen3.5-v2-study/test_summary.csv)
- [Reflection ablation](../../../results/qwen3.5-v2-ablation/test_summary.csv)
- [Study metadata](../../../results/qwen3.5-v2-study/study_metadata.json)

## Version Navigation

[V1](../v1/README.md) · [V2](README.md) · [V3](../v3/README.md) · [V4](../v4/README.md) · [V5](../v5/README.md)
