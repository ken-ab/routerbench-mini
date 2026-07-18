# RouterBench-Mini V1: Basic Routing Framework

**English** | [简体中文](README.zh-CN.md) | [Latest version](../../../README.md)

V1 establishes the Cheap/Strong pool, three task families, deterministic scoring, and four baseline routes. Its main implementation is preserved at commit [`4cf8f44`](https://github.com/ken-ab/routerbench-mini/tree/4cf8f44).

## Data and Settings

| Family | Datasets | Count |
|---|---|---:|
| Text | GSM8K 40, CommonsenseQA 30, BBH 30 | 100 |
| Vision | ScienceQA 80, ChartQA 10, OCR-VQA 10 | 100 |
| Tool | BFCL Simple 50, BFCL Multiple 50 | 100 |

The 300 examples are stratified into 60 validation and 240 test examples. Models are `qwen3.5-35b-a3b` and `qwen3.5-397b-a17b`, with `temperature=0`, `max_tokens=256`, thinking disabled, and server-default `top_p`.

## Methods

- **Always Cheap** and **Always Strong** provide model and cost boundaries.
- **Task-Aware** reads a manifest `rule_tier`, effectively assigning known datasets such as GSM8K to a fixed model.
- **Reflection** calls Cheap first, then checks format, self-reported confidence, and self-check. Validation selects a threshold of `0.8`.

A parsed tool call receives fixed confidence `0.75`; math and multiple-choice confidence is primarily self-reported by the model.

## Results

| Method | Accuracy | Avg cost (CNY) | Avg latency | Strong rate |
|---|---:|---:|---:|---:|
| Always Cheap | 80.00% | 0.00023496 | 707 ms | 0.00% |
| Always Strong | **81.67%** | 0.00063335 | 1,412 ms | 100.00% |
| Task-Aware | 81.25% | 0.00044290 | 1,063 ms | 49.17% |
| Reflection | 78.75% | 0.00057631 | 1,317 ms | 33.33% |

Task-Aware trails Always Strong by only one answer while cutting Strong usage to 49.17%. Reflection is below Always Cheap while approaching Always Strong's cost.

## Lessons

1. `rule_tier` leaks dataset identity and cannot represent deployment-time routing.
2. Raw confidence is not comparable across task formats.
3. The 0.8 threshold escalates almost every fixed-confidence tool call while accepting some high-confidence wrong math answers.
4. V1 does not test whether Strong review can damage a correct Cheap answer.

V2 therefore removes dataset-label routing, introduces observable features, and calibrates Cheap correctness before review-and-correct.

## Artifacts

- [Test summary](../../../results/qwen3.5-study/test_summary.csv)
- [Validation summary](../../../results/qwen3.5-study/validation_summary.csv)
- [Study metadata](../../../results/qwen3.5-study/study_metadata.json)

## Version Navigation

[V1](README.md) · [V2](../v2/README.md) · [V3](../v3/README.md) · [V4](../v4/README.md) · [V5](../v5/README.md)
