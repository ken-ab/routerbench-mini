# RouterBench-Mini V5: 3,200/800 Frozen Evaluation

**English** | [简体中文](README.zh-CN.md) | [Latest version](../../../README.md)

V5 tests whether V3/V4 instability is mainly caused by insufficient development data. It rebuilds 3,200 development and 800 independent test examples from pinned revisions, then freezes manifest hashes, source code, prompts, model configuration, estimators, and thresholds before test access.

## Data and Settings

- Development: 3,200 examples; Text 1,070, Vision 1,070, Tool 1,060; Standard 3,000, Hard 200.
- Test: 800 examples; Text 270, Vision 270, Tool 260; Standard 600, Hard 200.
- Datasets: GSM8K, CommonsenseQA, BBH, ScienceQA, MMMU, ChartQA, OCR-VQA, BFCL Simple/Multiple.
- Models: `qwen3.5-35b-a3b` and `qwen3.5-397b-a17b`.
- Decoding: `temperature=0.2`, `top_p=0.8`, `max_tokens=256`, thinking disabled.
- Fold IDs are predefined in the manifest; test data does not participate in vocabulary, feature, threshold, or method selection.

## Methods

- **Task-Aware:** frozen risk threshold 2.0 using only observable request structure.
- **Learned:** `y=Strong correctness-Cheap correctness`; Text-only, Structured-only, and Combined TF-IDF/13-feature representations with `Ridge(alpha=0.1)`.
- **Reflection:** confidence, format, and self-check feed logistic regression plus sigmoid calibration; threshold 0.75 and at most one Strong review.
- **Controls:** matched-rate Random, matched-budget Oracle, and Global Oracle.

Development contains 288 `+1`, 94 `-1`, and 2,818 zero labels. The Combined threshold is `-0.308617`; the Reflection threshold is `0.75`.

## Main Results

| Method | Accuracy | Avg cost (CNY) | Avg latency | Strong rate |
|---|---:|---:|---:|---:|
| Always Cheap | 68.25% | 0.00029186 | 722.5 ms | 0.00% |
| Always Strong | **72.75%** | 0.00079930 | 1,318.2 ms | 100.00% |
| Frozen Task-Aware | 72.12% | 0.00068789 | 1,133.6 ms | 69.88% |
| Learned Combined | 72.00% | 0.00078259 | 1,290.0 ms | 95.88% |
| Reflection | 72.00% | 0.00067258 | 1,750.9 ms | 67.38% |

Always Strong is highest. Task-Aware answers 577/800 correctly, while Learned Combined and Reflection each answer 576/800. Task-Aware leads both by one answer with a simpler and interpretable pre-response policy.

## Main Diagnostics

- Text-only and Combined both reach 72.00%; structured features add no accuracy.
- Learned uses Strong on 95.88% but trails Random@Learned at 72.52%.
- Reflection exceeds Random@Reflection's 70.69% by 1.31 points.
- Global Oracle reaches 76.25% with only 8% Strong use, so the core challenge is identifying 64 Strong-beneficial examples.
- Reflection Strong rates are 99.63% for Text, 100% for Vision, and 0% for Tool, exposing the fixed tool-confidence boundary.
- Reflection Standard/Hard accuracy is 78.33%/53.00%.

## Conclusion

More development data does not make Learned or Reflection routing stably superior to Task-Aware. Reflection beats its matched Random baseline but has an extreme escalation pattern and high latency; Frozen Task-Aware remains the most defensible cost-aware policy.

## Artifacts

- [Latest English README](../../../README.md)
- [Full V5 report](../../v5_large_scale_report.zh-CN.md)
- [Legacy implementation audit](../../v5_large_scale_audit.zh-CN.md)
- [V5 protocol](../../../configs/v5_large_scale.yaml)

## Version Navigation

[V1](../v1/README.md) · [V2](../v2/README.md) · [V3](../v3/README.md) · [V4](../v4/README.md) · [V5](README.md)
