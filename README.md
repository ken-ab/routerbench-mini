# RouterBench-Mini: Cost-Aware Model Reuse for Multimodal Agents

**English** | [简体中文](README.zh-CN.md)

This is a small personal experimental report that cannot independently support a paper contribution. It documents one step in my transition from engineering practice back toward research. The root README now presents only the latest V5 frozen evaluation; the designs, results, and failure analyses from V1 through V4 remain available in the [version index](#version-index).

RouterBench-Mini studies a model-selection question: **when is a cheaper model sufficient, and when does a stronger model provide enough quality gain to justify its cost?** V5 compares fixed-model baselines, rule-based routing, learned routing, and post-response Reflection under the same Qwen 3.5 multimodal model family, prompts, decoding settings, and deterministic scoring.

## V5 Experimental Design

### Models and API Settings

| Role | Model | Input/output price (CNY per million tokens) |
|---|---|---:|
| Cheap | `qwen3.5-35b-a3b` | 0.4 / 3.2 |
| Strong | `qwen3.5-397b-a17b` | 1.2 / 7.2 |

Both models support text, images, and tool calls. Every request uses the same settings:

| Parameter | Value |
|---|---|
| `temperature` | `0.2` |
| `top_p` | `0.8` |
| `max_tokens` | `256` |
| Thinking | Disabled |
| System prompt | No separate system message |
| Timeout / retries | 120 seconds / up to 4 attempts |
| Cheap and Strong prompts | Identical |

Math uses final-number matching, multiple choice uses option matching, open VQA uses normalized exact matching or 5% numeric tolerance, and tool calls use function-name and required-argument matching. Gold answers are used only for offline scoring and are never exposed to the router or models.

### Data and Split

V5 rebuilds the data from pinned revisions rather than reusing the historical 300 examples or the V3/V4 A/B sets. The 3,200-example development set is used for cached model outputs, five-fold out-of-fold training, ablation, and threshold selection. The 800-example test set is accessed only after all methods are frozen.

| Dataset | Task | Dev standard | Dev hard | Dev total | Test standard | Test hard | Test total |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | Math reasoning | 400 | 25 | 425 | 80 | 25 | 105 |
| CommonsenseQA | Text multiple choice | 300 | 15 | 315 | 60 | 15 | 75 |
| BBH | Logic/short-answer reasoning | 300 | 30 | 330 | 60 | 30 | 90 |
| ScienceQA | Visual multiple choice | 400 | 15 | 415 | 80 | 15 | 95 |
| MMMU | Multidisciplinary visual reasoning | 200 | 25 | 225 | 40 | 25 | 65 |
| ChartQA | Chart question answering | 200 | 15 | 215 | 40 | 15 | 55 |
| OCR-VQA | OCR visual question answering | 200 | 15 | 215 | 40 | 15 | 55 |
| BFCL Simple | Single-tool calling | 500 | 20 | 520 | 100 | 20 | 120 |
| BFCL Multiple | Multi-tool calling | 500 | 40 | 540 | 100 | 40 | 140 |
| **Total** |  | **3,000** | **200** | **3,200** | **600** | **200** | **800** |

| Split | Text | Vision | Tool | Total |
|---|---:|---:|---:|---:|
| Development | 1,070 | 1,070 | 1,060 | 3,200 |
| Test | 270 | 270 | 260 | 800 |

Each manifest row records its source ID, split, dataset revision, image ID/SHA-256, template group, BFCL schema group, difficulty rationale, and predefined fold. All prohibited overlap counts are zero under exact fingerprints, source IDs, images, templates, BFCL schemas, and near-duplicate checks.

## Routing Methods

### Always Cheap and Always Strong

- **Always Cheap:** sends every request to Cheap and provides the cost floor.
- **Always Strong:** sends every request to Strong and provides the single-model accuracy baseline.

### Frozen Task-Aware

Task-Aware reads only request features observable at inference time. V5 freezes the V2 threshold of 2.0 and the following rules:

- math cue `+3`, logic cue `+2`, at least 50 words `+1`;
- image with chart/OCR cue `+2`, at least three numbers `+1`, at least five choices `+1`;
- at least three tools `+2`, two tools `+1`, at least four required arguments `+1`, schema depth at least four `+1`.

A score of at least 2.0 selects Strong; otherwise it selects Cheap.

### Learned Router

Cheap and Strong answer every development example, and deterministic scoring produces:

```text
y = Strong correctness - Cheap correctness
```

`y=+1` means Strong repairs Cheap, `y=-1` means escalation regresses, and `y=0` means equal correctness. The 3,200 examples contain only 288 positive and 94 negative quality-gap labels; the remaining 2,818 labels are zero.

V5 compares three representations:

- **Text-only:** TF-IDF unigrams/bigrams with up to 1,500 dimensions;
- **Structured-only:** 13 observable image, choice, numeric, cue, and tool-schema features;
- **Combined:** TF-IDF concatenated with the 13 structured features.

All variants use `Ridge(alpha=0.1)`. Five-fold OOF predictions select a threshold, then the estimator is refit on all 3,200 examples and frozen. The primary Combined threshold is `-0.308617`.

### Reflection Router

Reflection calls Cheap first and extracts self-reported confidence, format validity, and self-check. `LogisticRegression(C=0.5)` plus sigmoid calibration estimates `P(Cheap is correct)`. If the probability falls below the frozen `0.75` threshold, or format/self-check fails, Strong sees the Cheap candidate and performs one `review_and_correct` call. Escalation cannot loop.

## V5 Main Results

| Method | Accuracy | Avg cost (CNY) | Avg latency | Strong rate |
|---|---:|---:|---:|---:|
| Always Cheap | 68.25% | 0.00029186 | 722.5 ms | 0.00% |
| Always Strong | **72.75%** | 0.00079930 | 1,318.2 ms | 100.00% |
| Frozen Task-Aware | 72.12% | 0.00068789 | 1,133.6 ms | 69.88% |
| Learned Combined | 72.00% | 0.00078259 | 1,290.0 ms | 95.88% |
| Reflection | 72.00% | 0.00067258 | 1,750.9 ms | 67.38% |

Always Strong remains highest at 582/800 correct. Frozen Task-Aware answers 577 correctly, while Learned Combined and Reflection each answer 576. The one-answer difference is not significant, but Task-Aware is simple, uses Strong on 69.88%, and is much faster than Reflection. Learned Combined routes almost every example to Strong without providing a useful cost advantage.

## Learned Feature Ablation

| Features | Accuracy | Strong rate |
|---|---:|---:|
| Text-only | **72.00%** | 95.12% |
| Structured-only | 71.75% | 80.38% |
| Combined | **72.00%** | 95.88% |

Text-only and Combined both answer 576 examples correctly. Combined spends six additional Strong calls without gaining accuracy, so the 13 structured features provide no measurable increment beyond text under the current linear representation.

## Random and Oracle Controls

| Method | Accuracy | Strong rate | Interpretation |
|---|---:|---:|---|
| Random@Learned Rate | 72.52% | 95.88% | Random Strong assignment at Learned's budget |
| Learned Combined | 72.00% | 95.88% | Actual learned routing |
| Oracle@Learned Rate | 76.25% | 95.88% | Answer-aware upper bound at the same budget |
| Random@Reflection Rate | 70.69% | 67.38% | Random escalation at Reflection's budget |
| Reflection | 72.00% | 67.38% | Actual post-response escalation |
| Oracle@Reflection Rate | 73.62% | 67.38% | Oracle upper bound at the same budget |
| Global Oracle | 76.25% | 8.00% | Per-example best of the two cached answers |

Learned trails its matched Random baseline by 0.52 percentage points. Reflection exceeds its matched Random baseline by 1.31 points, with randomization `p=0.0099`. The Global Oracle shows that perfectly identifying the 64 Strong-beneficial examples would reach 76.25% with only an 8% Strong rate.

## Reflection Diagnostics

Reflection Strong usage by task family is:

| Task family | Strong rate |
|---|---:|
| Text | 99.63% |
| Vision | 100.00% |
| Tool | 0.00% |

A successfully parsed tool call receives a fixed confidence of `0.75`, while escalation uses `p < 0.75`; every tool example is therefore accepted. Almost every text and vision probability falls below the threshold. The current Reflection policy behaves more like task-family routing induced by confidence construction than robust per-example uncertainty estimation.

## Standard vs. Hard

| Method | Standard | Hard |
|---|---:|---:|
| Always Cheap | 74.50% | 49.50% |
| Always Strong | 79.50% | 52.50% |
| Frozen Task-Aware | 79.00% | 51.50% |
| Learned Combined | 78.50% | 52.50% |
| Reflection | 78.33% | 53.00% |

Every method drops substantially on Hard, and Always Strong exceeds Always Cheap by only three points. Hard-example performance is therefore limited by both routing errors and Strong's own capability ceiling.

## Conclusions and Next Steps

V5 does not show that Learned routing or Reflection stably beats the simple rule. Combined uses Strong on 95.88% and still trails Task-Aware. Reflection beats its matched Random baseline, but its task-family escalation pattern is extreme, latency is high, and accuracy trails Task-Aware by one answer. Frozen Task-Aware remains the most defensible cost-aware policy in this experiment.

Next priorities:

1. Actively collect examples with nonzero Cheap/Strong quality gaps instead of adding mostly `y=0` data.
2. Optimize explicit utility: quality gain minus API cost, latency, and regression risk.
3. Add image-content representations for visual tasks rather than relying on question text and `has_image`.
4. Replace fixed tool confidence with a continuous learned signal and test `<` versus `<=` boundaries.
5. Add independent verifier, consistency, or repeated-sampling signals instead of relying on self-reported confidence.
6. Freeze all methods and thresholds again before evaluating on a third untouched confirmation set.

## Reproduction and Documentation

- [Full V5 experimental report](docs/v5_large_scale_report.zh-CN.md)
- [Audit of the legacy implementation](docs/v5_large_scale_audit.zh-CN.md)
- [V5 protocol configuration](configs/v5_large_scale.yaml)
- [Model configuration](configs/models.qwen_v5.yaml)
- [Data builder](scripts/build_v5_data.py)
- [Development and freeze runner](scripts/run_v5_phase1.py)
- [Frozen test runner](scripts/run_v5_phase2.py)

## Version Index

The root README presents only the latest version. Complete records for each stage are preserved here:

| Version | Main change | English | 中文 |
|---|---|---|---|
| V1 | Dataset-tier rules and raw self-reported confidence | [README](docs/versions/v1/README.md) | [README](docs/versions/v1/README.zh-CN.md) |
| V2 | Observable features, probability calibration, Reflection ablation | [README](docs/versions/v2/README.md) | [README](docs/versions/v2/README.zh-CN.md) |
| V3 | TF-IDF/Ridge quality-gap routing, five-fold OOF, feature ablation | [README](docs/versions/v3/README.md) | [README](docs/versions/v3/README.zh-CN.md) |
| V4 | 450-example development set and fresh B-set confirmation | [README](docs/versions/v4/README.md) | [README](docs/versions/v4/README.zh-CN.md) |
| V5 | 3,200/800 frozen protocol, Random/Oracle controls, difficulty slices | [README](docs/versions/v5/README.md) | [README](docs/versions/v5/README.zh-CN.md) |
