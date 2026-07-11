# RouterBench-Mini V2 Research Note

## Research Question

Can observable task features and validation-calibrated response signals select between a cheap and strong multimodal foundation model under accuracy, cost, and latency constraints? When escalation occurs, can Strong review a Cheap candidate without destroying correct answers?

## Protocol

- 300 tasks: 100 text, 100 vision, and 100 tool use.
- Vision mix: 40 ScienceQA, 20 ChartQA, 20 OCR-VQA, and one single-image MCQ from each of 20 MMMU subjects.
- Stratified split: 60 validation and 240 held-out test examples, seed 42.
- Cheap: `qwen3.5-35b-a3b`.
- Strong: `qwen3.5-397b-a17b`.
- Shared decoding: temperature 0.2, maximum output 256, thinking disabled.
- Shared prompt and structured output contract; native function calling for tools.
- Metrics: accuracy, measured token cost, observed latency, escalation, and strong usage.

Task-Aware uses only request-time observable features. Reflection fits a cross-validated Platt calibrator on validation-only Cheap responses. Thresholds maximize validation accuracy and break ties by cost and strong usage. No test labels are used for threshold selection.

## Main Test Results

| Method | Accuracy | Avg. cost/task (CNY) | Avg. latency (ms) | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 0.7667 | 0.00024165 | 1141.09 | 0.0000 |
| Always Strong | 0.7792 | 0.00065225 | 1782.91 | 1.0000 |
| Task-Aware Router | **0.8000** | 0.00052408 | 1610.24 | 0.6833 |
| Full Calibrated Reflection | 0.7667 | 0.00025260 | 1173.84 | 0.0208 |

Task-Aware is the best main method. Relative to Always Strong, it improves accuracy by 2.08 percentage points, reduces average cost by 19.7%, and reduces latency by 9.7%.

## Category Results

| Category | Cheap | Strong | Task-Aware | Full Reflection |
|---|---:|---:|---:|---:|
| Text | 68.75% | 76.25% | 76.25% | 68.75% |
| Vision | 76.25% | 77.50% | 81.25% | 76.25% |
| Tool | **85.00%** | 80.00% | 82.50% | **85.00%** |

Strong capacity helps text, while Cheap is better under the strict tool-call scorer. Observable routing benefits from this non-monotonic model relationship.

## Reflection Ablation

| Variant | Accuracy | Avg. cost | Avg. latency | Escalation |
|---|---:|---:|---:|---:|
| Format only | 76.67% | 0.00024165 | 1141 ms | 0.00% |
| Raw confidence | 76.67% | 0.00024524 | 1149 ms | 0.42% |
| Calibrated response only | **79.17%** | 0.00056005 | 2060 ms | 59.58% |
| Full response + task features | 76.67% | 0.00025260 | 1174 ms | 2.08% |

Response-only calibration is the strongest Reflection variant and exceeds Always Strong by 1.25 percentage points at 14.1% lower API cost. Its latency is higher because Cheap and Strong review are sequential. The full feature calibrator achieved 95% validation accuracy but did not generalize; with only 60 calibration examples, the higher-dimensional feature set overfit.

## Review-and-Correct Counterfactual

For each escalated response-only example, compare the observed review result with a counterfactual that blindly substitutes the independently generated Strong answer:

| Policy | Correct among 143 escalations | Beneficial | Harmful |
|---|---:|---:|---:|
| Blind Strong replacement | 113 | 14 | 8 |
| Review-and-correct | 113 | 11 | 5 |

Review-and-correct preserves the same number of correct final answers while reducing harmful escalation by 3 cases, or 37.5%. It is more conservative, but that conservatism also loses 3 beneficial corrections. Of 143 reviews, Strong kept the Cheap candidate 119 times and changed it 24 times.

The change therefore addresses the requested failure mode partially, not completely. A stronger future design would train an explicit candidate-verdict model or use a separate adjudication set; it should not be tuned on the current test results.

## Model Disagreement

Across the 240 test tasks:

- Both models correct: 170.
- Strong fixes Cheap: 17.
- Cheap is correct while Strong is wrong: 14.
- Both wrong: 39.

Only 31 tasks distinguish the models, so routing quality depends on identifying a small and asymmetric benefit region. Always Strong is not a reliable oracle, especially for tools.

## Limitations

- One provider, one model family, and 300 examples.
- Only 60 validation examples for thresholding and probability calibration.
- The current correctness probability is an empirical estimate, not a literal true probability.
- API latency includes remote service variance.
- Tool scoring checks one canonical call and required arguments.
- One OCR-VQA item rejected by both endpoints before generation is deterministically replaced by the next source item and never scored.
- The response-only ablation was examined after the main test and must be treated as analysis, not a newly selected headline policy.

All main artifacts are under `results/qwen3.5-v2-study/`; ablation and counterfactual tables are under `results/qwen3.5-v2-ablation/`.
