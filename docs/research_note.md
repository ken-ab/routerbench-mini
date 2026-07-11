# RouterBench-Mini Research Note

## Research Question

Can a lightweight router reuse a cheaper multimodal foundation model on easy tasks and selectively invoke a stronger model while preserving accuracy under cost and latency constraints?

The study deliberately uses two models from the same Qwen 3.5 family. Both models support text, vision, and tools. This controls for provider and interface differences and makes model capacity, task difficulty, and routing policy the variables of interest.

## Protocol

- Dataset: 300 examples, balanced across text, vision, and tool use.
- Split: stratified 60-example validation set and 240-example held-out test set, seed 42.
- Cheap model: `qwen3.5-35b-a3b`.
- Strong model: `qwen3.5-397b-a17b`.
- Shared decoding: temperature 0, maximum output 256 tokens, thinking disabled.
- Shared interface: one prompt contract, JSON responses for reasoning/VQA, native function calling for tools.
- Metrics: accuracy, measured API token cost, observed end-to-end latency, escalation rate, and strong-model usage.
- Threshold selection: choose the lowest-cost reflection threshold within two percentage points of Always Strong validation accuracy; if none qualifies, choose the most accurate threshold and break ties by cost.

The test split is used once after selecting the Reflection Router threshold on validation. API responses are cached by model, task, prompt version, and decoding configuration.

## Headline Test Results

| Method | Accuracy | Avg. cost/task (CNY) | Avg. latency (ms) | Strong usage |
|---|---:|---:|---:|---:|
| Always Cheap | 0.8000 | 0.00023496 | 707.13 | 0.0000 |
| Always Strong | **0.8167** | 0.00063335 | 1411.50 | 1.0000 |
| Task-Aware Router | 0.8125 | 0.00044290 | 1062.50 | 0.4917 |
| Reflection Router | 0.7875 | 0.00057631 | 1316.75 | 0.3333 |

Task-Aware routing is the best observed trade-off. It saves 30.1% API cost and 24.7% latency relative to Always Strong while losing only 0.42 percentage points of accuracy.

## Category Findings

- Text: the strong model improves over cheap from 70.00% to 76.25%; Task-Aware reaches 75.00%.
- Vision: the strong model reaches 88.75%, but Task-Aware and cheap both reach 86.25%. The fixed vision difficulty rule did not recover the full strong-model gain.
- Tool use: cheap reaches 83.75%, higher than strong at 80.00%. This is evidence that a larger model is not uniformly better under a strict function-call scoring contract.

Across the 240 test tasks, both models are correct on 183 tasks, the strong model fixes 13 cheap-model errors, the strong model regresses 9 cheap-model successes, and both fail on 35 tasks. The modest 13-task upside explains why indiscriminate strong-model use provides only a small aggregate accuracy gain.

## Why Reflection Failed

The selected confidence threshold is 0.80. Because tool-call responses do not expose a calibrated confidence value, their fallback confidence causes every test tool task to escalate, while high-confidence text and vision errors are often accepted. The resulting policy has:

- 35 false accepts: the cheap answer is wrong but is not escalated.
- 67 unnecessary escalations: the cheap answer is correct but the strong model is still called.
- 51 total errors, compared with 44 for Always Strong and 45 for Task-Aware.

This is not evidence that reflection is intrinsically ineffective. It shows that self-reported confidence plus deterministic format checks is insufficiently calibrated for cross-task routing.

## Interpretation

The central finding is not simply that one model is better. Strong-model advantage is sparse and task dependent. A small amount of task structure captures much of that advantage, whereas an uncalibrated generic confidence signal does not. For this model pair and benchmark, model selection benefits more from knowing the task than from asking the model how confident it feels.

## Limitations and Next Step

The benchmark is intentionally small and uses one provider and model family. The fixed rule tiers are hand-authored, BFCL scoring checks the first canonical function call and required arguments, and latency is sensitive to remote service load. Token prices are experiment inputs and may become stale.

The next experiment should fit a calibrated router on validation data using task family, dataset, answer format validity, cheap-model confidence, and cheap-model latency. A logistic model or isotonic confidence calibrator would be sufficient; no large routing network is needed. It should then be frozen and evaluated on the same held-out test protocol.

Raw summary tables, threshold sweeps, the Pareto plot, metadata, and error counts are stored under `results/qwen3.5-study/`.
