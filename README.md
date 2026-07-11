# RouterBench-Mini: Cost-Aware Model Reuse for Multimodal Agents

[中文说明](README.zh-CN.md)

RouterBench-Mini studies when a multimodal agent should reuse a cheaper model and when it should invoke a stronger one. Two models from the same Qwen 3.5 family solve text, vision, and tool-use tasks under one prompt, decoding, scoring, and measured-cost pipeline.

The current version adds a learned quality-gap router, out-of-fold threshold selection, two disjoint held-out replications, paired bootstrap intervals, and a stricter analysis of review-and-correct.

## Confirmatory Result

![V4 held-out accuracy-cost trade-off](results/qwen3.5-v4-study/pareto.png)

V4 is the final confirmatory set: 450 earlier examples are used for router development, while its 150 examples are fingerprint-disjoint and untouched until evaluation.

| Method | Accuracy | 95% bootstrap CI | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 78.67% | [72.00, 84.68] | 0.00023762 | 1,178 ms | 0.00% |
| Always Strong | **83.33%** | [77.33, 89.33] | 0.00064448 | 2,619 ms | 100.00% |
| **Handcrafted Task-Aware** | **82.67%** | [76.67, 88.00] | 0.00050139 | **1,767 ms** | 66.00% |
| Learned Cost-Aware | 80.00% | [73.33, 86.00] | **0.00043693** | 2,391 ms | 50.00% |
| Calibrated Reflection | 80.00% | [73.33, 86.00] | 0.00047537 | 2,202 ms | 46.00% |

The frozen Task-Aware baseline is the most robust trade-off. It is 0.67 percentage points below Always Strong on V4, with a paired 95% difference interval of [-2.67, +1.33] points, while reducing cost by **22.2%** and observed latency by **32.5%**.

## Replication, Not a Best-Run Claim

V3 and V4 each contain 150 new tasks with 50 text, 50 vision, and 50 tool-use examples. The frozen baselines can be pooled because their policies did not change between replications:

| Frozen method | Accuracy over 300 held-out tasks | Avg. cost | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 77.00% | 0.00024081 | 935 ms | 0.00% |
| Always Strong | 81.33% | 0.00065324 | 2,094 ms | 100.00% |
| **Handcrafted Task-Aware** | **80.67%** | **0.00050602** | **1,536 ms** | **65.67%** |

Task-Aware is again 0.67 points below Always Strong; the paired interval is [-2.33, +1.00] points. Its cost is **22.5% lower** and latency **26.6% lower**. This replaces the earlier V2 claim based on one 240-task split with a more conservative replicated conclusion.

## What Changed

### Learned routing

The new `LearnedQualityGapEstimator` predicts `Strong accuracy - Cheap accuracy` before generation. It uses TF-IDF question features and/or observable structured features, Ridge regularization, and five-fold out-of-fold predictions for threshold selection. Dataset names and test labels are unavailable to the router.

The result is informative but mixed:

- V3 combined router: 78.67% accuracy, 43.2% lower cost than Always Strong, 36% Strong use.
- V3 text-only ablation: 79.33%, exactly matching Always Strong at 32.7% lower cost.
- V4 confirmation of the selected text-only variant: 80.00% versus Strong's 83.33%, at 32.2% lower cost.

The V3 gain did not fully replicate. Only 52 of 450 development examples distinguish the two models, so the learned target remains sparse. The repository retains this negative result instead of selecting a new policy on V4.

### Reflection and review

Reflection fits a response-only correctness calibrator on development responses and selects its threshold from outer-fold predictions. Strong receives the original task, image/tools, and Cheap candidate, preserving the candidate if correct and changing it only when necessary.

This mechanism is not reliably superior to blind Strong replacement. On V4, review and blind Strong have the same 7 beneficial and 5 harmful escalations. On V3, review produces fewer beneficial and one more harmful escalation. Prompted self-reported confidence also shifts across replications, making it a weak routing signal. Reflection is therefore an agentic diagnostic, not the headline method.

## Experimental Design

### Tasks

Each 300-example block has the same composition:

| Category | Count | Sources | Scoring |
|---|---:|---|---|
| Text reasoning | 100 | 40 GSM8K, 30 CommonsenseQA, 30 BBH logical deduction | numeric or multiple-choice accuracy |
| Vision-language | 100 | 40 ScienceQA, 20 ChartQA, 20 OCR-VQA, 20 MMMU subjects | multiple-choice, exact match, or numeric tolerance |
| Agentic tool use | 100 | 50 BFCL V4 simple, 50 BFCL V4 multiple | function name and required arguments |

V3 and V4 use half-sized blocks with identical proportions. Query, choice, tool-schema, and image-content fingerprints enforce zero overlap across development, V3, and V4.

### Model pool

| Role | Model | Temperature | Max output | Thinking |
|---|---|---:|---:|---|
| Cheap | `qwen3.5-35b-a3b` | 0.2 | 256 tokens | disabled |
| Strong | `qwen3.5-397b-a17b` | 0.2 | 256 tokens | disabled |

Both models support text, images, and tools. The experiment studies capacity selection, not an artificial text-model/VLM boundary.

### Policies

1. **Always Cheap** is the low-cost fixed-model control.
2. **Always Strong** is the high-capacity fixed-model control.
3. **Handcrafted Task-Aware** uses transparent request-time cues and a frozen risk threshold of 2.
4. **Learned Cost-Aware** learns the model-pair quality gap from development responses.
5. **Calibrated Reflection** calls Cheap first and conditionally asks Strong to review-and-correct.

The handcrafted feature constants are heuristic, not literature-derived. The learned router is the principled alternative; the rule baseline remains useful because it generalizes more stably in this small-data regime.

## Literature Positioning

The project follows quality-gap and preference-based routing in [Hybrid LLM](https://arxiv.org/abs/2404.14618), [RouteLLM](https://arxiv.org/abs/2406.18665), and [LLM Routing with Benchmark Datasets](https://arxiv.org/abs/2309.15789); it treats post-response escalation as a cascade following [FrugalGPT](https://arxiv.org/abs/2305.05176) and [AutoMix](https://arxiv.org/abs/2310.12963). [Deep Model Reassembly](https://arxiv.org/abs/2210.17409) motivates model reuse under performance and resource constraints, but does not justify the handcrafted query thresholds.

See [`docs/literature_review.md`](docs/literature_review.md) and [`docs/supervisor_review.zh-CN.md`](docs/supervisor_review.zh-CN.md) for the detailed critique and method changes.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[study,test]"
python scripts/build_manifest.py
python scripts/build_v3_data.py
python -m pytest
```

Set `QWEN_API_KEY` and `QWEN_BASE_URL`, then run:

```bash
python scripts/run_v3_study.py --study-version V3 --workers 8
python scripts/run_v3_ablations.py --workers 8
python scripts/build_v3_data.py \
  --development data/manifest.jsonl data/v3_test.jsonl \
  --out data/v4_test.jsonl --image-dir data/v4_images \
  --seed 20260713 --version v4
python scripts/run_v3_study.py \
  --development data/manifest.jsonl data/v3_test.jsonl \
  --test data/v4_test.jsonl --out results/qwen3.5-v4-study \
  --learned-features text --study-version V4 --workers 8
python scripts/aggregate_replications.py
```

Never commit API keys. Responses are cached under `.cache/routerbench/`; cache identity includes task content, model, prompt version, solve/review mode, candidate answer, and decoding settings.

## Limitations

- The study uses one provider, one model family, and 600 total sampled tasks.
- The model pair disagrees on few development examples, limiting learned-router sample efficiency.
- TF-IDF captures textual similarity but not image content; a learned multimodal encoder is future work.
- Prompted confidence is not a reliable substitute for token-level or internal uncertainty.
- API latency includes remote queueing variance; cost conclusions are more stable than latency conclusions.
- BFCL evaluation checks the first canonical function call and required arguments.
- Public dataset revisions are not pinned, so rebuilding in the future may require version updates.

Main artifacts are under [`results/qwen3.5-v4-study`](results/qwen3.5-v4-study), replicated frozen-policy results under [`results/qwen3.5-confirmatory`](results/qwen3.5-confirmatory), and V3 learned-feature ablations under [`results/qwen3.5-v3-ablation`](results/qwen3.5-v3-ablation).
