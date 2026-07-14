# RouterBench-Mini: Cost-Aware Model Reuse for Multimodal Agents

[中文说明](README.zh-CN.md)

This is a small personal experimental report that cannot independently support a paper-level contribution. It records a small experiment in my gradual return from engineering practice to research.

I started from a direct question. Sending every request to a cheap model can hurt accuracy on difficult tasks, while always using a strong model keeps API cost and latency unnecessarily high. RouterBench-Mini studies this model-selection problem: when is the cheap model sufficient, and when is the stronger model worth invoking?

Every stage uses two unified multimodal models from the same Qwen 3.5 family under shared prompts, decoding settings, scoring rules, and measured API costs. The project develops through V1 to V4. Each stage retains its mistakes, results, and motivation for the next step instead of reporting only the best run.

## Research Question and Shared Setup

### Model pool

| Role | Model | Position | Input/output price (CNY/million tokens) |
|---|---|---|---:|
| Cheap | `qwen3.5-35b-a3b` | smaller and less expensive unified multimodal model | 0.4 / 3.2 |
| Strong | `qwen3.5-397b-a17b` | larger and stronger but more expensive unified multimodal model | 1.2 / 7.2 |

Both models handle text, images, and tools. The experiment therefore studies capacity selection within one shared capability boundary rather than imposing an artificial text-model/VLM split. V1 uses `temperature=0`; V2 through V4 use `temperature=0.2`; every stage limits output to 256 tokens and disables thinking.

### Three task families and five task formats

| Family | Task format | Sources | Scoring |
|---|---|---|---|
| Text | mathematical reasoning | GSM8K | final numeric match |
| Text | textual multiple choice | CommonsenseQA, BBH logical deduction | choice accuracy |
| Vision | visual multiple choice | ScienceQA, MMMU | choice accuracy |
| Vision | open visual question answering | ChartQA, OCR-VQA | normalized text or numeric tolerance |
| Tool | function calling | BFCL V4 simple and multiple | function name and gold-required arguments |

Gold answers are used only by deterministic scoring and are never exposed to the router or either Qwen model. The study compares Always Cheap, Always Strong, Task-Aware, Reflection, and, from V3 onward, Learned Cost-Aware.

## V1: Rule-Based Baseline

### Data and architecture

V1 contains 300 tasks, divided into 100 text, 100 vision, and 100 tool tasks, then split into 60 validation and 240 test examples. Text includes 40 GSM8K, 30 CommonsenseQA, and 30 BBH tasks; vision includes 80 ScienceQA, 10 ChartQA, and 10 OCR-VQA tasks; tools include 50 BFCL simple and 50 multiple tasks.

V1 establishes four baselines. Always Cheap and Always Strong bound cost and capacity. Task-Aware reads a preassigned dataset `rule_tier`: for example, GSM8K and logical deduction always use Strong, while CommonsenseQA always uses Cheap. Reflection calls Cheap first and uses answer format, prompted confidence, and self-check to decide whether to escalate to Strong. Its confidence threshold of 0.8 is selected on the 60 validation tasks.

### V1 test results

| Method | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 80.00% | 0.00023496 | 707 ms | 0.00% |
| Always Strong | **81.67%** | 0.00063335 | 1,412 ms | 100.00% |
| Task-Aware | 81.25% | 0.00044290 | 1,063 ms | 49.17% |
| Reflection | 78.75% | 0.00057631 | 1,317 ms | 33.33% |

V1 exposes two basic problems. First, Task-Aware routes by dataset identity, which leaks information and says little about unseen requests. Second, confidence is not comparable across task formats. A parseable tool call receives a hard-coded 0.75, so every test tool task escalates under the 0.8 threshold; some incorrect math answers report 0.95 or 1.0 and are accepted. Reflection costs almost as much as Strong while falling below Cheap, motivating the redesign in V2.

## V2: Observable Features and Probability Calibration

### What V2 changes

V2 removes `rule_tier` and limits routing to information observable at inference time. Task-Aware computes a risk score from question length, numbers, math and logic cues, images, choice count, tool count, required arguments, and schema depth. Validation selects a risk threshold of 2.0.

The study still uses 300 tasks and a 60/240 split, but vision is rebalanced to 40 ScienceQA, 20 ChartQA, 20 OCR-VQA, and 20 MMMU examples. Reflection trains logistic regression from Cheap confidence, format, self-check, and 13 request features, then applies Platt scaling to estimate `P(Cheap answer is correct)`. On escalation, Strong receives the original task and Cheap candidate and performs review-and-correct.

### V2 test results

| Method | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 76.67% | 0.00024165 | 1,141 ms | 0.00% |
| Always Strong | 77.92% | 0.00065225 | 1,783 ms | 100.00% |
| **Task-Aware** | **80.00%** | 0.00052408 | 1,610 ms | 68.33% |
| Reflection Full | 76.67% | **0.00025260** | **1,174 ms** | 2.08% |

Task-Aware exceeds Always Strong in V2, but this claim comes from one 240-task test split. Reflection reaches 95.00% on validation and only 76.67% on test, a clear sign of overfitting.

### V2 Reflection ablation

| Variant | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Format-only | 76.67% | **0.00024165** | **1,141 ms** | 0.00% |
| Raw confidence | 76.67% | 0.00024524 | 1,149 ms | 0.42% |
| **Calibrated response-only** | **79.17%** | 0.00056005 | 2,060 ms | 59.58% |
| Full: response plus 13 request features | 76.67% | 0.00025260 | 1,174 ms | 2.08% |

Cheap makes only five errors among the 60 validation tasks, yet the same small set fits the probability calibrator and selects the escalation threshold. With 13 request features added, the Full calibrator assigns high estimated correctness to many test errors and almost never escalates. Response-only works better, but a larger development set and genuinely out-of-sample threshold selection are still needed. This becomes the direct starting point for V3.

## V3: Learned Quality-Gap Routing

### Literature motivation and architecture

V3 draws on quality-gap prediction in [Hybrid LLM](https://arxiv.org/abs/2404.14618) and pairwise model routing in [RouteLLM](https://arxiv.org/abs/2406.18665), replacing handcrafted difficulty increments with supervised routing. [FrugalGPT](https://arxiv.org/abs/2305.05176) and [AutoMix](https://arxiv.org/abs/2310.12963) motivate the post-response Reflection cascade.

The old 300 tasks become development data, while a fingerprint-disjoint held-out set A adds 150 tasks with 50 text, 50 vision, and 50 tool examples. Learned Cost-Aware follows this pipeline:

```text
300 development tasks
  -> Cheap and Strong both answer
  -> deterministic labels y = Strong correct - Cheap correct
  -> question TF-IDF plus 13 observable structured features
  -> Ridge predicts the Strong-minus-Cheap quality gap
  -> five-fold out-of-fold scores select one global threshold
  -> route each new set-A task to Cheap or Strong before generation
```

Strong alone is correct on 18 development tasks, Cheap alone on 14, and the models have the same outcome on 268. The primary V3 Learned Router uses Combined features and threshold 0.04986. Reflection instead fits a response-only Cheap-correctness calibrator and selects threshold 0.65 from outer-fold probabilities.

### V3 main results

| Method | Accuracy | 95% bootstrap CI | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 75.33% | [68.00, 82.00] | 0.00024400 | 692 ms | 0.00% |
| Always Strong | **79.33%** | [72.67, 85.33] | 0.00066199 | 1,568 ms | 100.00% |
| Task-Aware | 78.67% | [72.00, 85.33] | 0.00051066 | 1,305 ms | 65.33% |
| Learned Combined | 78.67% | [72.00, 85.33] | **0.00037586** | **1,031 ms** | 36.00% |
| Reflection | 74.00% | [66.67, 81.33] | 0.00063306 | 1,732 ms | 66.00% |

Learned Combined answers one fewer task correctly than Always Strong while making 96 fewer Strong calls and reducing cost by 43.2%. Reflection escalates 99/150 tasks, but review fixes only five Cheap errors and damages seven correct Cheap answers, leaving it below Always Cheap.

### V3 Learned feature ablation

| Feature variant | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Combined: TF-IDF plus 13 structured features | 78.67% | **0.00037586** | **1,031 ms** | **36.00%** |
| Structured-only | 78.67% | 0.00039014 | 1,117 ms | 46.67% |
| **Text-only TF-IDF** | **79.33%** | 0.00044537 | 1,191 ms | 54.67% |

Text-only matches Always Strong on set A at 32.7% lower cost and is selected for V4 by the accuracy-first, cost-tiebreak rule. It answers only one more task correctly than Combined while making 28 more Strong calls. This is a candidate to confirm on new data, not a stable conclusion.

## V4: Confirmatory Evaluation

### What V4 resolves

V4 is not a new routing algorithm. After V3 method selection, set A joins the original 300 tasks to form 450 development examples. A different fingerprint-disjoint set B of 150 becomes the final untouched confirmation set.

The Learned Router freezes the V3-selected text-only training recipe, then refits TF-IDF, Ridge, and the five-fold threshold of 0.02606 on 450 tasks. Reflection retains the same response-only architecture, recalibrates on 450 tasks, and selects threshold 0.75. The V4 test set is unavailable to feature-mode, model, and threshold selection.

### V4 final results

![V4 held-out accuracy-cost trade-off](results/qwen3.5-v4-study/pareto.png)

| Method | Accuracy | 95% bootstrap CI | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 78.67% | [72.00, 84.68] | 0.00023762 | 1,178 ms | 0.00% |
| Always Strong | **83.33%** | [77.33, 89.33] | 0.00064448 | 2,619 ms | 100.00% |
| **Task-Aware** | **82.67%** | [76.67, 88.00] | 0.00050139 | **1,767 ms** | 66.00% |
| Learned Text-only | 80.00% | [73.33, 86.00] | **0.00043693** | 2,391 ms | 50.00% |
| Reflection | 80.00% | [73.33, 86.00] | 0.00047537 | 2,202 ms | 46.00% |

Task-Aware is 0.67 percentage points below Always Strong, with a paired 95% difference interval of [-2.67, +1.33] points, while reducing cost by 22.2% and observed latency by 32.5%. Learned Text-only still reduces cost by 32.2%, but it is 3.33 points below Strong. The V3 accuracy match does not replicate.

The apparent move from 79.33% in V3 to 80.00% in V4 cannot show that increasing development data helps because the test tasks differ; Always Strong itself moves from 79.33% to 83.33%. A valid 300-versus-450 data-scaling study must evaluate both training sizes on the same new test set.

### Two non-overlapping held-out batches

Sets A and B each contain 150 non-overlapping tasks, but set A participates in text-only selection and later joins V4 development. Only set B is the final untouched confirmation set. The pooled table therefore includes only Always Cheap, Always Strong, and Task-Aware, whose policies remain unchanged between V3 and V4, as a cross-batch stability check.

| Frozen method | Accuracy over 300 tasks across both batches | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 77.00% | 0.00024081 | 935 ms | 0.00% |
| Always Strong | **81.33%** | 0.00065324 | 2,094 ms | 100.00% |
| **Task-Aware** | 80.67% | **0.00050602** | **1,536 ms** | 65.67% |

Across both batches, Task-Aware remains 0.67 points below Strong with a paired interval of [-2.33, +1.00] points, while reducing cost by 22.5% and latency by 26.6%. This is more conservative and credible than the V2 statement that Task-Aware exceeds Strong on one split.

## What I Learned

TF-IDF, Ridge, logistic regression, probability calibration, and cross-validation are classical machine-learning methods and should not be presented as a new routing theory. For me, the useful part of the project is completing one small research iteration: formulate the accuracy-cost question, establish Cheap and Strong controls, discover label leakage and calibration overfitting, add learned routing and ablations, and then use a new confirmation set to overturn an attractive result.

The simple Task-Aware policy is ultimately the most stable. Learned routing is limited by sparse Cheap/Strong disagreements, while Reflection is weakened by shifted prompted confidence and regressions during Strong review. This demo is best read as a small personal report on gradually returning from engineering practice to research, not as a paper-level algorithmic contribution.

## Next Experiments

1. **Collect routing-informative data.** Only 52 of 450 development examples distinguish Cheap from Strong. Future sampling should target harder tasks and model disagreements and report learning curves instead of mainly adding `y=0` examples.
2. **Run a controlled data-scaling study.** Train the same router on 300 and 450 examples and evaluate both on one new frozen test set with identical features, threshold selection, and model calls.
3. **Improve request representations.** Compare a fixed pretrained text embedding against TF-IDF on the same split. For vision tasks, add a lightweight multimodal or image embedding so pre-generation routing can inspect image content rather than only `has_image`.
4. **Optimize an explicit utility.** The current policy prioritizes accuracy and uses cost only to break ties. A follow-up should prespecify `accuracy - lambda * cost - mu * latency`, or minimize cost under a fixed allowable accuracy loss.
5. **Strengthen uncertainty and verification.** Where available, compare token log-probability, entropy, sampling consistency, or an independent verifier against prompted confidence, while retaining blind Strong replacement as the counterfactual for review.
6. **Broaden model and data replication.** Add more model tiers, model families, and sampled batches; report paired intervals and failures; freeze all policies and hyperparameters before constructing another final confirmation set.

## Relationship to Prior Work

In addition to [Hybrid LLM](https://arxiv.org/abs/2404.14618), [RouteLLM](https://arxiv.org/abs/2406.18665), [FrugalGPT](https://arxiv.org/abs/2305.05176), and [AutoMix](https://arxiv.org/abs/2310.12963), which directly motivate V3, [LLM Routing with Benchmark Datasets](https://arxiv.org/abs/2309.15789) provides routing-benchmark context. [Deep Model Reassembly](https://arxiv.org/abs/2210.17409) motivates model reuse under performance and resource constraints but does not directly justify this project's handcrafted thresholds.

See [`docs/literature_review.md`](docs/literature_review.md) and [`docs/supervisor_review.zh-CN.md`](docs/supervisor_review.zh-CN.md) for the detailed literature review and supervisor-style critique.

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

API keys are never committed. Responses are cached under `.cache/routerbench/`; cache identity includes task content, model, prompt version, solve/review mode, candidate answer, and decoding parameters.

## Limitations

- The study uses one provider, one model family, and 600 sampled tasks.
- Few development tasks produce different correctness outcomes for the two models, limiting learned routing.
- TF-IDF does not inspect image content, so pre-generation visual difficulty is represented weakly.
- Prompted confidence is not a substitute for token-level or internal uncertainty.
- API latency includes remote queueing variance; cost conclusions are more stable than latency conclusions.
- BFCL scoring checks the first canonical function call and arguments required by the gold answer.
- Public dataset revisions are not pinned, so future rebuilds may require script updates.

Main artifacts are under [`results/qwen3.5-v4-study`](results/qwen3.5-v4-study), cross-batch frozen-policy results under [`results/qwen3.5-confirmatory`](results/qwen3.5-confirmatory), and V3 feature ablations under [`results/qwen3.5-v3-ablation`](results/qwen3.5-v3-ablation).
