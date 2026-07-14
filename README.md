# RouterBench-Mini: Cost-Aware Model Reuse for Multimodal Agents

[中文说明](README.zh-CN.md)

RouterBench-Mini is a small experiment I completed while gradually returning from engineering practice to research training. Its goal is to work through the proposal, validation, failure analysis, and correction of a multimodal model-routing study at a limited scale.

## Abstract

In multimodal agent systems, sending every request to a smaller model can reduce API cost and response latency, but it may not preserve accuracy on difficult tasks. Sending every request to a stronger model is safer, but it continuously increases inference cost. RouterBench-Mini studies this tension: for tasks involving text reasoning, visual understanding, and tool use, can a system decide before or after generation whether a stronger model is actually necessary?

The experiment consistently uses two unified multimodal models from the Qwen 3.5 family under shared prompts, decoding parameters, scoring rules, and API cost accounting. It first establishes Always Cheap and Always Strong as two reference boundaries, then studies task-feature rules, a confidence-based Reflection cascade, and a learned router that predicts the quality gap between Cheap and Strong. The project develops through four stages, V1 to V4, and each stage is motivated by problems exposed in the previous one rather than by keeping only the best run.

The final confirmatory study shows that the more complex learned router does not stably outperform a simple rule. Across two non-overlapping evaluation batches containing 300 tasks in total, Task-Aware trails Always Strong by only 0.67 percentage points while reducing average cost by 22.5% and latency by 26.6%. Learned routing is constrained by the small number of examples where Cheap and Strong genuinely differ in correctness. Reflection is weakened by unreliable self-reported confidence and by Strong review sometimes changing correct answers into incorrect ones.

This project therefore does not claim a new routing theory. It records how a small experiment progressed from basic rules through dataset-label leakage, probability-calibration overfitting, learned routing, and confirmatory evaluation, ending with a conclusion that is less striking than the early result but more credible.

## 1. Problem Background

A practical agent rarely handles only one type of request. The same system may need to solve mathematical reasoning problems, answer textual multiple-choice questions, understand charts and images, and call functions. These tasks do not place equal demands on model capability.

If every request goes to a cheap model, the system costs less but may lack capability on difficult tasks. If every request goes to a strong model, many easy tasks incur unnecessary API cost and latency. A practical multi-model system must therefore answer the following question:

> Without access to the gold answer, how can the system determine whether a smaller model is sufficient for the current request, and when invoking a stronger model provides enough quality improvement to justify its cost?

This question can be divided into three experimental questions:

1. Can inference-time observable request features support effective routing before generation?
2. After Cheap generates an answer, can confidence, output format, and self-check signals identify when escalation is needed?
3. Can historical examples teach a router the relative quality gap between Cheap and Strong so that model selection becomes data-driven?

RouterBench-Mini studies these three approaches in sequence.

## 2. Shared Experimental Setup

### 2.1 Model Pool

The experiment uses two unified multimodal models from the same Qwen 3.5 family:

| Role | Model | Position | Input/output price (CNY/million tokens) |
|---|---|---|---:|
| Cheap | `qwen3.5-35b-a3b` | smaller and less expensive to call | 0.4 / 3.2 |
| Strong | `qwen3.5-397b-a17b` | larger and stronger but more expensive | 1.2 / 7.2 |

Both models support text, images, and tool calls. The study therefore does not route between a text-only model and a vision model. It selects model scale within a shared modality boundary according to task difficulty.

V1 uses `temperature=0`, while V2 through V4 use `temperature=0.2`. Every version disables thinking and limits generation to 256 tokens.

### 2.2 Tasks

The benchmark covers three task families and five task formats:

| Family | Task format | Sources | Scoring |
|---|---|---|---|
| Text | mathematical reasoning | GSM8K | final numeric match |
| Text | textual multiple choice | CommonsenseQA, BBH logical reasoning | choice accuracy |
| Vision | visual multiple choice | ScienceQA, MMMU | choice accuracy |
| Vision | open visual question answering | ChartQA, OCR-VQA | normalized text match or numeric tolerance |
| Tool | function calling | BFCL V4 simple and multiple | function name and required-argument match |

Gold answers are used only by deterministic scorers. They are never exposed to the router, Cheap, or Strong.

### 2.3 Evaluation Metrics

Every method reports the same metrics:

- **Accuracy:** the proportion of final answers accepted by deterministic scoring.
- **Average cost per task:** API cost calculated from actual input and output token counts.
- **Average latency:** observed time from request submission to final response.
- **Strong use:** the proportion of tasks for which Strong is called.

Accuracy measures the quality of the routed system, while Strong use and average cost measure model-call overhead. API latency is affected by remote queueing and service conditions, so cost conclusions are generally more stable than latency conclusions.

## 3. V1: Establishing the Basic Routing Framework

### 3.1 Method and Data

V1 first constructs a small dataset covering three task families, with 300 tasks in total:

- 100 text tasks: 40 GSM8K, 30 CommonsenseQA, and 30 BBH logical reasoning tasks.
- 100 vision tasks: 80 ScienceQA, 10 ChartQA, and 10 OCR-VQA tasks.
- 100 tool tasks: 50 BFCL simple and 50 BFCL multiple tasks.

The 300 tasks are split into 60 validation tasks and 240 test tasks. Validation selects the Reflection confidence threshold, while the test set compares routing strategies.

V1 defines four baseline routes:

- **Always Cheap:** every task is answered by Cheap, providing the minimum-cost baseline.
- **Always Strong:** every task is answered by Strong, providing the high-capability baseline.
- **Task-Aware:** the router selects a model from a preassigned dataset `rule_tier`. For example, GSM8K and logical reasoning always go to Strong, while CommonsenseQA always goes to Cheap.
- **Reflection:** Cheap answers first, then answer format, self-reported confidence, and self-check determine whether to escalate to Strong. Validation selects a confidence threshold of 0.8.

### 3.2 V1 Results

| Method | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 80.00% | 0.00023496 | 707 ms | 0.00% |
| Always Strong | **81.67%** | 0.00063335 | 1,412 ms | 100.00% |
| Task-Aware | 81.25% | 0.00044290 | 1,063 ms | 49.17% |
| Reflection | 78.75% | 0.00057631 | 1,317 ms | 33.33% |

Always Strong achieves the highest accuracy, but it improves on Always Cheap by only 1.67 percentage points while increasing average cost by about 169.6%. Task-Aware trails Strong by only 0.42 points and reduces Strong use to 49.17%, providing early evidence that routing may be useful.

Reflection does not perform as expected. Its average cost approaches Always Strong, but its accuracy is only 78.75%, below Always Cheap.

### 3.3 Problems Exposed by V1

Further inspection reveals two fundamental problems.

First, Task-Aware directly reads a preassigned `rule_tier`. This rule effectively uses dataset identity, such as knowing in advance that a task comes from GSM8K or CommonsenseQA, to select a model. A deployed router normally does not receive this manual label, so the result contains clear information leakage and does not represent routing on unseen requests.

Second, Reflection assumes that self-reported confidence has the same meaning across task formats, but the evidence does not support this assumption. Some incorrect math answers report confidence of 0.95 or even 1.0. A tool call that parses successfully is assigned a fixed confidence of 0.75 by the program. Under the 0.8 threshold, almost every tool task escalates while some high-confidence wrong answers are accepted.

V2 therefore needs two changes:

1. Remove the dataset-dependent `rule_tier` and restrict routing to information that is genuinely observable at inference time.
2. Stop comparing raw confidence directly and instead calibrate the probability that a Cheap answer is correct.

## 4. V2: Observable Features and Probability Calibration

### 4.1 Method and Data Changes

V2 retains 300 tasks and the 60/240 validation-test split, but rebalances the vision tasks:

- 40 ScienceQA tasks.
- 20 ChartQA tasks.
- 20 OCR-VQA tasks.
- 20 MMMU tasks.

This change reduces ScienceQA's dominance and adds the more difficult and diverse MMMU benchmark.

V2 still compares four routes but redesigns Task-Aware and Reflection:

- **Always Cheap:** remains the minimum-cost baseline.
- **Always Strong:** remains the all-Strong capability baseline.
- **Task-Aware:** no longer reads dataset labels. It computes a risk score from observable features including question length, number count, math cues, logic cues, image presence, choice count, candidate-tool count, required-argument count, and schema depth. Validation selects a risk threshold of 2.0.
- **Reflection Full:** trains logistic regression from Cheap's raw confidence, answer format, self-check, and 13 request-side features, then applies Platt scaling to estimate `P(Cheap answer is correct)`. When this probability falls below the threshold, Strong receives the original task and Cheap candidate and performs review-and-correct.

### 4.2 V2 Results

| Method | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 76.67% | 0.00024165 | 1,141 ms | 0.00% |
| Always Strong | 77.92% | 0.00065225 | 1,783 ms | 100.00% |
| **Task-Aware** | **80.00%** | 0.00052408 | 1,610 ms | 68.33% |
| Reflection Full | 76.67% | **0.00025260** | **1,174 ms** | 2.08% |

Task-Aware reaches 80.00% accuracy on this 240-task test set, exceeding both Always Cheap and Always Strong. This result suggests that inference-time structural features may identify some tasks that benefit more from Strong.

However, the advantage comes from a single test split and is not sufficient evidence that Task-Aware stably outperforms Strong.

Reflection Full reaches 95.00% accuracy on validation but only 76.67% on test, identical to Always Cheap. It escalates only 2.08% of test requests, indicating that the calibrator assigns overly high correctness probabilities to many wrong answers and fails out of sample.

### 4.3 Reflection Feature Ablation

V2 evaluates four Reflection ablations. Format-only retains only whether the answer can be parsed. Raw confidence uses only Cheap's self-reported score. Calibrated response-only combines answer format, raw confidence, and self-check, then calibrates the probability that Cheap is correct. Full adds 13 request-level structural features to response-only.

| Variant | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Format-only | 76.67% | **0.00024165** | **1,141 ms** | 0.00% |
| Raw confidence | 76.67% | 0.00024524 | 1,149 ms | 0.42% |
| **Calibrated response-only** | **79.17%** | 0.00056005 | 2,060 ms | 59.58% |
| Full: response + 13 request features | 76.67% | 0.00025260 | 1,174 ms | 2.08% |

Calibrated response-only achieves the highest accuracy at 79.17%, suggesting that joint calibration of format, confidence, and self-check is more effective than raw confidence alone. Adding 13 request features causes Full to almost stop escalating, suggesting that extra features increase validation overfitting under the current small-sample setting.

### 4.4 V2 Limitations and the V3 Plan

The main V2 problem is validation size. Cheap makes only five errors among the 60 validation tasks, yet the same data must fit the probability calibrator and select an escalation threshold. Reflection therefore reaches 95.00% on validation and falls to 76.67% on test. In addition, predicting whether Cheap is wrong is not the same as predicting whether Strong provides value. If both models are wrong, escalation only adds cost. If Cheap is correct and Strong is wrong, escalation reduces accuracy.

V3 therefore moves all 300 existing tasks into development data and changes the learning target to Strong's quality advantage over Cheap:

```text
300 development tasks
  -> Cheap and Strong both answer
  -> deterministic scoring records whether each model is correct
  -> construct y = Strong correct - Cheap correct
  -> extract question TF-IDF and 13 observable structural features
  -> use Ridge to predict Strong's quality advantage over Cheap
  -> use five-fold out-of-fold scores to select a global routing threshold
  -> route 150 new set-A tasks to Cheap or Strong before generation
```

This change asks the router to predict whether calling Strong has practical value rather than merely whether Cheap may be wrong.

## 5. V3: Learned Quality-Gap Routing

### 5.1 Method Design

V3 draws on quality-gap prediction from [Hybrid LLM](https://arxiv.org/abs/2404.14618) and pairwise model routing from [RouteLLM](https://arxiv.org/abs/2406.18665), reformulating the decision to use Strong as a supervised learning problem. [FrugalGPT](https://arxiv.org/abs/2305.05176) and [AutoMix](https://arxiv.org/abs/2310.12963) provide background for the post-generation Reflection cascade.

The original 300 tasks are no longer used as test data. They become the development set, and a fingerprint-disjoint test set A is constructed with 150 tasks:

- 50 text tasks.
- 50 vision tasks.
- 50 tool-calling tasks.

V3 adds Learned Cost-Aware as a fifth method:

- **Always Cheap:** every request uses Cheap.
- **Always Strong:** every request uses Strong.
- **Task-Aware:** retains V2's observable structural-risk rule.
- **Learned Cost-Aware:** predicts Strong's expected quality gain over Cheap before generation and selects a model using a threshold.
- **Reflection:** calls Cheap first, predicts Cheap correctness from answer-side confidence, format, and self-check, and invokes Strong review below a threshold.

The Learned Router follows this pipeline:

```text
300 development tasks
  -> Cheap and Strong both answer
  -> deterministic scoring records whether each model is correct
  -> construct y = Strong correct - Cheap correct
  -> extract question TF-IDF and 13 observable structural features
  -> use Ridge to predict Strong's quality advantage over Cheap
  -> use five-fold out-of-fold scores to select a global routing threshold
  -> route 150 new set-A tasks to Cheap or Strong before generation
```

Among the 300 development tasks:

- Strong alone is correct on 18.
- Cheap alone is correct on 14.
- Both models have the same correctness outcome on 268.

Only 32 tasks therefore provide nonzero supervision for model selection; the other 268 have a quality-gap label of zero.

The primary V3 Learned Router uses a Combined representation of TF-IDF and 13 structural features, with a threshold of 0.04986. Reflection keeps only response-side features and selects an escalation threshold of 0.65 from outer-fold predictions.

### 5.2 V3 Main Results

| Method | Accuracy | 95% bootstrap CI | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 75.33% | [68.00, 82.00] | 0.00024400 | 692 ms | 0.00% |
| Always Strong | **79.33%** | [72.67, 85.33] | 0.00066199 | 1,568 ms | 100.00% |
| Task-Aware | 78.67% | [72.00, 85.33] | 0.00051066 | 1,305 ms | 65.33% |
| Learned Combined | 78.67% | [72.00, 85.33] | **0.00037586** | **1,031 ms** | 36.00% |
| Reflection | 74.00% | [66.67, 81.33] | 0.00063306 | 1,732 ms | 66.00% |

Learned Combined reaches 78.67% accuracy, only one task below Always Strong, while reducing Strong use from 100% to 36%. It makes 96 fewer Strong calls and lowers average cost by 43.2%.

On this single set-A result, Learned Combined provides a favorable accuracy-cost balance and suggests that learned routing may outperform a fixed rule in cost efficiency.

Reflection again falls short. It escalates 99 of 150 tasks, but Strong review fixes only five Cheap errors and changes seven correct Cheap answers into errors. Its final accuracy is 74.00%, below Always Cheap.

This result shows that asking Strong to review Cheap is not equivalent to receiving Strong's independent capability. The candidate answer may anchor Strong, and the review prompt can introduce regressions.

### 5.3 Learned Router Feature Ablation

V3 ablates the Learned Router representation. Combined uses question TF-IDF plus 13 structural features. Structured-only removes TF-IDF and retains question length, numbers, images, choices, and tool-schema structure. Text-only removes all structural features and keeps only question TF-IDF.

| Feature variant | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Combined: TF-IDF + 13 structural features | 78.67% | **0.00037586** | **1,031 ms** | **36.00%** |
| Structured-only | 78.67% | 0.00039014 | 1,117 ms | 46.67% |
| **Text-only TF-IDF** | **79.33%** | 0.00044537 | 1,191 ms | 54.67% |

Text-only TF-IDF reaches 79.33% on set A, matching Always Strong while reducing average cost by 32.7%. Structured-only and Combined have equal accuracy, but Structured-only calls Strong more often. Combined has the lowest cost and answers only one fewer task correctly than Text-only. Under the selection rule of prioritizing accuracy and using cost to break accuracy ties, Text-only advances to V4, but its advantage still requires validation on a new test set.

### 5.4 From V3 to V4

V3 appears to produce a favorable result: Text-only Learned Router matches Always Strong in accuracy and saves about one third of the cost.

However, set A has already been used to compare Combined, Structured-only, and Text-only and to select a feature representation. It is therefore no longer a completely untouched final test set. Treating set A as the final result could mistake a chance feature-selection advantage for a stable improvement.

V4 consequently introduces no new routing algorithm. It performs confirmatory evaluation:

1. Freeze the Text-only recipe selected in V3.
2. Add set A to development data.
3. Refit the model without changing the feature mode or method.
4. Construct a new set B that has not participated in method selection.
5. Test whether V3's result of matching Strong at lower cost replicates.

## 6. V4: Confirmatory Evaluation

### 6.1 Confirmatory Design

V4 combines the original 300 development tasks with the 150 tasks in set A, producing 450 development tasks. It then constructs a 150-task confirmation set B whose fingerprints do not overlap with any existing data.

Set A has participated in V3 feature selection and is no longer final confirmation data. Set B is used only after the model, features, and procedure are frozen, making it the experiment's only completely untouched confirmation set.

V4 compares:

- **Always Cheap:** unchanged.
- **Always Strong:** unchanged.
- **Task-Aware:** retains the observable-risk rule used since V2.
- **Learned Text-only:** freezes V3's TF-IDF representation and refits Ridge and the five-fold threshold on 450 development tasks. The selected threshold is 0.02606.
- **Reflection:** freezes the response-only architecture and recalibrates it on 450 development tasks. The selected escalation threshold is 0.75.

Set B is not used to select the feature mode, learner type, or threshold.

### 6.2 V4 Final Results

![V4 held-out accuracy-cost trade-off](results/qwen3.5-v4-study/pareto.png)

| Method | Accuracy | 95% bootstrap CI | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 78.67% | [72.00, 84.68] | 0.00023762 | 1,178 ms | 0.00% |
| Always Strong | **83.33%** | [77.33, 89.33] | 0.00064448 | 2,619 ms | 100.00% |
| **Task-Aware** | **82.67%** | [76.67, 88.00] | 0.00050139 | **1,767 ms** | 66.00% |
| Learned Text-only | 80.00% | [73.33, 86.00] | **0.00043693** | 2,391 ms | 50.00% |
| Reflection | 80.00% | [73.33, 86.00] | 0.00047537 | 2,202 ms | 46.00% |

Always Strong reaches the highest V4 accuracy at 83.33%. Task-Aware reaches 82.67%, only 0.67 percentage points lower, with a paired 95% accuracy-difference interval of [-2.67, +1.33]. At the same time, Task-Aware reduces average cost by 22.2% and observed latency by 32.5%.

Learned Text-only reaches 80.00% and costs 32.2% less than Always Strong, but its accuracy is 3.33 points lower. The V3 result in which Learned Router matched Always Strong does not replicate on the new confirmation set.

Reflection also reaches 80.00%, but it calls Cheap first and then invokes Strong for 46% of requests. Given the extra request stage introduced by the cascade, it does not show a stable advantage over pre-generation routing.

### 6.3 Interpreting the Difference Between V3 and V4

The change from 79.33% in V3 to 80.00% in V4 cannot demonstrate that adding development data improves Learned Router accuracy because the two versions use different test tasks.

Always Strong itself reaches 79.33% in V3 and 83.33% in V4, showing that the test batches differ in difficulty and model performance. Absolute accuracy should therefore not be compared directly across V3 and V4.

A valid comparison of training on 300 versus 450 tasks must evaluate both training variants on the same new test set while holding the feature representation, threshold-selection procedure, and model API configuration constant.

The conclusion supported by V4 is narrower:

> After freezing the feature recipe and method, V3's result in which Text-only Learned Router matched Always Strong did not replicate on confirmation set B.

This result is less attractive than the V3 outcome, but it is more credible than reporting only one selected run.

## 7. Stable Baselines Across Two Non-Overlapping Evaluation Batches

Sets A and B each contain 150 non-overlapping tasks, but set A participated in V3 feature selection and later joined the V4 development set. Only set B is the final untouched confirmation set.

Always Cheap, Always Strong, and Task-Aware did not change between V3 and V4. Their results on sets A and B can therefore be pooled as a cross-batch stability check for fixed baselines.

| Frozen method | Accuracy over 300 tasks across both batches | Avg. cost/task (CNY) | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 77.00% | 0.00024081 | 935 ms | 0.00% |
| Always Strong | **81.33%** | 0.00065324 | 2,094 ms | 100.00% |
| **Task-Aware** | 80.67% | **0.00050602** | **1,536 ms** | 65.67% |

Across the 300 tasks, Task-Aware trails Always Strong by only 0.67 percentage points, with a paired 95% difference interval of [-2.33, +1.00], while:

- Reducing average cost by 22.5%.
- Reducing average latency by 26.6%.
- Reducing Strong use from 100% to 65.67%.

Task-Aware exceeds Always Strong on the single V2 split, but the pooled result supports a more conservative interpretation: Task-Aware does not stably outperform Strong. It maintains similar accuracy while avoiding about one third of Strong calls.

## 8. Main Findings

### 8.1 The Simple Rule Is the Most Stable Current Method

The most stable final method is not the learned router but Task-Aware, which uses observable structural features.

It does not need to call Cheap first, depend on model-reported confidence, or estimate a quality gap from sparse pairwise disagreements. Across two non-overlapping batches, it consistently reduces cost and latency while remaining close to Always Strong in accuracy.

This does not show that handcrafted rules are generally superior to learned routing. It shows that under the current data scale and model pair, the simpler method has lower estimation variance.

### 8.2 Learned Routing Is Limited More by Informative Labels Than by Model Simplicity

Among 450 development tasks, only 52 distinguish Cheap and Strong in correctness. On all remaining examples, both models are either correct or incorrect.

For a router, many `y=0` samples only show that the two models perform equally. They provide little evidence about which tasks deserve Strong. Replacing Ridge with a more complex classifier or regressor is unlikely to produce stable generalization unless the number of model-disagreement samples also increases.

### 8.3 Self-Reported Confidence Is Not a Reliable Cross-Task Uncertainty Measure

Confidence is produced differently across task formats. Math tasks can produce high-confidence wrong answers, while tool confidence may be assigned by a program rule rather than reflecting model uncertainty.

V2 shows that calibrating these signals on a very small validation set can overfit. V3 and V4 show that even with cross-validation, response-side confidence alone is insufficient for stable Reflection routing.

### 8.4 Strong Review Can Introduce Negative Corrections

Reflection assumes that when Cheap is unreliable, showing its candidate to Strong for correction should not produce a worse final answer.

V3 failure cases reject this assumption. Strong review fixes some Cheap errors but also changes some correct Cheap answers into incorrect ones. Future Reflection studies should compare two escalation modes:

- **Review Strong:** Strong sees and reviews the Cheap candidate.
- **Blind Strong:** Strong answers the original task independently without seeing the Cheap candidate.

This comparison is necessary to separate improvements from Strong's capability from effects introduced by the review mechanism.

### 8.5 Confirmatory Evaluation Can Overturn a Chance Improvement

Text-only matches Always Strong on set A and appears to be the project's strongest result. Once the feature recipe is frozen and evaluated on set B, the result does not replicate.

This failure is informative. It demonstrates why method-selection data and final confirmation data should be separated, and why a single small-sample result should not be interpreted as a stable gain.

## 9. Positioning of the Experiment

RouterBench-Mini uses classical methods including TF-IDF, Ridge regression, logistic regression, Platt scaling, and cross-validation. These methods should not be presented as a new model-routing theory.

The project's main value is the complete record of a small research iteration:

```text
formulate the accuracy-cost question
  -> establish Cheap and Strong reference boundaries
  -> design rule-based routing and Reflection
  -> discover dataset-label leakage
  -> switch to inference-time observable features
  -> discover small-sample probability-calibration overfitting
  -> learn the Cheap-Strong quality gap
  -> run feature ablations and select a method
  -> construct a new confirmation set
  -> overturn an apparently strong result
  -> reach a more conservative but credible conclusion
```

It is therefore best read as a personal experimental report from a return to research training after engineering practice, not as an algorithmic contribution that independently supports a paper.

## 10. Future Experiments

### 10.1 Actively Collect Routing-Informative Examples

Only 52 of the 450 development tasks distinguish Cheap from Strong. A follow-up should not merely add more random easy examples. It should actively seek:

- Tasks that Cheap often fails but Strong can solve.
- Tasks on which Cheap and Strong follow different reasoning paths.
- Tasks whose visual content is decisive for difficulty.
- Tool tasks with more tools, parameter dependencies, and call combinations.

The study should also report learning curves to show whether routing quality becomes more stable as model-disagreement data increases.

### 10.2 Run a Controlled Training-Scale Study

To determine whether more development data helps, compare on the same new test set:

- A router trained on 300 tasks.
- A router trained on 450 tasks.
- A router trained with additional actively sampled disagreement tasks.

All variants must share the same features, model pair, threshold-selection procedure, and model API configuration.

### 10.3 Improve Request Representation

TF-IDF reads only request text and cannot inspect an image. For vision tasks, the router mainly observes structural indicators such as `has_image`, not visual content.

A follow-up can compare under identical data splits:

- TF-IDF.
- Fixed pretrained text embeddings.
- Image embeddings.
- Joint text-image embeddings.
- A lightweight multimodal encoder.

The goal is not simply to use a larger router, but to determine whether visual content supplies stable model-selection signals.

### 10.4 Prespecify a Cost-Utility Objective

The current threshold procedure prioritizes accuracy and uses cost only to break accuracy ties. It does not explicitly define the trade-off among accuracy, cost, and latency.

A follow-up can prespecify a utility function:

```text
utility = accuracy - lambda * cost - mu * latency
```

Alternatively, it can impose an accuracy constraint:

```text
minimize average call cost subject to an accuracy loss no greater than delta.
```

Prespecification prevents choosing the preferred accuracy-cost trade-off only after observing test results.

### 10.5 Improve Uncertainty Signals

Where supported by the model API, future work can compare:

- Token-level log probability.
- Output entropy.
- Agreement across multiple samples.
- An independent verifier score.
- Semantic disagreement between Cheap and the verifier.
- Agreement across different internal reasoning paths from Cheap.

These signals are closer to model uncertainty than prompted self-reported confidence, although they introduce additional computation or API calls.

### 10.6 Broaden Model and Data Replication

The current experiment uses one provider and one model family. Future work should add:

- More model-capacity tiers.
- Different model families.
- Different providers.
- Multiple randomly sampled data batches.
- More open visual QA and complex tool-calling tasks.

After all methods and hyperparameters are frozen, another final confirmation set should be built. Results should include paired confidence intervals, model-disagreement matrices, and representative failure cases.

## 11. Relationship to Prior Work

V3 learned routing is primarily motivated by:

- [Hybrid LLM](https://arxiv.org/abs/2404.14618): routing through predicted model-quality gaps.
- [RouteLLM](https://arxiv.org/abs/2406.18665): routing through pairwise preferences and relative model capability.
- [FrugalGPT](https://arxiv.org/abs/2305.05176): cost-aware cascades across multiple language models.
- [AutoMix](https://arxiv.org/abs/2310.12963): uncertainty-guided model escalation.
- [LLM Routing with Benchmark Datasets](https://arxiv.org/abs/2309.15789): benchmark context for language-model routing.
- [Deep Model Reassembly](https://arxiv.org/abs/2210.17409): motivation for model reuse under performance and resource constraints.

These works provide methodological context but do not directly establish that this project's handcrafted features, thresholds, or learners generalize.

More detailed literature and supervisor-style reviews are available in:

- [`docs/literature_review.md`](docs/literature_review.md)
- [`docs/supervisor_review.zh-CN.md`](docs/supervisor_review.zh-CN.md)

## 12. Reproduction

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[study,test]"

python scripts/build_manifest.py
python scripts/build_v3_data.py
python -m pytest
```

Set the required environment variables:

```bash
export QWEN_API_KEY="YOUR_API_KEY"
export QWEN_BASE_URL="YOUR_API_BASE_URL"
```

Run the V3 main study and feature ablation:

```bash
python scripts/run_v3_study.py \
  --study-version V3 \
  --workers 8

python scripts/run_v3_ablations.py \
  --workers 8
```

Build the V4 confirmation set:

```bash
python scripts/build_v3_data.py \
  --development data/manifest.jsonl data/v3_test.jsonl \
  --out data/v4_test.jsonl \
  --image-dir data/v4_images \
  --seed 20260713 \
  --version v4
```

Run the V4 confirmatory study:

```bash
python scripts/run_v3_study.py \
  --development data/manifest.jsonl data/v3_test.jsonl \
  --test data/v4_test.jsonl \
  --out results/qwen3.5-v4-study \
  --learned-features text \
  --study-version V4 \
  --workers 8
```

Aggregate results across batches:

```bash
python scripts/aggregate_replications.py
```

API keys are never committed. Model responses are cached under:

```text
.cache/routerbench/
```

Cache identity includes task content, model, prompt version, solve/review mode, candidate answer, and decoding parameters, preventing duplicate API calls when the configuration is unchanged.

## 13. Limitations

- The experiment uses only one provider and one model family.
- It contains only 600 sampled tasks, limiting statistical power.
- Few development examples produce different correctness outcomes for Cheap and Strong.
- TF-IDF cannot inspect image content, so visual routing representation is weak.
- Prompted self-reported confidence is not a substitute for model-internal uncertainty.
- Strong review may be anchored by the Cheap candidate.
- API latency includes remote queueing and service variation.
- BFCL scoring checks only the first canonical function call and arguments required by the gold answer.
- Public dataset revisions are not pinned, so future data reconstruction may require script updates.
- Results apply only to the models, task distribution, and pricing used in this experiment and should not be directly generalized to other model pools.

## 14. Result Files

The main artifacts are available under:

- V4 confirmatory study: [`results/qwen3.5-v4-study`](results/qwen3.5-v4-study)
- Frozen policies across batches: [`results/qwen3.5-confirmatory`](results/qwen3.5-confirmatory)
- V3 feature ablation: [`results/qwen3.5-v3-ablation`](results/qwen3.5-v3-ablation)

## Final Conclusion

At the current scale, RouterBench-Mini does not show that learned routing can stably match Always Strong, nor does it show that the post-generation Reflection cascade outperforms pre-generation routing.

The most stable result comes from the simple Task-Aware rule. Across two non-overlapping batches containing 300 tasks, it trails Always Strong by only 0.67 percentage points while reducing call cost by 22.5% and observed latency by 26.6%.

This conclusion does not imply that simple rules are the final answer to model routing. It more specifically shows that when model-disagreement labels are sparse, visual representation is limited, and uncertainty signals are unstable, the estimation error of a complex router can exceed its theoretical advantage.

What RouterBench-Mini ultimately preserves is not the most attractive single result, but an auditable experimental path: which methods appeared to work, which findings did not replicate, and which data and experiments are genuinely needed next.
