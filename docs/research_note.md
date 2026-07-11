# RouterBench-Mini V4 Research Note

## Research Question

Can request-time features or post-response confidence select between a Cheap and Strong multimodal model while preserving accuracy and reducing measured API cost? Do learned routing and review-and-correct generalize to new samples?

## Protocol Evolution

- V2: 60 validation and 240 test tasks. Used to identify methodological weaknesses.
- V3: old 300 tasks become development; 150 new fingerprint-disjoint tasks become held-out test.
- V3 ablation: combined, structured-only, and text-only quality-gap routers.
- V4: V3 is added to development after model-family selection; another 150 new tasks form the final confirmation set.

The final V4 protocol therefore has 450 development examples and 150 untouched test examples. Every test block contains 50 text, 50 vision, and 50 tool tasks. Query, options, tool schema, and image bytes are included in leakage fingerprints.

## Models and Decoding

- Cheap: `qwen3.5-35b-a3b`.
- Strong: `qwen3.5-397b-a17b`.
- Temperature: 0.2.
- Maximum output: 256 tokens.
- Thinking: disabled.
- Shared structured prompt; native function calling for tools.

## Methods

### Handcrafted Task-Aware

The V2 observable-feature score and threshold 2 are frozen across V3 and V4. This is an interpretable heuristic baseline, not a learned router.

### Learned Cost-Aware

For each development task, the target is:

```text
y = Strong_correct - Cheap_correct, y in {-1, 0, 1}
```

The estimator combines TF-IDF and/or structured request features with Ridge regression. Thresholds are selected from five-fold out-of-fold scores by maximum development accuracy, breaking ties by measured cost and Strong use. The V3 ablation selected text-only for V4.

### Calibrated Reflection

A response-only logistic/Platt model estimates Cheap correctness from prompted confidence, format validity, and self-check. Threshold selection uses outer-fold probabilities and cached development review outcomes. At test time, only escalated tasks call Strong sequentially.

## V4 Confirmatory Results

| Method | Accuracy | Avg. cost | Avg. latency | Strong use |
|---|---:|---:|---:|---:|
| Always Cheap | 0.7867 | 0.00023762 | 1177.61 ms | 0.0000 |
| Always Strong | **0.8333** | 0.00064448 | 2619.40 ms | 1.0000 |
| Handcrafted Task-Aware | **0.8267** | 0.00050139 | 1766.81 ms | 0.6600 |
| Learned Cost-Aware | 0.8000 | **0.00043693** | 2391.25 ms | 0.5000 |
| Calibrated Reflection | 0.8000 | 0.00047537 | 2202.13 ms | 0.4600 |

Task-Aware is the strongest trade-off. Its paired accuracy difference from Always Strong is -0.0067 with a 95% bootstrap interval of [-0.0267, 0.0133]. Its average cost is 22.2% lower.

## Replicated Frozen Policies

Pooling V3 and V4 is valid for Always Cheap, Always Strong, and Task-Aware because these policies remain unchanged:

| Method | Accuracy | 95% accuracy CI | Avg. cost | Avg. latency |
|---|---:|---:|---:|---:|
| Always Cheap | 0.7700 | [0.7200, 0.8167] | 0.00024081 | 934.72 ms |
| Always Strong | 0.8133 | [0.7667, 0.8567] | 0.00065324 | 2093.65 ms |
| Task-Aware | 0.8067 | [0.7600, 0.8500] | 0.00050602 | 1535.95 ms |

The paired Task-Aware-minus-Strong difference is -0.0067 with interval [-0.0233, 0.0100]. Cost savings are 22.5%, and the paired cost-difference interval remains strictly below zero.

## Learned Feature Ablation

On V3:

| Variant | Accuracy | Avg. cost | Strong use |
|---|---:|---:|---:|
| Structured only | 0.7867 | 0.00039014 | 0.4667 |
| Text only | **0.7933** | 0.00044537 | 0.5467 |
| Combined | 0.7867 | **0.00037586** | 0.3600 |

Text-only matches Always Strong on V3 and is selected for V4. On V4 it reaches 0.8000 versus Strong's 0.8333. The advantage does not fully replicate.

The central data issue is sparse pairwise supervision. Among 450 development tasks, only 31 favor Strong, 21 favor Cheap, and 398 are ties. A linear router can learn broad task patterns but has little evidence about the narrow decision boundary.

## Reflection and Review

V4 Reflection escalates 69 tasks. Relative to Cheap, review produces 7 beneficial and 5 harmful changes. Blind independent Strong on the same subset also produces 7 beneficial and 5 harmful outcomes. Review keeps 53 candidates and changes 16.

On V3, review produces 5 beneficial and 7 harmful outcomes, compared with 11 beneficial and 6 harmful for blind Strong. The V2 harmful-change reduction is therefore not robust across replications.

Prompted confidence is also miscalibrated under shift. V3's largest 85-example bin has mean predicted correctness 0.535 but empirical Cheap accuracy 0.776. V4 is better aligned after adding V3 development data, but the signal remains coarse.

## Conclusions

1. The replicated result supports cost-aware reuse: a transparent router retains near-Strong accuracy with about 22.5% lower measured API cost.
2. The earlier claim that routing exceeds Always Strong does not survive the stricter replication framing.
3. Learned quality-gap routing reduces Strong use substantially but needs more pairwise-disagreement data to match the handcrafted baseline reliably.
4. Prompted confidence and review-and-correct are not stable enough to serve as the headline policy.
5. The most valuable next experiment is larger paired-response collection with multimodal embeddings and repeated model samples.
