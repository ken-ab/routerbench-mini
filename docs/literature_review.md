# Literature Review and Method Rationale

## Scope

RouterBench-Mini is an external model-routing study: a policy chooses between complete Cheap and Strong foundation models. This differs from token-level Mixture-of-Experts routing and from reassembling internal network blocks.

## Closest Routing Work

### Hybrid LLM

Ding et al., [Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing](https://arxiv.org/abs/2404.14618), predict the response-quality gap between a small and large model from the query and select a threshold according to a desired quality level. This directly motivates the V3 `LearnedQualityGapEstimator` and its target:

```text
Strong correctness - Cheap correctness
```

The paper supports learned query-difficulty routing and empirical threshold selection. It does not support fixed constants such as 50 words or four required tool arguments.

### RouteLLM

Ong et al., [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665), train BERT, causal-LM, matrix-factorization, and similarity-weighted routers from pairwise preference labels. They also show that routers degrade under distribution shift and improve when target-like data is added. This motivates both pairwise model labels and the V3-to-V4 confirmation protocol.

### Benchmark-trained routing

Shnitzer et al., [Large Language Model Routing with Benchmark Datasets](https://arxiv.org/abs/2309.15789), formulate model selection as binary classification using benchmark outcomes. Their observation that no single model dominates every task supports per-query selection rather than assuming parameter count is a universal ordering.

### Cascades and post-response verification

Chen et al., [FrugalGPT](https://arxiv.org/abs/2305.05176), learn cascades that trade API cost against performance. Aggarwal et al., [AutoMix](https://arxiv.org/abs/2310.12963), use small-model self-verification and a POMDP router to decide escalation. These works motivate Reflection as a post-response cascade, while also warning that self-verification is noisy.

### Multimodal routing

[VL-RouterBench](https://arxiv.org/abs/2512.23562) evaluates routing over vision-language models using sample-model quality and cost matrices. It reports remaining headroom from finer visual cues and textual structure. RouterBench-Mini currently models textual structure but does not encode image content in the learned router, which is a clear next step.

## Relation to Xingyi Yang's Work

Yang et al., [Deep Model Reassembly](https://arxiv.org/abs/2210.17409), introduce general-purpose model reuse by partitioning heterogeneous pretrained networks and selecting blocks under performance and computational constraints. Its "no single wins for all" observation and constrained selection objective are conceptually aligned with RouterBench-Mini.

The connection is motivational rather than methodological:

- Deep Model Reassembly selects and combines internal pretrained blocks for downstream transfer.
- RouterBench-Mini selects a complete API model for each inference request.
- The paper does not define query-length, OCR, chart, or tool-schema thresholds.

Yang et al., [Mixture of Experts Made Intrinsically Interpretable](https://arxiv.org/abs/2503.07639), concerns token-to-expert routing inside one model and optimizes activation sparsity for interpretability. It should not be cited as direct evidence for an external Cheap-versus-Strong request router.

## Consequences for This Repository

1. Handcrafted Task-Aware is explicitly labeled a transparent heuristic baseline.
2. Learned Cost-Aware predicts the model-pair quality gap from request-time features.
3. Thresholds use outer-fold development predictions rather than fitted-sample scores.
4. V3 is used to select the text-only ablation, and V4 is a new fingerprint-disjoint confirmation set.
5. Test failures are retained. The V3 text-only result did not fully reproduce on V4.
6. Accuracy differences are reported with paired bootstrap intervals; cost savings are not used to imply accuracy equivalence without uncertainty.

## Remaining Method Gap

The development pool has 450 tasks, but only 52 distinguish the two model outcomes. TF-IDF or linear structured features therefore receive a sparse and noisy supervision signal. A stronger study would collect thousands of paired model responses, include lightweight visual embeddings, use repeated generations to reduce label noise, and evaluate on model families and datasets not seen during router training.
