# RouterBench-Mini Research Note

## Question

The project studies a practical model-reuse question:

> Can lightweight routing and verification signals decide when a cheap model is sufficient and when an agent should escalate to a stronger LLM or VLM?

This is not intended to be a paper-scale benchmark. It is a small, reproducible RA-application project showing competence in model routing, multimodal agents, efficient inference, and evaluation design.

## Task Mix

The planned full manifest contains three task families:

1. Text reasoning: GSM8K, evaluated by final numeric exact match.
2. Multimodal question answering: image-grounded ScienceQA, evaluated by multiple-choice accuracy.
3. Agentic tool use: BFCL-style function calling, evaluated by function name and required argument match.

The included `data/mini_manifest.jsonl` is a tiny smoke-test set. Use `scripts/build_manifest.py` to build a larger manifest from public datasets.

## Model Pool

The benchmark uses roles rather than hard-coded model names:

- `cheap_text`: Qwen3-8B or another low-cost text model.
- `strong_text`: Qwen3-32B, Qwen3-Max, or another stronger text model.
- `cheap_vlm`: Qwen3-VL-8B or another low-cost vision-language model.
- `strong_vlm`: Qwen3-VL-32B/235B or another stronger VLM.

This keeps the experiment portable across local inference, DashScope, OpenRouter, vLLM, and other OpenAI-compatible APIs.

## Router Baselines

1. `always_cheap`: use the cheapest compatible model.
2. `always_strong`: use the strongest compatible model.
3. `rule_based`: use a fixed human-written rule.
4. `selective_escalation`: ask a cheap model first, verify its output format/confidence, then escalate only when needed.
5. `oracle`: post-hoc upper bound that chooses the cheapest correct model.

The main research comparison is whether `selective_escalation` approaches `always_strong` accuracy at much lower average cost.

## Metrics

The primary metrics are:

- Accuracy
- Average relative cost
- Average latency
- Escalation rate

A useful result is not simply the highest accuracy. A useful result is a better accuracy-cost trade-off, such as retaining most of the strong model accuracy while using much less strong-model budget.

## Suggested Ablations

For the first real experiment, run these ablations:

1. Remove confidence signal: escalate only on invalid format.
2. Remove deterministic verifier: escalate only on model self-confidence.
3. Change confidence threshold from 0.45 to 0.65.
4. Compare `cheap_text -> strong_text` against `cheap_vlm -> strong_vlm` for image tasks.

These ablations help answer which routing signals are actually useful.

