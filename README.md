# RouterBench-Mini: Cost-Aware Model Reuse for Multimodal Agents

[中文说明](README.zh-CN.md)

RouterBench-Mini is a compact, reproducible study of a practical routing question:

> When is a smaller multimodal model sufficient, and when should an agent pay to use a stronger model?

The benchmark evaluates two models from the same Qwen 3.5 family on 300 text, vision, and tool-use tasks. It compares four inference policies under one prompt, decoding configuration, and scoring pipeline. The focus is the accuracy-cost-latency trade-off, not building a large agent framework.

## Main Result

![Accuracy-cost trade-off](results/qwen3.5-study/pareto.png)

On the held-out 240-task test set, a simple task-aware router matched the strong model within **0.42 percentage points** while reducing average API cost by **30.1%** and latency by **24.7%**. The reflection router did not improve the trade-off: uncalibrated self-reported confidence caused both false accepts and unnecessary escalations.

| Method | Accuracy | Avg. cost/task (CNY) | Avg. latency | Strong-model use |
|---|---:|---:|---:|---:|
| Always Cheap | 80.00% | 0.00023496 | 707 ms | 0.00% |
| Always Strong | **81.67%** | 0.00063335 | 1,412 ms | 100.00% |
| Task-Aware Router | 81.25% | 0.00044290 | 1,063 ms | 49.17% |
| Reflection Router | 78.75% | 0.00057631 | 1,317 ms | 33.33% |

The key result is deliberately not hidden: reflection underperformed. It made 35 false accepts and 67 unnecessary escalations. For this small study, task identity was more useful than asking a model how confident it felt.

Full artifacts are in [`results/qwen3.5-study`](results/qwen3.5-study), including per-category summaries, the validation threshold sweep, experiment metadata, and [error analysis](results/qwen3.5-study/error_analysis.md).

## Experimental Design

### Tasks

The deterministic builder creates 300 tasks and uses a category-stratified 20% validation / 80% test split.

| Category | Count | Sources | Scoring |
|---|---:|---|---|
| Text reasoning | 100 | 40 GSM8K, 30 CommonsenseQA, 30 BBH logical deduction | numeric or multiple-choice accuracy |
| Vision-language | 100 | 80 ScienceQA, 10 ChartQA, 10 OCR-VQA | multiple-choice, exact match, or numeric tolerance |
| Agentic tool use | 100 | 50 BFCL V4 simple, 50 BFCL V4 multiple | function name and required arguments |

Source datasets: [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa), [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [ChartQA](https://huggingface.co/datasets/docintel/ChartQA), [OCR-VQA](https://huggingface.co/datasets/pppop7/OCR-VQA), and [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).

### Model Pool

Both models accept text, images, and tools, so the experiment studies model capacity and routing rather than an artificial text-model/VLM boundary.

| Role | Model | Temperature | Max output | Thinking |
|---|---|---:|---:|---|
| Cheap | `qwen3.5-35b-a3b` | 0 | 256 tokens | disabled |
| Strong | `qwen3.5-397b-a17b` | 0 | 256 tokens | disabled |

The OpenAI-compatible provider uses the same prompt template and structured output contract for both roles. Native function calling is used for tool tasks. Responses are cached locally by task, model, prompt, and decoding settings so analysis can be reproduced without paying for duplicate calls.

### Routing Policies

1. **Always Cheap** sends every task to the cheaper model.
2. **Always Strong** sends every task to the stronger model.
3. **Task-Aware Router** uses fixed, predeclared task difficulty rules. GSM8K, BBH, ChartQA, OCR, and BFCL-multiple are assigned to the strong tier; easier subsets use the cheap tier.
4. **Reflection Router** first calls the cheap model, checks output validity, self-check status, and confidence, then escalates when verification fails. Its confidence threshold is selected on the 60-example validation split, never on the test split.

## Reproduce

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,analysis,test]"
python scripts/build_manifest.py
```

The builder downloads the public source datasets, writes `data/manifest.jsonl`, `data/validation.jsonl`, and `data/test.jsonl`, and stores 100 images under the git-ignored `data/images/` directory.

For a no-key smoke test:

```bash
python -m routerbench_mini.cli \
  --manifest data/mini_manifest.jsonl \
  --models configs/models.mock.yaml \
  --out results/mock
python -m pytest
```

For the real Qwen experiment:

```bash
export QWEN_API_KEY="your-api-key"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
python scripts/probe_models.py
python scripts/run_study.py --workers 8
```

Never place an API key in a YAML file or commit it to Git. Model names, prices, and decoding settings are defined in [`configs/models.qwen_api.yaml`](configs/models.qwen_api.yaml).

## Repository Layout

```text
configs/                    Model and cost configurations
data/                       300-task manifest and validation/test splits
docs/research_note.md       Protocol, analysis, and research takeaways
results/qwen3.5-study/      Headline tables, plot, and error analysis
scripts/build_manifest.py   Deterministic public-dataset builder
scripts/probe_models.py     Text, vision, and tool-call API probes
scripts/run_study.py        End-to-end experiment and analysis
src/routerbench_mini/       Providers, routers, verifiers, scoring, metrics
tests/                      Unit tests
```

## Limitations

- This is a 300-example portfolio study, not a comprehensive benchmark.
- Results cover one provider and one model family; they should not be generalized to all multimodal models.
- BFCL scoring is simplified to the first canonical call and required arguments.
- API latency varies with service load, and configured token prices can become stale.
- The 60-example validation set is too small for a complex learned router.

A natural next step is a calibrated lightweight router using task family, answer validity, and cheap-model confidence, trained only on validation data. The full protocol and interpretation are in [`docs/research_note.md`](docs/research_note.md).
