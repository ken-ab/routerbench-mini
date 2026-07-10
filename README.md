# RouterBench-Mini

**面向多模态智能体的成本感知模型复用。**

RouterBench-Mini 是一个小型研究风格基准，用来测试轻量级路由器能否判断何时使用低成本模型已经足够，以及何时应升级到更强的文本模型或视觉语言模型。

研究问题：

> 选择性升级能否在显著降低平均成本的同时，接近强模型的准确率？

## 项目动机

现代智能体系统通常可以访问多种基础模型：低成本文本模型、强推理模型、低成本视觉语言模型，以及更强的视觉语言模型。始终调用最强模型通常更准确，但成本高；始终调用最便宜的模型效率高，但可靠性不足。这个仓库研究两者之间的折中方案：在准确率和成本约束下进行模型路由。

本项目刻意保持精简，适合作为围绕 LLM 路由、模型复用、多模态智能体和高效推理的可复现科研助理申请作品集项目。

## 任务

该基准围绕三类任务设计：

| 任务类型 | 数据集 | 评估方式 |
|---|---|---|
| 文本推理 | GSM8K | 最终数值精确匹配 |
| 多模态问答 | ScienceQA 图像子集 | 多选题准确率 |
| 智能体工具使用 | BFCL 风格函数调用 | 函数名 + 必需参数 |

仓库内置的 `data/mini_manifest.jsonl` 是一个小型 smoke-test 清单。可以用下面的命令构建更大的清单：

```bash
python scripts/build_manifest.py --out data/manifest.jsonl --gsm8k 100 --scienceqa 100 --save-images
```

对于 BFCL，请下载对应的 JSONL 子集，并通过下面的参数传入：

```bash
python scripts/build_manifest.py --out data/manifest.jsonl --gsm8k 100 --scienceqa 100 --bfcl-file path/to/bfcl.jsonl
```

## 模型角色

代码使用模型角色，而不是把实验绑定到某一个提供方：

| 角色 | 示例模型 |
|---|---|
| `cheap_text` | Qwen3-8B |
| `strong_text` | Qwen3-32B / Qwen3-Max |
| `cheap_vlm` | Qwen3-VL-8B |
| `strong_vlm` | Qwen3-VL-32B / Qwen3-VL-235B |

项目包含 mock providers，因此无需 API key 也能运行基准。

## 路由器

| 路由器 | 含义 |
|---|---|
| `always_cheap` | 始终选择兼容的最便宜模型 |
| `always_strong` | 始终选择兼容的最强模型 |
| `rule_based` | 固定人工规则 |
| `selective_escalation` | 先使用低成本模型，再由验证器决定是否升级 |
| `oracle` | 事后上界：选择能答对的最低成本模型 |

主要方法是 `selective_escalation`。

## 运行

安装依赖：

```bash
pip install -e ".[test]"
```

运行 mock 实验：

```bash
python -m routerbench_mini.cli --manifest data/mini_manifest.jsonl --models configs/models.mock.yaml --costs configs/costs.yaml --out results/mock
```

运行测试：

```bash
python -m pytest
```

## 输出

CLI 会写入：

- `results/mock/predictions.csv`
- `results/mock/summary.csv`

最重要的指标包括准确率、平均相对成本、平均延迟和升级率。

mock smoke-test 示例输出：

| 路由器 | 准确率 | 平均成本 | 平均延迟 | 升级率 |
|---|---:|---:|---:|---:|
| `always_cheap` | 0.4167 | 1.3333 | 466.67 ms | 0.0000 |
| `always_strong` | 1.0000 | 9.3333 | 1400.00 ms | 0.0000 |
| `rule_based` | 0.9167 | 6.0000 | 1033.33 ms | 0.0000 |
| `selective_escalation` | 1.0000 | 6.8333 | 1216.67 ms | 0.5833 |
| `oracle` | 1.0000 | 5.6667 | 983.33 ms | 0.5833 |

这张表由确定性的 mock providers 生成，因此仅用于 smoke test。真正的研究表格应使用真实的 Qwen/API providers，并在更大的清单上生成。

## 真实 API 设置

复制示例配置：

```bash
cp configs/models.qwen_api.example.yaml configs/models.qwen_api.yaml
```

设置提供方环境变量：

```bash
set QWEN_API_KEY=your_key
set QWEN_BASE_URL=https://your-openai-compatible-endpoint/v1
```

然后运行：

```bash
python -m routerbench_mini.cli --models configs/models.qwen_api.yaml --manifest data/manifest.jsonl --out results/qwen
```

## 研究笔记

实验框架、指标和建议消融实验见 `docs/research_note.md`。
