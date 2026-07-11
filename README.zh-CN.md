# RouterBench-Mini：面向多模态智能体的成本感知模型复用

[English README](README.md)

RouterBench-Mini 研究一个实际的模型选择问题：什么时候复用便宜模型已经足够，什么时候值得调用强模型？项目在统一提示词、解码配置和评分流程下，使用同一 Qwen 3.5 系列的两款多模态模型评测 300 道文本、视觉和工具调用任务。

## V2 主要结果

![V2 准确率与成本权衡](results/qwen3.5-v2-study/pareto.png)

在 240 道独立测试题上，只使用可观察特征的 Task-Aware Router 达到 **80.00%**，比 Always Strong 高 2.08 个百分点，同时平均 API 成本降低 **19.7%**、延迟降低 **9.7%**。它在 68.33% 的任务上使用 Strong，但不读取数据集名称或答案标签。

| 方法 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong 使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 76.67% | 0.00024165 | 1,141 ms | 0.00% |
| Always Strong | 77.92% | 0.00065225 | 1,783 ms | 100.00% |
| **Task-Aware Router** | **80.00%** | 0.00052408 | 1,610 ms | 68.33% |
| Full Calibrated Reflection | 76.67% | 0.00025260 | 1,174 ms | 2.08% |

Reflection 消融中表现最好的是更简单的 response-only 概率校准：准确率 **79.17%**，平均成本 CNY 0.00056005，比 Always Strong 低 14.1%；但由于 Cheap 和 review 串行调用，延迟更高。

### Review-and-Correct 结论

Reflection 触发升级时，Strong 会收到原题、图片/工具定义和 Cheap 候选答案。Strong 必须先独立核验：候选正确则保留，错误才修正。

在 response-only calibrated ablation 触发的 143 次升级中：

| 最终答案策略 | 有效升级 | 有害升级 |
|---|---:|---:|
| 直接用 Strong 独立答案覆盖 | 14 | 8 |
| Strong review-and-correct | 11 | **5** |

Review-and-correct 将 harmful escalation 降低了 **37.5%**，但也少修正了 3 道题，因此它有效但不完美。加入全部任务特征的校准器在 60 道验证题上发生过拟合；项目保留这个负结果，没有根据测试集继续调参。

完整产物位于 [`results/qwen3.5-v2-study`](results/qwen3.5-v2-study) 和 [`results/qwen3.5-v2-ablation`](results/qwen3.5-v2-ablation)。

## 实验设计

### 任务集

构建脚本确定性地生成 300 道题，并按任务大类分层划分为 20% 验证集和 80% 测试集。

| 任务大类 | 数量 | 数据来源 | 评估方式 |
|---|---:|---|---|
| 文本推理 | 100 | GSM8K 40、CommonsenseQA 30、BBH 逻辑题 30 | 数值或选择题准确率 |
| 视觉语言 | 100 | ScienceQA 40、ChartQA 20、OCR-VQA 20、20 个 MMMU 学科 | 选择题、精确匹配或数值容差 |
| 智能体工具调用 | 100 | BFCL V4 simple 50、BFCL V4 multiple 50 | 函数名与必需参数匹配 |

数据来源：[GSM8K](https://huggingface.co/datasets/openai/gsm8k)、[CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)、[BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)、[ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA)、[ChartQA](https://huggingface.co/datasets/docintel/ChartQA)、[OCR-VQA](https://huggingface.co/datasets/pppop7/OCR-VQA)、[MMMU](https://huggingface.co/datasets/MMMU/MMMU) 和 [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)。

### 模型池

两款模型都支持文本、图片和工具调用，因此实验研究的是模型能力与路由，而不是人为制造“文本模型/VLM”边界。

| 角色 | 模型 | Temperature | 最大输出 | Thinking |
|---|---|---:|---:|---|
| Cheap | `qwen3.5-35b-a3b` | 0.2 | 256 tokens | 关闭 |
| Strong | `qwen3.5-397b-a17b` | 0.2 | 256 tokens | 关闭 |

### 四种路由策略

1. **Always Cheap**：所有任务都交给 Cheap。
2. **Always Strong**：所有任务都交给 Strong。
3. **Task-Aware Router**：根据推理时可观察的题目长度、数字、数学/逻辑线索、图片、图表/OCR线索、选项数、工具数、必需参数数和 Schema 深度计算透明风险分数。它不读取 `dataset`、`source`、`rule_tier` 或标准答案，风险阈值只在验证集选择。
4. **Reflection Router**：先调用 Cheap，再用交叉验证 Platt 校准估计 `P(Cheap 回答正确)`，并结合格式和 self-check 信号决定是否升级。升级后由 Strong 执行 review-and-correct，而不是直接覆盖。

阈值选择以验证集准确率最高为第一目标，同准确率时选择成本和 Strong 使用率更低的方案。测试集标签不参与阈值选择。

## 复现实验

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[study,test]"
python scripts/build_manifest.py
python -m pytest
```

无需 API Key 的 smoke test：

```bash
python -m routerbench_mini.cli \
  --manifest data/mini_manifest.jsonl \
  --models configs/models.mock.yaml \
  --out results/mock
```

运行真实 V2：

```bash
export QWEN_API_KEY="your-api-key"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
python scripts/probe_models.py
python scripts/run_study.py --workers 8
python scripts/run_ablations.py --workers 8
```

不要把 API Key 写入 YAML 或提交到 Git。模型名、价格和解码参数位于 [`configs/models.qwen_api.yaml`](configs/models.qwen_api.yaml)。响应按照任务、模型、提示词版本、solve/review 模式、候选答案和解码参数缓存在 `.cache/routerbench/`。

## 仓库结构

```text
configs/                       模型与成本配置
data/                          300 题清单及验证/测试划分
docs/                          实验协议与消融说明
results/qwen3.5-v2-study/      V2 主表、图与误差分析
results/qwen3.5-v2-ablation/   Reflection 消融与反事实比较
scripts/build_manifest.py      数据集构建
scripts/run_study.py           V2 主实验
scripts/run_ablations.py       Reflection 消融
src/routerbench_mini/          Provider、特征、校准、路由和评分
tests/                         单元测试
```

## 项目边界

- 这是一个 300 题、单服务商、单模型系列的小型研究项目。
- 60 道验证题较少，完整特征校准器没有很好泛化。
- Review-and-correct 减少但没有消除有害改写。
- 串行 review 可能比单次 Always Strong 更慢。
- BFCL 评分只检查第一个规范函数调用及其必需参数。
- 一条被两款 API 在生成前同时拒绝的 OCR-VQA 样本会确定性顺延替换；它不进入评分，也不是按正确率筛选。
- API 延迟会受服务负载影响，配置价格也可能变化。

完整 V2 分析见 [`docs/research_note.md`](docs/research_note.md)。V1 历史结果保留在 `results/qwen3.5-study/`；由于视觉配比和温度不同，V1 与 V2 不能直接混为同一组实验。
