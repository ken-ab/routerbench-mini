# RouterBench-Mini：面向多模态智能体的成本感知模型复用

[English README](README.md)

RouterBench-Mini 研究一个模型选择问题：什么时候复用便宜模型已经足够，什么时候值得调用强模型？项目使用同一Qwen 3.5系列的两款多模态模型，在统一提示词、解码配置、评分和真实成本统计下评测文本、视觉与工具调用任务。

当前版本新增学习式质量差路由、out-of-fold阈值选择、两组互不重叠的独立测试、配对Bootstrap区间，以及更严格的review-and-correct分析。

## V4确认性结果

![V4准确率与成本权衡](results/qwen3.5-v4-study/pareto.png)

V4是最终确认集：前450题用于路由开发，V4的150题与此前数据指纹完全不重叠，并且只在方法冻结后评估一次。

| 方法 | 准确率 | 95% Bootstrap区间 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 78.67% | [72.00, 84.68] | 0.00023762 | 1,178 ms | 0.00% |
| Always Strong | **83.33%** | [77.33, 89.33] | 0.00064448 | 2,619 ms | 100.00% |
| **Handcrafted Task-Aware** | **82.67%** | [76.67, 88.00] | 0.00050139 | **1,767 ms** | 66.00% |
| Learned Cost-Aware | 80.00% | [73.33, 86.00] | **0.00043693** | 2,391 ms | 50.00% |
| Calibrated Reflection | 80.00% | [73.33, 86.00] | 0.00047537 | 2,202 ms | 46.00% |

冻结的Task-Aware是最稳健的权衡点：V4上仅比Always Strong低0.67个百分点，配对差值95%区间为[-2.67,+1.33]，同时成本降低**22.2%**、观测延迟降低**32.5%**。

## 跨两次独立测试的结果

V3和V4各包含150道新题，文本、视觉、工具各50题。三种冻结策略在两次实验间完全不变，因此可以合并统计：

| 冻结方法 | 300道独立测试准确率 | 平均成本 | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 77.00% | 0.00024081 | 935 ms | 0.00% |
| Always Strong | 81.33% | 0.00065324 | 2,094 ms | 100.00% |
| **Handcrafted Task-Aware** | **80.67%** | **0.00050602** | **1,536 ms** | **65.67%** |

Task-Aware依然只低0.67个百分点，配对区间为[-2.33,+1.00]；成本降低**22.5%**，延迟降低**26.6%**。这个结论替代了V2基于单次240题测试得出的“Task-Aware超过Strong”表述，更保守，也更可信。

## 本轮自动优化做了什么

### 学习式路由

新增的`LearnedQualityGapEstimator`在生成前预测：

```text
Strong准确率 - Cheap准确率
```

它使用问题TF-IDF和/或可观察结构特征、Ridge正则化以及五折out-of-fold预测选择阈值。路由器看不到数据集名称和测试答案。

结果有价值，但不是单向成功：

- V3组合特征：78.67%，比Always Strong节省43.2%成本，Strong使用率36%。
- V3 text-only消融：79.33%，与Always Strong相同，成本低32.7%。
- 在全新V4上确认text-only：80.00%，Always Strong为83.33%，成本仍低32.2%。

因此V3的优势没有完全复现。450道开发题中只有52题能区分两个模型，学习目标过于稀疏。项目保留这个负结果，没有继续利用V4调参。

### Reflection与review

Reflection在开发集上拟合response-only正确率校准器，并使用外层交叉验证概率选择升级阈值。触发升级时，Strong收到原题、图片/工具和Cheap候选，候选正确则保留，错误才修改。

这个机制没有稳定优于直接使用Strong。V4中，review和盲目Strong覆盖都是7次有效升级、5次有害升级；V3中review反而少修正错误并多产生一次有害升级。模型通过提示词自报的confidence还出现跨数据漂移，因此Reflection保留为agentic诊断，而不是主方法。

## 实验设计

每个300题数据块包括：

| 任务大类 | 数量 | 数据来源 | 评估方式 |
|---|---:|---|---|
| 文本推理 | 100 | GSM8K 40、CommonsenseQA 30、BBH逻辑题30 | 数值或选择题准确率 |
| 视觉语言 | 100 | ScienceQA 40、ChartQA 20、OCR-VQA 20、MMMU 20 | 选择题、精确匹配或数值容差 |
| 工具调用 | 100 | BFCL V4 simple 50、multiple 50 | 函数名与必需参数匹配 |

V3和V4按一半规模保持相同比例。问题、选项、工具Schema和图片内容哈希共同保证开发集、V3与V4之间零重叠。

### 模型池

| 角色 | 模型 | Temperature | 最大输出 | Thinking |
|---|---|---:|---:|---|
| Cheap | `qwen3.5-35b-a3b` | 0.2 | 256 tokens | 关闭 |
| Strong | `qwen3.5-397b-a17b` | 0.2 | 256 tokens | 关闭 |

两个模型都支持文本、图片和工具，因此研究的是能力与成本选择，而不是人为制造“文本模型/VLM”边界。

### 五种策略

1. **Always Cheap**：固定使用Cheap。
2. **Always Strong**：固定使用Strong。
3. **Handcrafted Task-Aware**：使用可观察线索和冻结风险阈值2。
4. **Learned Cost-Aware**：从开发集的模型成对表现学习质量差。
5. **Calibrated Reflection**：先调用Cheap，再按校准概率决定是否让Strong审查修正。

手工特征的具体常数是heuristic，不是论文推导。学习式路由是更原则化的替代方案，但在当前小样本、稀疏分歧条件下，手工基线反而泛化更稳定。

## 文献关系

本项目参考[Hybrid LLM](https://arxiv.org/abs/2404.14618)、[RouteLLM](https://arxiv.org/abs/2406.18665)和[LLM Routing with Benchmark Datasets](https://arxiv.org/abs/2309.15789)中的质量差/偏好路由；Reflection对应[FrugalGPT](https://arxiv.org/abs/2305.05176)和[AutoMix](https://arxiv.org/abs/2310.12963)的级联思路。[Deep Model Reassembly](https://arxiv.org/abs/2210.17409)支持性能与资源约束下的模型复用动机，但不能作为手工问题阈值的直接依据。

详细文献综述和导师视角审查见[`docs/literature_review.md`](docs/literature_review.md)与[`docs/supervisor_review.zh-CN.md`](docs/supervisor_review.zh-CN.md)。

## 复现

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[study,test]"
python scripts/build_manifest.py
python scripts/build_v3_data.py
python -m pytest
```

配置`QWEN_API_KEY`和`QWEN_BASE_URL`后：

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

API Key不会写入仓库。响应缓存在`.cache/routerbench/`，缓存身份包括任务、模型、提示词版本、solve/review模式、候选答案和解码参数。

## 局限

- 当前只使用一个服务商、一个模型系列和600道采样任务。
- 两模型真正产生不同正确性的开发样本很少，限制学习式路由。
- TF-IDF不读取图像内容，未来应使用轻量多模态编码器。
- 提示词自报confidence不能替代token-level或模型内部不确定性。
- API延迟包含远端排队波动，成本结论比延迟结论稳定。
- BFCL只检查第一个规范函数调用及其必需参数。
- 公共数据集revision尚未固定，未来重建可能需要更新脚本。

主要产物位于[`results/qwen3.5-v4-study`](results/qwen3.5-v4-study)，跨复现冻结策略结果位于[`results/qwen3.5-confirmatory`](results/qwen3.5-confirmatory)，V3学习特征消融位于[`results/qwen3.5-v3-ablation`](results/qwen3.5-v3-ablation)。
