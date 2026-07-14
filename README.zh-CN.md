# RouterBench-Mini：面向多模态智能体的成本感知模型复用

[English README](README.md)

这是一个规模很小、不能独立支撑论文成果的个人实验报告，记录了我从工程实践逐步转回科研的一次小型实验。

我做这个实验，来自一个很直接的疑问：如果所有问题都交给便宜模型，复杂任务的准确率可能不足；如果所有问题都交给强模型，API成本和延迟又会持续增加。RouterBench-Mini研究的就是这个模型选择问题：什么时候使用便宜模型已经足够，什么时候值得调用更强模型？

实验从始至终使用同一Qwen 3.5系列的两款统一多模态模型，在相同提示词、解码配置、评分规则和真实API成本统计下完成比较。项目经过V1至V4四个阶段；每一版都保留了当时的问题、结果和下一步，而不是只展示最好的一次运行。

## 实验问题与统一设置

### 模型池

| 角色 | 模型 | 定位 | 输入/输出价格（CNY/百万tokens） |
|---|---|---|---:|
| Cheap | `qwen3.5-35b-a3b` | 较小、较便宜的统一多模态模型 | 0.4 / 3.2 |
| Strong | `qwen3.5-397b-a17b` | 更大、能力更强但更贵的统一多模态模型 | 1.2 / 7.2 |

两者都能处理文本、图片和工具调用，因此这里研究的是同一能力边界下的模型规模选择，而不是人为区分“文本模型”和“视觉模型”。V1使用`temperature=0`；V2至V4使用`temperature=0.2`；所有版本最大输出均为256 tokens并关闭thinking。

### 三个任务族与五种任务格式

| 任务族 | 任务格式 | 数据来源 | 评分方式 |
|---|---|---|---|
| 文本 | 数学推理 | GSM8K | 最终数值匹配 |
| 文本 | 文本选择题 | CommonsenseQA、BBH逻辑推理 | 选项准确率 |
| 视觉 | 视觉选择题 | ScienceQA、MMMU | 选项准确率 |
| 视觉 | 开放视觉问答 | ChartQA、OCR-VQA | 规范化文本或数值容差 |
| 工具 | 工具调用 | BFCL V4 simple、multiple | 函数名与标准答案必需参数匹配 |

标准答案只由确定性评分器使用，不会提供给Router或Qwen模型。项目比较Always Cheap、Always Strong、Task-Aware、Reflection，以及V3后新增的Learned Cost-Aware五类策略。

## V1：基础规则版

### 数据与架构

V1共300题，按文本100、视觉100、工具100构建，再分成60题验证集和240题测试集。文本包括GSM8K 40、CommonsenseQA 30、BBH逻辑题30；视觉包括ScienceQA 80、ChartQA 10、OCR-VQA 10；工具包括BFCL simple 50和multiple 50。

V1建立了四个基础方法：Always Cheap和Always Strong提供成本与能力上下界；Task-Aware读取数据中预先写好的`rule_tier`，例如GSM8K和逻辑题固定使用Strong，CommonsenseQA固定使用Cheap；Reflection先调用Cheap，再根据答案格式、自报confidence和self-check决定是否升级到Strong。Reflection在60题验证集上选择confidence阈值0.8。

### V1测试结果

| 方法 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 80.00% | 0.00023496 | 707 ms | 0.00% |
| Always Strong | **81.67%** | 0.00063335 | 1,412 ms | 100.00% |
| Task-Aware | 81.25% | 0.00044290 | 1,063 ms | 49.17% |
| Reflection | 78.75% | 0.00057631 | 1,317 ms | 33.33% |

V1暴露了两个核心问题。第一，Task-Aware根据数据集身份决策，存在明显的信息泄漏，不能代表面对新题时的真实路由。第二，不同任务的confidence不可直接比较：可解析工具调用被程序固定赋值0.75，因此在阈值0.8下全部升级；一些错误数学答案却自报0.95甚至1.0而被接受。Reflection成本接近Strong，准确率却低于Cheap，这推动了V2对特征和置信度的重新设计。

## V2：可观察特征与概率校准

### V2解决了什么

V2取消`rule_tier`，只允许Router使用推理时可见的信息。Task-Aware从题目长度、数字数量、数学与逻辑词、图片、选项数量、工具数量、必需参数和Schema深度计算风险分数，并在验证集选择风险阈值2.0。

数据仍为300题和60/240划分，但视觉部分改为ScienceQA 40、ChartQA 20、OCR-VQA 20、MMMU 20，使视觉任务不再被ScienceQA单一主导。Reflection使用Cheap的原始confidence、格式、自检和13个请求特征训练逻辑回归，再用Platt scaling估计`P(Cheap回答正确)`；升级时Strong看到原题和Cheap候选，执行review-and-correct。

### V2测试结果

| 方法 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 76.67% | 0.00024165 | 1,141 ms | 0.00% |
| Always Strong | 77.92% | 0.00065225 | 1,783 ms | 100.00% |
| **Task-Aware** | **80.00%** | 0.00052408 | 1,610 ms | 68.33% |
| Reflection Full | 76.67% | **0.00025260** | **1,174 ms** | 2.08% |

V2中的Task-Aware超过Always Strong，但这个结论只来自一次240题测试。Reflection在验证集达到95.00%，测试却只有76.67%，显示明显过拟合。

### V2 Reflection消融

| 变体 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Format-only | 76.67% | **0.00024165** | **1,141 ms** | 0.00% |
| Raw confidence | 76.67% | 0.00024524 | 1,149 ms | 0.42% |
| **Calibrated response-only** | **79.17%** | 0.00056005 | 2,060 ms | 59.58% |
| Full：response + 13题目特征 | 76.67% | 0.00025260 | 1,174 ms | 2.08% |

60道验证题中Cheap只错5题，却要同时拟合概率校准器和选择升级阈值。加入13个题目特征后，Full校准器把大量测试错误答案估计为高概率，最终几乎不升级。Response-only消融更好，但仍需要更大的开发集和真正的样本外阈值选择，这成为V3的直接起点。

## V3：学习式质量差路由

### 文献启发与架构

V3参考[Hybrid LLM](https://arxiv.org/abs/2404.14618)的质量差预测和[RouteLLM](https://arxiv.org/abs/2406.18665)的成对模型路由思想，将“什么时候用Strong”从手工加分改为监督学习。[FrugalGPT](https://arxiv.org/abs/2305.05176)和[AutoMix](https://arxiv.org/abs/2310.12963)则为回答后升级的Reflection级联提供了方法背景。

旧300题全部转为开发集，另建与其指纹不重叠的A组150题，包含文本、视觉和工具各50题。Learned Cost-Aware的训练流程是：

```text
300道开发题
  -> Cheap和Strong分别回答
  -> 确定性评分生成 y = Strong正确 - Cheap正确
  -> 问题TF-IDF和13维可观察结构特征
  -> Ridge预测Strong相对Cheap的质量差
  -> 五折out-of-fold分数选择全局阈值
  -> 在A组150道新题回答前选择Cheap或Strong
```

300题中，Strong独自正确18题、Cheap独自正确14题、两者结果相同268题。V3的主要Learned版本使用Combined特征和阈值0.04986。Reflection则只用Cheap回答侧的confidence、格式与自检拟合正确率校准器，通过外层五折概率选择阈值0.65。

### V3主实验结果

| 方法 | 准确率 | 95% Bootstrap区间 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 75.33% | [68.00, 82.00] | 0.00024400 | 692 ms | 0.00% |
| Always Strong | **79.33%** | [72.67, 85.33] | 0.00066199 | 1,568 ms | 100.00% |
| Task-Aware | 78.67% | [72.00, 85.33] | 0.00051066 | 1,305 ms | 65.33% |
| Learned Combined | 78.67% | [72.00, 85.33] | **0.00037586** | **1,031 ms** | 36.00% |
| Reflection | 74.00% | [66.67, 81.33] | 0.00063306 | 1,732 ms | 66.00% |

Learned Combined只比Always Strong少答对1题，却少调用96次Strong并降低43.2%成本。Reflection升级99/150题，但review只修复5个Cheap错误，同时改坏7个Cheap正确答案，最终低于Always Cheap。

### V3 Learned特征消融

| 特征版本 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Combined：TF-IDF + 13结构特征 | 78.67% | **0.00037586** | **1,031 ms** | **36.00%** |
| Structured-only | 78.67% | 0.00039014 | 1,117 ms | 46.67% |
| **Text-only TF-IDF** | **79.33%** | 0.00044537 | 1,191 ms | 54.67% |

Text-only在A组与Always Strong同为79.33%，成本低32.7%，因此按照“准确率优先、并列时成本优先”的规则进入V4。但它只比Combined多答对1题，却多调用28次Strong；这是一项需要新数据确认的候选结果，而不是稳定结论。

## V4：确认性评测

### V4解决了什么

V4不是新的路由算法，而是V3方法选择完成后的确认阶段。A组150题在完成V3分析后加入原始300题，形成450题开发集；另一套与前述数据指纹完全不重叠的B组150题成为最终未触碰确认集。

Learned Router固定使用V3选出的Text-only训练方案，再在450题上重新拟合TF-IDF、Ridge和五折阈值0.02606。Reflection保持相同response-only架构，在450题上重新校准并选择阈值0.75。V4测试集不参与特征模式、模型或阈值选择。

### V4最终结果

![V4准确率与成本权衡](results/qwen3.5-v4-study/pareto.png)

| 方法 | 准确率 | 95% Bootstrap区间 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 78.67% | [72.00, 84.68] | 0.00023762 | 1,178 ms | 0.00% |
| Always Strong | **83.33%** | [77.33, 89.33] | 0.00064448 | 2,619 ms | 100.00% |
| **Task-Aware** | **82.67%** | [76.67, 88.00] | 0.00050139 | **1,767 ms** | 66.00% |
| Learned Text-only | 80.00% | [73.33, 86.00] | **0.00043693** | 2,391 ms | 50.00% |
| Reflection | 80.00% | [73.33, 86.00] | 0.00047537 | 2,202 ms | 46.00% |

Task-Aware只比Always Strong低0.67个百分点，配对差值95%区间为[-2.67,+1.33]，同时成本降低22.2%、观测延迟降低32.5%。Learned Text-only仍降低32.2%成本，但比Strong低3.33个百分点；V3中“匹配Strong”的准确率结果没有复现。

不能用V3的79.33%和V4的80.00%证明增加开发数据有效，因为两版测试题不同，Always Strong本身也从79.33%变为83.33%。若要研究300题与450题训练规模的影响，必须让两个训练版本在同一批全新测试题上比较。

### 两批无重叠held-out评测

A组和B组各包含150道无重叠题目，但A组参与了Text-only选择并随后加入V4开发集，因此只有B组是最终未触碰确认集。下面只合并V3与V4之间完全未改变的Always Cheap、Always Strong和Task-Aware，用于观察基线的跨批次稳定性。

| 冻结方法 | 两批合计300题准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 77.00% | 0.00024081 | 935 ms | 0.00% |
| Always Strong | **81.33%** | 0.00065324 | 2,094 ms | 100.00% |
| **Task-Aware** | 80.67% | **0.00050602** | **1,536 ms** | 65.67% |

Task-Aware跨两批只比Strong低0.67个百分点，配对区间为[-2.33,+1.00]；成本降低22.5%，延迟降低26.6%。这比V2单次测试中“Task-Aware超过Strong”的说法更保守，也更可信。

## 我从这个实验得到什么

这个项目使用的TF-IDF、Ridge、逻辑回归、概率校准和交叉验证都是经典机器学习方法，不能被包装成新的路由理论。对我而言，它的价值在于完整经历了一次小型研究迭代：从提出准确率与成本问题、建立Cheap/Strong基线，到发现标签泄漏和校准过拟合，加入学习式路由与消融，再用全新确认集推翻一次看起来很好的结果。

最终最稳定的方法反而是简单的Task-Aware。Learned Router受Cheap/Strong成对分歧样本过少限制；Reflection受提示词自报confidence漂移和Strong review回退影响。这个demo更适合作为一份从工程实践逐步转回科研的个人小型report，而不是论文级算法贡献。

## 后续实验如何优化

1. **增加真正有路由价值的数据。** 当前450道开发题中只有52题能区分Cheap与Strong。下一轮应主动采样更困难、模型更容易产生分歧的任务，并绘制学习曲线，而不是主要增加`y=0`样本。
2. **做受控的数据规模实验。** 在同一个全新测试集上比较300题训练与450题训练，保持特征、阈值选择和模型调用完全一致，才能判断增加开发数据是否有效。
3. **升级问题表示。** 在相同划分上比较固定预训练文本Embedding与TF-IDF；视觉任务进一步加入轻量多模态编码器或图像Embedding，让回答前Router看到图片内容，而不只是`has_image`。
4. **使用明确的成本效用目标。** 当前阈值选择以准确率优先、成本仅用于打破并列。后续可预先固定`accuracy - lambda * cost - mu * latency`，或在允许的准确率下降约束下最小化成本。
5. **改进Reflection的不确定性信号。** 如果API支持，应比较token log-probability、entropy、多次采样一致性或独立Verifier，减少对提示词自报confidence的依赖，并始终保留blind Strong替换作为review的反事实基线。
6. **扩大模型与数据复现。** 加入更多模型档位、模型系列和随机采样批次，报告配对区间与失败案例；所有方法和超参数冻结后，再建立新的最终确认集。

## 与文献的关系

除V3直接使用的[Hybrid LLM](https://arxiv.org/abs/2404.14618)、[RouteLLM](https://arxiv.org/abs/2406.18665)、[FrugalGPT](https://arxiv.org/abs/2305.05176)和[AutoMix](https://arxiv.org/abs/2310.12963)外，[LLM Routing with Benchmark Datasets](https://arxiv.org/abs/2309.15789)提供了路由benchmark背景；[Deep Model Reassembly](https://arxiv.org/abs/2210.17409)支持性能与资源约束下的模型复用动机，但不直接证明本项目的手工阈值。

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
- TF-IDF不读取图像内容，回答前Router对视觉难度的表示有限。
- 提示词自报confidence不能替代token-level或模型内部不确定性。
- API延迟包含远端排队波动，成本结论比延迟结论稳定。
- BFCL评分只检查第一个规范函数调用及标准答案要求的参数。
- 公共数据集revision尚未固定，未来重建可能需要更新脚本。

主要产物位于[`results/qwen3.5-v4-study`](results/qwen3.5-v4-study)，跨批次冻结策略结果位于[`results/qwen3.5-confirmatory`](results/qwen3.5-confirmatory)，V3特征消融位于[`results/qwen3.5-v3-ablation`](results/qwen3.5-v3-ablation)。
