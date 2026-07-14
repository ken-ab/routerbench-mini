# RouterBench-Mini：面向多模态智能体的成本感知模型复用

[English README](README.md)

这是一个规模很小、不能独立支撑论文成果的个人实验报告，记录了我在几天内从工程原型逐步走向较规范研究流程的一次尝试。

RouterBench-Mini 研究一个模型选择问题：什么时候复用便宜模型已经足够，什么时候值得调用强模型？项目使用同一Qwen 3.5系列的两款多模态模型，在统一提示词、解码配置、评分和真实成本统计下评测文本、视觉与工具调用任务。它不提出新的基础算法：TF-IDF、Ridge、逻辑回归、概率校准和交叉验证都是经典方法。这个demo的价值主要在于把模型路由问题做成一个可复现的小型benchmark，并诚实记录设计失误、负结果和逐步修正。

## 从V1到V4：研究怎么迭代

V1至V4不是四个越来越复杂的模型。V1和V2是方法开发，V3是探索与特征消融，V4则是V3方法选择完成后的确认性评测。真正的演变同时发生在路由方法和实验可信度上。

| 阶段 | 数据协议 | 主要变化 | 暴露的问题或结论 |
|---|---|---|---|
| V1 原型 | 300题，60验证/240测试 | 数据集标签规则；原始confidence触发Reflection | 标签泄漏；不同任务的confidence不可比较；Reflection低于Cheap |
| V2 修正 | 重建300题，60验证/240测试 | 可观察特征、风险阈值、Platt校准、review-and-correct | 60题同时校准和选阈值，Reflection明显过拟合 |
| V3 探索 | 旧300题开发；全新A组150题测试 | 质量差监督、TF-IDF/Ridge、五折OOF、特征消融 | Text-only在A组匹配Strong，但只是一次候选结果 |
| V4 确认 | 旧300题+A组150题开发；全新B组150题测试 | 冻结Text-only训练方案，在新数据上确认 | 成本优势保留，匹配Strong的准确率结论没有复现 |

### V1：先把真实实验跑通

V1使用`qwen3.5-35b-a3b`和`qwen3.5-397b-a17b`，`temperature=0`、最大输出256 tokens。Task-Aware直接读取数据中预先写入的`rule_tier`：例如GSM8K和逻辑题固定走Strong，CommonsenseQA固定走Cheap。这能跑通流程，却把数据集身份带进了路由决策，无法证明面对新问题也能泛化。

Reflection先调用Cheap，再根据格式、自报confidence和self-check决定是否调用Strong。验证集选出阈值0.8，但可解析的工具调用被程序固定赋值0.75，因此测试中的工具题全部升级；一些错误数学答案却自报0.95或1.0而被接受。测试准确率为Always Cheap 80.00%、Always Strong 81.67%、Task-Aware 81.25%、Reflection 78.75%。V1证明系统可运行，也证明原始confidence不能直接作为统一路由信号。

### V2：去掉数据集标签，尝试校准回答可信度

V2将温度改为0.2，并把视觉任务调整为ScienceQA 40、ChartQA 20、OCR-VQA 20、MMMU 20。Task-Aware不再读取数据集名，而是从题目长度、数字数量、数学/逻辑词、图片、选项和工具Schema复杂度计算可观察风险分数，验证集选择阈值2.0。

Reflection使用Cheap的原始confidence、格式、自检和13个题目特征训练逻辑回归，并用Platt scaling估计`P(Cheap回答正确)`；升级后Strong执行review-and-correct。主实验中Task-Aware为80.00%，Always Strong为77.92%，但Reflection验证集95.00%、测试集76.67%。只有60道验证题、其中Cheap仅错5题，却同时用于拟合校准器和选择阈值，导致明显过拟合。V2消融中的response-only变体达到79.17%，因此后续版本减少了校准器输入并加强样本外阈值选择。

### V3：学习模型质量差，并做特征消融

V3把旧300题全部作为开发数据，另建与其指纹不重叠的A组150题。每道开发题同时由Cheap和Strong回答，确定性评分器生成`y = Strong正确 - Cheap正确`。问题TF-IDF和13个可观察结构特征被输入Ridge，以预测Strong相对Cheap的质量差；五折out-of-fold预测用于选择全局升级阈值，避免在训练样本自身的拟合分数上调参。

V3对Learned Router进行了三种特征消融：Combined为78.67%、Strong使用率36.00%；Structured-only为78.67%、Strong使用率46.67%；Text-only为79.33%、Strong使用率54.67%。Text-only在A组与Always Strong同为79.33%，且成本低32.7%，因此按“准确率优先”的规则被选为下一阶段候选。这个差异只有一道题，不能被当作稳定结论。

V3还把Reflection改为response-only校准，并用外层五折概率选择阈值。它最终升级99/150题，却只有74.00%准确率；review修复5个Cheap错误，同时改坏7个Cheap正确答案。这个负结果说明格式、自报confidence和self-check仍不足以判断语义正确性。

### V4：确认V3结论，而不是发明新算法

V4不是新的路由架构。V3的A组150题在完成方法选择后加入旧300题，形成450题开发集；随后建立另一套完全不同的B组150题作为最终未触碰确认集。Learned Router固定使用V3选出的Text-only方案，再在450题上重新拟合TF-IDF、Ridge和OOF阈值；Reflection也用450题重新校准。

V4中Learned Text-only为80.00%，Always Strong为83.33%。它仍降低32.2%成本，但V3中“匹配Strong”的准确率结果没有复现。因为V3与V4测试题不同，不能把79.33%到80.00%的表面变化归因于开发数据从300增加到450；Always Strong本身也从79.33%变为83.33%。V4真正回答的是“V3选出的方案能否在新题上泛化”，答案是成本收益可以复现，准确率优势不能完全复现。

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

## 两批无重叠held-out评测

V3的A组和V4的B组各包含150道无重叠题目，文本、视觉、工具各50题。A组曾用于选择Text-only并随后加入V4开发集，因此不再是最终未触碰确认集；只有B组承担这个角色。Always Cheap、Always Strong和Task-Aware在两批评测间完全不变，下面的合并统计仅用于检查这三种冻结策略的跨批次稳定性：

| 冻结方法 | 两批合计300题准确率 | 平均成本 | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 77.00% | 0.00024081 | 935 ms | 0.00% |
| Always Strong | 81.33% | 0.00065324 | 2,094 ms | 100.00% |
| **Handcrafted Task-Aware** | **80.67%** | **0.00050602** | **1,536 ms** | **65.67%** |

Task-Aware依然只低0.67个百分点，配对区间为[-2.33,+1.00]；成本降低**22.5%**，延迟降低**26.6%**。这个结论替代了V2基于单次240题测试得出的“Task-Aware超过Strong”表述，更保守，也更可信。

## 最终方法说明与结果解读

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

## 最终反思

这个项目没有把经典机器学习包装成新的路由理论。Learned Router本质上是TF-IDF/结构特征上的Ridge回归，Reflection是逻辑回归概率门控的两阶段级联，阈值通过交叉验证经验选择。它们的算法含金量有限，但迭代过程仍然展示了一个小型研究练习应具备的基本动作：定义可复现任务、建立强弱基线、发现信息泄漏和过拟合、做消融、保留负结果，并用全新确认集检查一次看起来很好的结论。

最终最稳定的方法反而是简单的Handcrafted Task-Aware。Learned Router受Cheap/Strong成对分歧样本稀疏限制；Reflection受自报confidence漂移和Strong review回退影响。这个demo更适合作为一份从开发思维回到研究思维的小型report，而不是论文级贡献。

## 后续实验如何优化

1. **增加真正有路由价值的数据。** 当前450道开发题中只有52题能区分Cheap与Strong。下一轮应主动采样更困难、模型更容易产生分歧的题目，并绘制训练规模学习曲线，而不是只增加大量`y=0`样本。
2. **做受控的数据规模实验。** 在同一个全新测试集上比较“300题训练”和“450题训练”，保持特征、阈值选择和模型调用完全一致，才能判断增加开发数据是否真的有效。
3. **升级问题表示。** 在相同划分上加入固定预训练文本Embedding，与TF-IDF进行配对消融；视觉任务进一步使用轻量多模态编码器或图像Embedding，让回答前Router能够看到图片内容而不只是`has_image`。
4. **使用明确的成本效用目标。** 当前阈值规则是准确率优先、并列时才比较成本。后续可直接优化`accuracy - lambda * cost - mu * latency`，或在允许的准确率下降约束下最小化成本，并在实验前固定`lambda`或约束。
5. **改进Reflection的不确定性信号。** 如果API支持，应比较token log-probability、entropy、多次采样一致性或独立Verifier，减少对提示词自报confidence的依赖；同时保留blind Strong替换作为review-and-correct的反事实基线。
6. **扩大复现范围。** 加入更多模型档位、模型系列和随机采样批次，报告配对置信区间与失败案例；所有方法和超参数冻结后，再建立一套新的最终确认集，避免继续在V4上选择最好结果。

## 实验设计

每个300题数据块包括：

| 任务大类 | 数量 | 数据来源 | 评估方式 |
|---|---:|---|---|
| 文本推理 | 100 | GSM8K 40、CommonsenseQA 30、BBH逻辑题30 | 数值或选择题准确率 |
| 视觉语言 | 100 | ScienceQA 40、ChartQA 20、OCR-VQA 20、MMMU 20 | 选择题、精确匹配或数值容差 |
| 工具调用 | 100 | BFCL V4 simple 50、multiple 50 | 函数名与必需参数匹配 |

A组和B组按一半规模保持相同比例。问题、选项、工具Schema和图片内容哈希共同保证原始300题、A组与B组之间零重叠。

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
