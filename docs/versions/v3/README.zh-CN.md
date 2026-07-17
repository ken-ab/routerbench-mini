# RouterBench-Mini V3：学习式质量差路由

[English](README.md) | **简体中文** | [返回最新版](../../../README.zh-CN.md)

V3首次加入Learned Cost-Aware Router，并把模型选择从手工分数改为监督学习问题。对应主要提交为[`2f0510c`](https://github.com/ken-ab/routerbench-mini/tree/2f0510c)。

## 协议

旧300题全部作为开发集；另建与其精确指纹不重叠的A组150题：文本50、视觉50、工具50。所有学习器和阈值只使用开发集，A组用于一次测试。

模型仍为`qwen3.5-35b-a3b`与`qwen3.5-397b-a17b`，使用`temperature=0.2`、`max_tokens=256`、thinking关闭。

## Learned Router架构

1. Cheap和Strong分别回答300道开发题。
2. 确定性评分得到两模型是否正确。
3. 构造`y = Strong正确 - Cheap正确`。
4. 用TF-IDF unigram/bigram和13维可观察结构特征表示题目。
5. `Ridge(alpha=0.1)`预测Strong质量优势。
6. 五折OOF预测选择阈值；主Combined阈值为`0.04986`。
7. 在完整300题上重训后，对A组回答前路由。

开发标签只有18个`+1`、14个`-1`和268个`0`，真正提供模型分歧监督的题仅32道。

Reflection改为response-only校准，只读取confidence、格式和self-check；五折外层预测选择阈值`0.65`，升级后执行一次Strong review。

## 主结果

| 方法 | Accuracy | 95% CI | Avg Cost（CNY） | Avg Latency | Strong Rate |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 75.33% | [68.00, 82.00] | 0.00024400 | 692 ms | 0.00% |
| Always Strong | **79.33%** | [72.67, 85.33] | 0.00066199 | 1,568 ms | 100.00% |
| Task-Aware | 78.67% | [72.00, 85.33] | 0.00051066 | 1,305 ms | 65.33% |
| Learned Combined | 78.67% | [72.00, 85.33] | **0.00037586** | **1,031 ms** | 36.00% |
| Reflection | 74.00% | [66.67, 81.33] | 0.00063306 | 1,732 ms | 66.00% |

Learned Combined只比Always Strong少答对1题，却把Strong使用率降到36%。Reflection升级99题，只修复5个Cheap错误，却把7个正确Cheap答案改错，最终低于Always Cheap。

## Learned特征消融

| 特征版本 | Accuracy | Avg Cost（CNY） | Avg Latency | Strong Rate |
|---|---:|---:|---:|---:|
| Combined | 78.67% | **0.00037586** | **1,031 ms** | **36.00%** |
| Structured-only | 78.67% | 0.00039014 | 1,117 ms | 46.67% |
| **Text-only** | **79.33%** | 0.00044537 | 1,191 ms | 54.67% |

Text-only在A组匹配Always Strong，同时成本降低32.7%。但A组已经用于比较三个特征版本并选择Text-only，因此它不再是完全未触碰的最终确认集。

## 暴露的问题

1. 300题中只有32个非零质量差标签，Learned Router容易受个别分歧题影响。
2. Text-only只比Combined多答对1题，可能是特征选择偶然性。
3. Reflection的Strong review既能修复也能回退，不能假设升级总有益。
4. A组参与方法选择，必须再建立新的确认集验证Text-only结果。

V4因此冻结Text-only方案，把A组并入开发数据，并使用全新B组做确认性评测。

## 文件

- [主结果](../../../results/qwen3.5-v3-study/test_summary.csv)
- [Learned消融](../../../results/qwen3.5-v3-ablation/test_summary.csv)
- [实验元数据](../../../results/qwen3.5-v3-study/study_metadata.json)
- [配对比较](../../../results/qwen3.5-v3-study/paired_comparisons.csv)

## 版本导航

[V1](../v1/README.zh-CN.md) · [V2](../v2/README.zh-CN.md) · [V3](README.zh-CN.md) · [V4](../v4/README.zh-CN.md) · [V5](../v5/README.zh-CN.md)
