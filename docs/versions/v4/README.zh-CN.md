# RouterBench-Mini V4：确认性评测

[English](README.md) | **简体中文** | [返回最新版](../../../README.zh-CN.md)

V4不是新算法版本，而是对V3方法选择结果的确认性实验。对应公开文档提交为[`a9db8ef`](https://github.com/ken-ab/routerbench-mini/tree/a9db8ef)。

## 协议

- 开发集：旧300题 + V3 A组150题，共450题。
- 确认集：全新B组150题，文本、视觉、工具各50题。
- Learned：冻结V3选出的Text-only TF-IDF，`Ridge(alpha=0.1)`，五折阈值`0.02606`。
- Reflection：冻结response-only校准架构，阈值`0.75`。
- Task-Aware：继续冻结V2风险规则与阈值`2.0`。
- 模型参数：`temperature=0.2`、`max_tokens=256`、thinking关闭。

B组在特征模式、学习器和方法选择完成后才使用。V3 A组与V4 B组没有精确指纹重叠，但A组参与了V4训练，只有B组属于最终未触碰确认集。

## B组结果

| 方法 | Accuracy | 95% CI | Avg Cost（CNY） | Avg Latency | Strong Rate |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 78.67% | [72.00, 84.68] | 0.00023762 | 1,178 ms | 0.00% |
| Always Strong | **83.33%** | [77.33, 89.33] | 0.00064448 | 2,619 ms | 100.00% |
| **Task-Aware** | **82.67%** | [76.67, 88.00] | 0.00050139 | **1,767 ms** | 66.00% |
| Learned Text-only | 80.00% | [73.33, 86.00] | **0.00043693** | 2,391 ms | 50.00% |
| Reflection | 80.00% | [73.33, 86.00] | 0.00047537 | 2,202 ms | 46.00% |

Task-Aware只比Always Strong少答对1题，平均成本降低22.2%。Learned Text-only比Strong少5题，V3中“匹配Strong”的结果没有在B组复现。Reflection同为80.00%，但需要先调用Cheap并对46%的题执行Strong review。

## V3/V4应如何比较

V3的79.33%和V4的80.00%来自不同测试题，不能用于证明450题训练优于300题训练。V3的Always Strong只有79.33%，V4则为83.33%，说明两批测试集本身难度不同。

V4支持的结论只有：**在方法和特征冻结后，V3 Text-only匹配Always Strong的结果没有在全新B组复现。**

固定基线在A/B两批合计300题上为：Always Cheap 77.00%、Always Strong 81.33%、Task-Aware 80.67%。Task-Aware比Strong低0.67个百分点，同时成本降低22.5%、延迟降低26.6%、Strong使用率为65.67%。

## 暴露的问题

1. 450题中只有31个Strong-only和21个Cheap-only样本，非零质量差标签仍然稀疏。
2. 单次小确认集无法区分数据规模、测试难度和API随机性。
3. Reflection的自报confidence与review回退问题仍然存在。
4. 旧manifest缺少原始row ID、dataset revision、图片ID和严格近重复审计。

V5因此从固定revision重建3,200/800数据，预先保存fold和来源字段，并加入Random、Oracle、Hard切片与严格冻结清单。

## 文件

- [主结果](../../../results/qwen3.5-v4-study/test_summary.csv)
- [实验元数据](../../../results/qwen3.5-v4-study/study_metadata.json)
- [研究说明](../../research_note.md)
- [导师视角审查](../../supervisor_review.zh-CN.md)

## 版本导航

[V1](../v1/README.zh-CN.md) · [V2](../v2/README.zh-CN.md) · [V3](../v3/README.zh-CN.md) · [V4](README.zh-CN.md) · [V5](../v5/README.zh-CN.md)
