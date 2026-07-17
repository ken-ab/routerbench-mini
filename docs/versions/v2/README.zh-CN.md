# RouterBench-Mini V2：可观察特征与概率校准

[English](README.md) | **简体中文** | [返回最新版](../../../README.zh-CN.md)

V2修复V1最明显的数据集标签泄漏，并重新设计视觉任务和Reflection。对应主要提交为[`c5dc5cc`](https://github.com/ken-ab/routerbench-mini/tree/c5dc5cc)。

## 数据与参数

V2仍使用300题和60/240划分，但视觉任务改为ScienceQA 40、ChartQA 20、OCR-VQA 20、MMMU 20；文本和工具任务保持各100题。

- Cheap：`qwen3.5-35b-a3b`
- Strong：`qwen3.5-397b-a17b`
- `temperature=0.2`、`max_tokens=256`、thinking关闭
- Task-Aware阈值：`2.0`
- Reflection校准阈值：`0.5`

## Task-Aware

Router不再读取数据集名称，只使用推理时可观察特征：问题长度、数字数量、数学/逻辑/chart/OCR词、图片、选项数、工具数、必需参数数和schema深度。风险分达到2.0时选择Strong。

## Reflection

Reflection Full从Cheap回答提取原始confidence、格式、自检结果，并拼接13个请求侧结构特征。逻辑回归通过三折Platt scaling估计`P(Cheap正确)`；概率低于阈值时，Strong接收原题与Cheap候选并执行review-and-correct。

## 主结果

| 方法 | Accuracy | Avg Cost（CNY） | Avg Latency | Strong Rate |
|---|---:|---:|---:|---:|
| Always Cheap | 76.67% | 0.00024165 | 1,141 ms | 0.00% |
| Always Strong | 77.92% | 0.00065225 | 1,783 ms | 100.00% |
| **Task-Aware** | **80.00%** | 0.00052408 | 1,610 ms | 68.33% |
| Reflection Full | 76.67% | **0.00025260** | **1,174 ms** | 2.08% |

Task-Aware在这一次240题划分中超过Always Strong，但这只是单次小样本结果。Reflection在60题校准集达到95.00%，测试却退回Always Cheap的76.67%，并且只升级5/240题。

## Reflection消融

| 变体 | Accuracy | Avg Cost（CNY） | Avg Latency | Strong Rate |
|---|---:|---:|---:|---:|
| Format-only | 76.67% | **0.00024165** | **1,141 ms** | 0.00% |
| Raw confidence | 76.67% | 0.00024524 | 1,149 ms | 0.42% |
| **Calibrated response-only** | **79.17%** | 0.00056005 | 2,060 ms | 59.58% |
| Full：response + 13结构特征 | 76.67% | 0.00025260 | 1,174 ms | 2.08% |

Response-only明显优于Full，说明13个额外特征在只有60题的校准数据上增加了过拟合，而不是稳定改善不确定性估计。

## 暴露的问题

1. 同一60题既训练校准器又选择阈值，且Cheap只错5题，导致严重乐观偏差。
2. “Cheap可能错误”不等于“Strong能够修复Cheap”。两模型都错时升级只增加成本；Cheap对而Strong错时升级会降低准确率。
3. Task-Aware的手工分值有工程直觉，但缺少独立验证。

V3因此把旧300题全部转为开发数据，直接学习`Strong正确-Cheap正确`的质量差，并建立全新的150题A组测试集。

## 文件

- [主结果](../../../results/qwen3.5-v2-study/test_summary.csv)
- [Reflection消融](../../../results/qwen3.5-v2-ablation/test_summary.csv)
- [实验元数据](../../../results/qwen3.5-v2-study/study_metadata.json)

## 版本导航

[V1](../v1/README.zh-CN.md) · [V2](README.zh-CN.md) · [V3](../v3/README.zh-CN.md) · [V4](../v4/README.zh-CN.md) · [V5](../v5/README.zh-CN.md)
