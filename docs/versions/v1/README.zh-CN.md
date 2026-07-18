# RouterBench-Mini V1：基础路由框架

[English](README.md) | **简体中文** | [返回最新版](../../../README.zh-CN.md)

V1是项目的基础版本，目标是先建立Cheap/Strong模型池、三类任务、统一评分和四条基本路由路线。对应的主要实现提交为[`4cf8f44`](https://github.com/ken-ab/routerbench-mini/tree/4cf8f44)。

## 研究问题

如果所有请求都使用Cheap，困难题可能能力不足；如果全部使用Strong，简单题也要承担高成本。V1尝试回答：简单规则和Cheap回答后的自检，能否在准确率接近Strong的同时减少Strong调用？

## 数据与参数

| 任务族 | 数据集 | 数量 |
|---|---|---:|
| Text | GSM8K 40、CommonsenseQA 30、BBH 30 | 100 |
| Vision | ScienceQA 80、ChartQA 10、OCR-VQA 10 | 100 |
| Tool | BFCL Simple 50、BFCL Multiple 50 | 100 |

300题按任务族分层划分为60题验证集和240题测试集。模型为`qwen3.5-35b-a3b`与`qwen3.5-397b-a17b`；`temperature=0`、`max_tokens=256`、thinking关闭，`top_p`使用服务端默认值。

## 方法

- **Always Cheap**：全部使用Cheap。
- **Always Strong**：全部使用Strong。
- **Task-Aware**：直接读取manifest中的`rule_tier`；GSM8K和逻辑题等预先标记任务固定交给Strong。
- **Reflection**：Cheap先回答，再检查答案格式、自报confidence和self-check；验证集选择统一阈值`0.8`，低于阈值时升级Strong。

工具题只要函数调用可解析，就被程序赋予固定confidence `0.75`；不可解析时为较低固定值。数学和选择题的confidence主要来自模型在JSON中的自报值。

## 测试结果

| 方法 | Accuracy | Avg Cost（CNY） | Avg Latency | Strong Rate |
|---|---:|---:|---:|---:|
| Always Cheap | 80.00% | 0.00023496 | 707 ms | 0.00% |
| Always Strong | **81.67%** | 0.00063335 | 1,412 ms | 100.00% |
| Task-Aware | 81.25% | 0.00044290 | 1,063 ms | 49.17% |
| Reflection | 78.75% | 0.00057631 | 1,317 ms | 33.33% |

Task-Aware只比Always Strong少答对1题，同时Strong使用率降到49.17%。Reflection则低于Always Cheap，且成本已经接近Always Strong。

## 暴露的问题

1. Task-Aware使用`rule_tier`，实际上提前知道数据集或人工难度标签，存在信息泄漏，无法代表真实请求上的路由。
2. 原始confidence跨任务不可比。错误数学答案可能自报0.95或1.0，而工具题固定为0.75。
3. 阈值0.8使几乎所有工具题升级，却接受一部分高置信度错误数学答案。
4. Reflection没有验证“Strong review是否会把正确Cheap答案改错”。

V2因此删除数据集标签路由，改用推理时可观察特征，并尝试概率校准与review-and-correct。

## 文件

- [测试汇总](../../../results/qwen3.5-study/test_summary.csv)
- [验证集汇总](../../../results/qwen3.5-study/validation_summary.csv)
- [实验元数据](../../../results/qwen3.5-study/study_metadata.json)

## 版本导航

[V1](README.zh-CN.md) · [V2](../v2/README.zh-CN.md) · [V3](../v3/README.zh-CN.md) · [V4](../v4/README.zh-CN.md) · [V5](../v5/README.zh-CN.md)
