# RouterBench-Mini V5：3,200/800冻结评测

[English](README.md) | **简体中文** | [返回最新版](../../../README.zh-CN.md)

V5检验V3/V4的不稳定是否主要来自开发数据过少。它不是简单扩大旧manifest，而是从固定revision重建3,200题开发集和800题独立测试集，并在测试前冻结数据哈希、源码、提示词、模型配置、学习器与阈值。

## 数据与设置

- Development：3,200题，Text 1,070、Vision 1,070、Tool 1,060；Standard 3,000、Hard 200。
- Test：800题，Text 270、Vision 270、Tool 260；Standard 600、Hard 200。
- 数据集：GSM8K、CommonsenseQA、BBH、ScienceQA、MMMU、ChartQA、OCR-VQA、BFCL Simple/Multiple。
- 模型：`qwen3.5-35b-a3b`与`qwen3.5-397b-a17b`。
- 解码：`temperature=0.2`、`top_p=0.8`、`max_tokens=256`、thinking关闭。
- 五折fold在manifest中预先固定，测试集不参与词表、特征、阈值或方法选择。

## 方法

- **Task-Aware**：冻结风险阈值2.0，只使用问题、图片存在、选项和工具schema特征。
- **Learned**：`y=Strong正确-Cheap正确`；比较Text-only、Structured-only和Combined；使用TF-IDF、13维结构特征与`Ridge(alpha=0.1)`。
- **Reflection**：confidence、格式和self-check输入逻辑回归与sigmoid校准；阈值0.75；最多一次Strong review。
- **Controls**：同Strong使用率Random、同预算Oracle与Global Oracle。

3,200题中有288个`+1`、94个`-1`和2,818个`0`。Combined阈值为`-0.308617`，Reflection阈值为`0.75`。

## 主结果

| 方法 | Accuracy | Avg Cost（CNY） | Avg Latency | Strong Rate |
|---|---:|---:|---:|---:|
| Always Cheap | 68.25% | 0.00029186 | 722.5 ms | 0.00% |
| Always Strong | **72.75%** | 0.00079930 | 1,318.2 ms | 100.00% |
| Frozen Task-Aware | 72.12% | 0.00068789 | 1,133.6 ms | 69.88% |
| Learned Combined | 72.00% | 0.00078259 | 1,290.0 ms | 95.88% |
| Reflection | 72.00% | 0.00067258 | 1,750.9 ms | 67.38% |

Always Strong取得最高准确率。Task-Aware答对577/800题，Learned Combined与Reflection均答对576/800题；Task-Aware比两者多1题，同时保持更简单、可解释的回答前路由。

## 关键诊断

- Text-only与Combined都为72.00%，结构特征没有提供额外准确率。
- Learned Strong使用率95.88%，仍低于Random@Learned的72.52%。
- Reflection比Random@Reflection的70.69%高1.31个百分点。
- Global Oracle以8%的Strong使用率达到76.25%，说明主要瓶颈是识别64道真正有升级收益的题。
- Reflection对Text、Vision、Tool的Strong使用率为99.63%、100%和0%，暴露固定工具confidence与阈值边界问题。
- Reflection的Standard/Hard为78.33%/53.00%。

## 结论

扩大开发集没有使Learned Router或Reflection稳定超过Task-Aware。Reflection虽然胜过同升级率Random，但升级分布极端、延迟较高；当前最稳妥的成本感知方法仍是冻结Task-Aware。

## 文件

- [最新版中文README](../../../README.zh-CN.md)
- [完整V5报告](../../v5_large_scale_report.zh-CN.md)
- [旧版本实现审计](../../v5_large_scale_audit.zh-CN.md)
- [V5配置](../../../configs/v5_large_scale.yaml)

## 版本导航

[V1](../v1/README.zh-CN.md) · [V2](../v2/README.zh-CN.md) · [V3](../v3/README.zh-CN.md) · [V4](../v4/README.zh-CN.md) · [V5](README.zh-CN.md)
