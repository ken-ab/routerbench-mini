# RouterBench-Mini 消融实验说明

## 什么是消融实验

消融实验是在数据、模型、提示词、温度和评分方式保持不变的情况下，从完整方法中一次移除一个组件，再观察性能变化。它回答的不是“完整方法好不好”，而是“究竟是哪一个组件带来了收益或损失”。

例如，完整 Reflection Router 同时使用格式检查、自检、原始 confidence、概率校准和任务特征。如果完整方法优于 baseline，但没有消融实验，就无法判断改进来自概率校准、任务特征，还是仅仅来自更多模型调用。

## 本项目的 Reflection 消融

`scripts/run_ablations.py` 定义四个逐步增加信息的版本：

| 版本 | 使用的信号 | 要回答的问题 |
|---|---|---|
| `reflection_format_only` | 仅答案格式和工具 Schema | 纯确定性检查能解决多少错误？ |
| `reflection_raw_confidence` | 格式 + self-check + 模型自报 confidence | 未校准的自信程度是否有路由价值？ |
| `reflection_calibrated_response_only` | 格式 + self-check + 校准后的响应信号 | 概率校准本身是否有效？ |
| `reflection_full` | 上述全部 + 可观察任务特征 | 题目和工具复杂度是否提供额外收益？ |

比较时应重点报告准确率、平均成本、延迟、升级率、有效升级 precision 和有效升级 recall。

有效升级定义为：Cheap 回答错误且 Strong 回答正确。一次升级如果只是把 Cheap 的正确答案替换成 Strong 的错误答案，应记为 harmful escalation，而不是成功。

## Task-Aware 消融

Task-Aware Router 的风险分数由四组特征组成：

- 基本复杂度：题目长度、数字数量、选项数量。
- 推理线索：数学和逻辑关键词。
- 视觉线索：是否有图片、图表和 OCR 关键词。
- 工具线索：候选工具数、必需参数数和 Schema 深度。

后续可分别去掉视觉线索、工具线索或数学/逻辑线索，再重新选择验证集阈值。若去掉某组特征后测试集表现明显下降，才能说明这组特征对路由确实有贡献。

## 实验纪律

- 每次消融只改变一个组件。
- 所有阈值只在验证集选择。
- 测试集只在方法冻结后运行一次。
- 不根据测试结果回头修改规则。
- V1 和 V2 使用不同数据与温度，结果不能直接混为同一张主表。

## V2 实际结果

| 版本 | 准确率 | 平均成本 | 平均延迟 | 升级率 |
|---|---:|---:|---:|---:|
| Format only | 76.67% | 0.00024165 | 1,141 ms | 0.00% |
| Raw confidence | 76.67% | 0.00024524 | 1,149 ms | 0.42% |
| Calibrated response only | **79.17%** | 0.00056005 | 2,060 ms | 59.58% |
| Full response + task features | 76.67% | 0.00025260 | 1,174 ms | 2.08% |

完整特征版本在验证集达到 95%，但测试集没有提升，说明 60 条验证数据不足以稳定拟合较高维特征。response-only 版本泛化更好，但串行 review 增加了延迟。

在 response-only 的 143 次升级中，review-and-correct 将 harmful escalation 从盲目 Strong 覆盖的 8 次降到 5 次，同时 beneficial escalation 从 14 次降到 11 次。它更保守，减少了破坏，也错过了部分修正。

## V3 学习式路由消融

V3新增学习式quality-gap路由，并在同一开发集、同一五折out-of-fold协议下逐项移除特征：

| 版本 | 准确率 | 平均成本 | Strong使用率 |
|---|---:|---:|---:|
| Structured only | 78.67% | 0.00039014 | 46.67% |
| Text only | **79.33%** | 0.00044537 | 54.67% |
| Text + structured | 78.67% | **0.00037586** | 36.00% |

Text-only在V3匹配Always Strong，因此被预先选为V4确认方案。V4中text-only为80.00%，Always Strong为83.33%。这说明消融结果帮助提出了候选方法，但其准确率优势没有在新测试集复现。

## V3/V4 Reflection复现

| 测试 | Review有效/有害 | Blind Strong有效/有害 | 结论 |
|---|---:|---:|---|
| V3 | 5 / 7 | 11 / 6 | review更差 |
| V4 | 7 / 5 | 7 / 5 | 两者相同 |

因此V2中“review减少有害升级”的结果不能作为稳定结论。后续应使用独立candidate-verdict模型，并将是否接受候选与生成修正答案分开评估。
