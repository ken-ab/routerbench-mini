# 导师视角审查与本轮修改

## 总体评价

这个项目的优点是问题明确、规模可控，并且同时测量准确率、真实Token成本、延迟和Strong使用率。它适合作为RA申请中的小型研究型项目，因为重点是模型复用与决策，而不是简单调用多个API。

但V2的结论存在几个需要收紧的地方。

## 原版本的主要问题

### 1. 手工参数缺少直接科学依据

`50`词、`3`个数字、`4`个必填参数、Schema深度`4`和各项`+1/+2/+3`均为启发式设置。数学、OCR、图表和工具复杂度作为特征有合理性，但具体常数不能声称来自文献。

此外，原关键词采用子字符串匹配，存在`sum`命中`assume`、`read`命中`already`的风险；`numeric_mentions >= 2`已经触发数学分数，使“图片中数字>=3”的额外规则基本冗余。

### 2. 同一小验证集承担太多角色

V2使用60题同时拟合概率校准器和选择升级阈值，容易产生乐观偏差。全特征Reflection在验证集很好但测试失败，已经是过拟合证据。

### 3. 单次测试上的最好结果被解释得过强

V2中Task-Aware在240题上达到80%，高于Always Strong的77.92%。但没有不确定性区间，也没有第二独立测试，因此不能据此下结论说路由器普遍超过Strong。

### 4. Learned Router名不副实

原四种方法中的“Task-Aware”实际上是规则路由，不是从模型成对表现学习出来的模型选择器。与RouteLLM、Hybrid LLM等工作相比，缺少真正的learned routing baseline。

### 5. Reflection依赖不可靠的自报confidence

API只返回通过提示词要求模型填写的confidence，不是logit、entropy或内部不确定性。这个数字可能按任务格式聚类，也可能随数据分布漂移。V3与V4的reliability table证实了这一风险。

### 6. Review-and-correct结论缺少复现

V2的事后消融显示review减少harmful escalation，但V3和V4没有复现这一优势。review机制可以保留，但不能再包装成已经稳定解决“Strong改坏Cheap正确答案”。

## 本轮实施的修改

1. 新增`LearnedQualityGapEstimator`，预测Strong相对Cheap的正确率差。
2. 使用问题TF-IDF和可观察结构特征，模型为正则化Ridge回归。
3. 使用五折out-of-fold预测选择路由阈值，避免用拟合样本分数调阈值。
4. 将旧300题全部转为开发数据，另建完全不重叠的V3测试集150题。
5. 在V3完成特征消融后，将V3转为第二开发集，再建全新V4确认集150题。
6. 用问题、选项、工具Schema和图片内容哈希检查三组数据零重叠。
7. 增加配对Bootstrap差值区间、分类别/数据集结果、confidence reliability和review反事实表。
8. 修复review统计中“最终正确”和`review_action=correct`共用计数键的问题。
9. 将Task-Aware明确改名为`Handcrafted Task-Aware`，不再暗示权重由数据或论文推导。

## 结果应如何解释

### 稳健主结论

在V3和V4合并的300道独立测试题上：

| 方法 | 准确率 | 平均成本 | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Strong | 81.33% | 0.00065324 | 2,094 ms | 100.00% |
| Handcrafted Task-Aware | 80.67% | 0.00050602 | 1,536 ms | 65.67% |

Task-Aware相对Strong的准确率差为-0.67个百分点，95%配对区间[-2.33,+1.00]；成本低22.5%，延迟低26.6%。因此可以说它在本实验中获得了接近Strong的准确率和更低成本，不能说它显著超过Strong。

### Learned Router结论

V3 text-only消融一度达到与Strong相同的79.33%，成本低32.7%；但冻结后在V4只有80.00%，Strong为83.33%。这说明方法有明显成本收益，但准确率优势没有复现。

### Reflection结论

V4中Reflection为80.00%，Strong为83.33%。69次升级中，review和盲目Strong替换都产生7次有效升级、5次有害升级。当前证据不支持review稳定优于blind replacement。

## 如果继续研究，优先级

1. 收集更大的模型成对响应集，使“Strong独有正确”和“Cheap独有正确”各至少有数百例。
2. 对同一题重复采样，估计模型胜率而不是用单次0/1结果训练路由器。
3. 加入轻量图像编码器或预计算视觉embedding，使learned router真正看到图像内容。
4. 使用显式效用目标：预测准确率收益减去成本和延迟惩罚，而不只是准确率优先、成本破同分。
5. 在不同模型系列上测试迁移，例如训练于Qwen、测试于其他模型对。
6. 对review训练独立的candidate-verdict模型，并将“是否修改”和“修正答案生成”分成两个阶段。

## 面向RA申请的表述

推荐表述：

> I built a compact multimodal routing benchmark and found that a transparent request-time router retained 80.67% accuracy versus 81.33% for always-strong inference while reducing measured API cost by 22.5% across two held-out replications. I also implemented a learned quality-gap router and calibrated review cascade; their mixed transfer results highlighted sparse pairwise supervision and confidence shift as key bottlenecks.

这个版本比“我的路由器超过了大模型”更可信，也更像真实研究：它包含方法、复现、负结果、机制分析和下一步问题。
