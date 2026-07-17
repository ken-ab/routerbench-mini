# RouterBench-Mini V5 大规模实验报告

## 协议

V5 使用 3,200 道开发题完成所有训练、五折样本外预测、阈值选择与冻结；随后只读取一次 800 道独立测试题。测试集没有参与词表拟合、特征选择、阈值选择或提示词修改。

- 开发集 SHA-256：`76deb1ead26dfa8b24032f53949c802985f8c30fdb3d0a481a6f2c8b6cefa44d`
- 测试集 SHA-256：`c27c996cd543d432bd00dcba7b4615013cca88db5693a5b9094911c0e8e63d39`
- Cheap：`qwen3.5-35b-a3b`
- Strong：`qwen3.5-397b-a17b`

开发集包含文本1,070题、视觉1,070题和工具1,060题；测试集包含文本270题、视觉270题和工具260题。所有数据集精确指纹、原始ID、图片ID/SHA、模板组、BFCL schema组和近重复检查的跨集合重叠数均为0。

## 开发阶段与冻结

3,200题五折样本外（OOF）结果如下。阈值只在这些OOF预测上选择，随后用全部开发题重训并冻结；测试题在冻结前调用数为0。

| 方法 | OOF Accuracy | Strong Rate | 冻结阈值 |
|---|---:|---:|---:|
| Always Cheap | 71.53% | 0.00% | - |
| Always Strong | 77.59% | 100.00% | - |
| Frozen Task-Aware | 76.25% | 64.50% | `2.0` |
| Learned Text-only | 77.25% | 93.47% | `-0.275390` |
| Learned Structured-only | 77.69% | 81.50% | `-0.002837` |
| Learned Combined | 77.31% | 94.56% | `-0.308617` |
| Reflection | 76.34% | 66.50% | `0.75` |

成对标签中有2,195题两模型都对、288题仅Strong正确、94题仅Cheap正确、623题都错。真正非零的质量差标签只有382题（11.94%），这限制了Learned Router可学习的监督信号。

## 主结果

| 方法 | Accuracy | Avg Cost | Avg Latency (ms) | Strong Rate |
|---|---:|---:|---:|---:|
| Always Cheap | 68.25% | 0.00029186 | 722.5 | 0.00% |
| Always Strong | 72.75% | 0.00079930 | 1318.2 | 100.00% |
| Frozen Task-Aware | 72.12% | 0.00068789 | 1133.6 | 69.88% |
| Learned Combined | 72.00% | 0.00078259 | 1290.0 | 95.88% |
| Reflection | 72.00% | 0.00067258 | 1750.9 | 67.38% |
| Random@Learned Rate | 72.52% | 0.00077843 | 1293.8 | 95.88% |
| Oracle@Learned Rate | 76.25% | 0.00078565 | 1296.3 | 95.88% |
| Random@Reflection Rate | 70.69% | 0.00089224 | 1832.3 | 67.38% |
| Oracle@Reflection Rate | 73.62% | 0.00097353 | 1841.2 | 67.38% |
| Global Oracle | 76.25% | 0.00032054 | 767.7 | 8.00% |

## 标签与路由空间

- Cheap 对、Strong 对：518
- Cheap 错、Strong 对（真正有升级价值）：64
- Cheap 对、Strong 错（升级会回退）：28
- Cheap 错、Strong 错：190

## 对 11 个问题的回答

1. Learned Combined 在 V5 为 72.00%；V4 为 80.00%。这是不同测试集上的描述性比较，不能单独归因于开发集扩大。
2. Reflection 在 V5 为 72.00%；V4 为 80.00%，同样不能把差异只归因于数据规模。
3. Learned Combined 相比 Task-Aware 的准确率差为 -0.12%。
4. Reflection 相比 Task-Aware 的准确率差为 -0.12%。
5. Learned 相比 matched random 的平均增益为 -0.52%，随机化 p=1.0000。
6. Reflection 相比 matched random 的平均增益为 +1.31%，随机化 p=0.0099。
7. Learned 距 matched oracle 仍有 4.25% 准确率差距。
8. 三种 Learned 特征中，`text_only`和`combined`同为72.00%，并列最高；`structured_only`为71.75%。
9. Combined 相比 Text-only 的差为 +0.00%；这直接衡量 13 维结构特征在文本特征之上的互补价值。
10. Learned 的 hard/standard 准确率为 52.50%/78.50%；Reflection 为 53.00%/78.33%。
11. 主要限制由结果共同判断：有效 +1/-1 标签数量、Strong 本身回退、Learned 与 matched oracle 的差距，以及 Reflection 概率与真实错误的区分能力。不能仅用数据量解释。

## 消融与机制检查

| Learned特征 | Accuracy | Strong Rate |
|---|---:|---:|
| Text-only | 72.00% | 95.12% |
| Structured-only | 71.75% | 80.38% |
| Combined | 72.00% | 95.88% |

Combined没有优于Text-only，说明这13维手工结构特征在当前数据和线性模型下没有提供可验证的增量价值。Learned Combined在测试集把95.88%的题交给Strong，却仍比Always Strong低0.75个百分点（配对bootstrap 95% CI：-1.38至-0.25个百分点，双侧p=0.0085）；其路由准确率还比同升级率随机路由低0.52个百分点。

Reflection的总体Strong使用率为67.38%，但并不是均匀的不确定性升级：文本、视觉、工具的升级率分别为99.63%、100%和0%。原因是工具调用的固定confidence恰好为0.75，而判定条件是`p < 0.75`；工具题因此被接受，文本和视觉题则几乎全部升级。Reflection比同升级率随机路由高1.31个百分点（随机化p=0.0099），但当前结果更接近一种由置信度格式诱导出的任务族路由，不能直接解释为校准器学会了可靠的逐题不确定性。

在800道测试题上，真正“Cheap错而Strong对”的只有64题，“Cheap对而Strong错”的有28题。Global Oracle只需升级8%的题即可达到76.25%，说明可利用的路由空间存在，但当前Learned与Reflection尚未准确定位这些题。

## 难度切片

| 方法 | Standard | Hard |
|---|---:|---:|
| Always Cheap | 74.50% | 49.50% |
| Always Strong | 79.50% | 52.50% |
| Frozen Task-Aware | 79.00% | 51.50% |
| Learned Combined | 78.50% | 52.50% |
| Reflection | 78.33% | 53.00% |

所有方法在Hard切片上都显著下降，且Strong仅比Cheap高3个百分点。困难题上的主要瓶颈因此不只是路由选择，还包括Strong模型本身的能力上限。

## 结论边界

即使将开发集扩展至 3,200 题，学习式路由与反思式升级仍未获得对冻结 Task-Aware 的稳定优势，说明旧版本的问题不能主要归因于开发数据不足。

## API 统计

- 开发与测试共执行12,000次逻辑API调用、12,842次实际请求尝试，合计6,153,179 tokens。
- 全实验按响应usage和冻结单价估算成本为7.7060572元。
- 测试 Cheap solve：800 次，411,695 tokens，0.2334852元。
- 测试 Strong solve：800 次，411,412 tokens，0.6394404元。
- 测试 Strong review：800 次，454,731 tokens，0.7119672元。

完整的标准/困难、任务族、数据集、消融、随机、Oracle、失败案例和逐题输出均保存在同一结果目录。
