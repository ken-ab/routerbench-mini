# RouterBench-Mini V5：3,200/800冻结评测报告

## 1. 实验目的

V5检验一个明确的问题：V3/V4中Learned Router与Reflection的不稳定，是否主要来自开发数据过少？

实验扩大到3,200道开发题和800道独立测试题。所有模型输出、路由器训练、交叉验证、阈值选择和方法选择均在开发集完成；冻结之后才读取测试集。V5不做300/800/1,600题训练规模曲线，也不使用测试结果重新调参。

- Cheap：`qwen3.5-35b-a3b`
- Strong：`qwen3.5-397b-a17b`
- 解码：`temperature=0.2`、`top_p=0.8`、`max_tokens=256`、thinking关闭
- 任务族：文本、视觉、工具调用
- 主要方法：Always Cheap、Always Strong、冻结Task-Aware、Learned Combined、Reflection
- 控制方法：同升级率Random、同升级率Oracle、Global Oracle

## 2. 数据与隔离

| 集合 | 总数 | Text | Vision | Tool | Standard | Hard |
|---|---:|---:|---:|---:|---:|---:|
| Development | 3,200 | 1,070 | 1,070 | 1,060 | 3,000 | 200 |
| Test | 800 | 270 | 270 | 260 | 600 | 200 |

数据来自GSM8K、CommonsenseQA、BBH、ScienceQA、ChartQA、OCR-VQA、MMMU和BFCL。manifest逐题保存原始来源、split、revision、原始ID、图片信息、难度原因和预分配fold。

隔离检查覆盖：

- 规范化问题和完整任务精确指纹；
- 原始source ID、image ID与image SHA-256；
- BBH整任务文件分组；
- BFCL官方generation group与规范化tool schema连通分量；
- 模板组与字符近重复；
- 历史300题、V3 A组和V4 B组。

所有开发/测试及新旧数据的禁止重叠计数均为0。

| 文件 | SHA-256 |
|---|---|
| `development_manifest.jsonl` | `76deb1ead26dfa8b24032f53949c802985f8c30fdb3d0a481a6f2c8b6cefa44d` |
| `test_manifest.jsonl` | `c27c996cd543d432bd00dcba7b4615013cca88db5693a5b9094911c0e8e63d39` |

## 3. 开发阶段

每道开发题都缓存Cheap solve、Strong solve和Strong review输出。Learned Router的监督标签为：

`y = Strong是否正确 - Cheap是否正确`

| 标签关系 | 题数 |
|---|---:|
| Cheap与Strong都正确 | 2,195 |
| 仅Strong正确（`y=+1`） | 288 |
| 仅Cheap正确（`y=-1`） | 94 |
| 两者都错误 | 623 |

只有382题（11.94%）提供非零质量差信号。Learned的Text-only、Structured-only和Combined版本都使用manifest中预先冻结的五折；每折的TF-IDF、Scaler、Ridge或校准器只在当前训练折拟合。Reflection只使用Cheap回答侧的confidence、格式有效性和self-check信号。

### 3.1 OOF结果与冻结阈值

| 方法 | OOF Accuracy | Strong Rate | 冻结阈值 |
|---|---:|---:|---:|
| Always Cheap | 71.53% | 0.00% | - |
| Always Strong | 77.59% | 100.00% | - |
| Frozen Task-Aware | 76.25% | 64.50% | `2.0` |
| Learned Text-only | 77.25% | 93.47% | `-0.275390` |
| Learned Structured-only | 77.69% | 81.50% | `-0.002837` |
| Learned Combined | 77.31% | 94.56% | `-0.308617` |
| Reflection | 76.34% | 66.50% | `0.75` |

Learned预测分数与真实质量差的OOF相关系数分别为0.0548、0.1312和0.0564。Reflection校准概率与Cheap是否正确的相关系数为0.1487，Brier分数为0.1991。这些信号都存在，但区分能力较弱。

冻结文件记录了manifest哈希、配置、prompt、模型、缓存、四个训练后模型、阈值和源码哈希。冻结时测试API调用数为0。

## 4. 800题主结果

| 方法 | Accuracy | Avg Cost（元） | Avg Latency（ms） | Strong Rate |
|---|---:|---:|---:|---:|
| Always Cheap | 68.25% | 0.00029186 | 722.5 | 0.00% |
| Always Strong | 72.75% | 0.00079930 | 1,318.2 | 100.00% |
| Frozen Task-Aware | 72.12% | 0.00068789 | 1,133.6 | 69.88% |
| Learned Combined | 72.00% | 0.00078259 | 1,290.0 | 95.88% |
| Reflection | 72.00% | 0.00067258 | 1,750.9 | 67.38% |

Always Strong准确率最高。冻结Task-Aware只低0.625个百分点，同时少用30.125%的Strong调用。Learned Combined与Reflection均为72.00%，没有超过Task-Aware。

Learned Combined把95.88%的测试题交给Strong，但仍比Always Strong低0.75个百分点。这是因为它保留的少量Cheap题中包含了可由Strong修复的错误，同时部分升级题发生Strong回退。

## 5. Learned特征消融

| 特征版本 | Accuracy | Strong Rate |
|---|---:|---:|
| Text-only | 72.00% | 95.12% |
| Structured-only | 71.75% | 80.38% |
| Combined | 72.00% | 95.88% |

Text-only与Combined并列，Combined相对Text-only的配对差为0，13维结构特征没有带来可测的增量。Combined相对Structured-only高0.25个百分点，但95% bootstrap区间为-0.625至1.25个百分点，不能确认稳定优势。

## 6. 随机与Oracle控制

| 控制 | Accuracy | Strong Rate |
|---|---:|---:|
| Random@Learned Rate | 72.52% | 95.88% |
| Learned Combined | 72.00% | 95.88% |
| Oracle@Learned Rate | 76.25% | 95.88% |
| Random@Reflection Rate | 70.69% | 67.38% |
| Reflection | 72.00% | 67.38% |
| Oracle@Reflection Rate | 73.62% | 67.38% |
| Global Oracle | 76.25% | 8.00% |

- Learned比同升级率Random低0.52个百分点，95% CI为-1.00至-0.06个百分点；距matched Oracle 4.25个百分点。
- Reflection比同升级率Random高1.31个百分点，随机化检验`p=0.0099`；距matched Oracle 1.625个百分点。
- Global Oracle只升级64/800题（8%）即可达到76.25%，因为测试集中恰有64题是Cheap错而Strong对。

因此，Strong调用数量不是核心瓶颈。真正的问题是能否在调用前或Cheap回答后准确找出64道有收益的题，同时避开28道Cheap对而Strong错的题。

## 7. Reflection机制诊断

Reflection总体Strong使用率为67.38%，但任务族切片显示：

| 任务族 | Accuracy | Strong Rate |
|---|---:|---:|
| Text | 58.89% | 99.63% |
| Vision | 77.04% | 100.00% |
| Tool | 80.38% | 0.00% |

工具调用的Cheap confidence由解析规则固定为0.75，而冻结阈值也恰好是0.75。实现使用`p < threshold`升级，因此工具题全部被接受；文本与视觉回答的校准概率则几乎全部低于0.75。Reflection在V5实际上近似变成了“文本/视觉升级、工具不升级”的任务族路由。

这不否定它相对matched Random的显著增益，但限制了解释：当前结果不能证明校准器已经学会可靠的逐题不确定性。测试review结果为43次有效修复、14次有害回退、211次升级后仍错误和532次保持正确。

## 8. Standard与Hard

| 方法 | Standard | Hard |
|---|---:|---:|
| Always Cheap | 74.50% | 49.50% |
| Always Strong | 79.50% | 52.50% |
| Frozen Task-Aware | 79.00% | 51.50% |
| Learned Combined | 78.50% | 52.50% |
| Reflection | 78.33% | 53.00% |

Hard切片使所有方法下降25个百分点左右，而Always Strong只比Always Cheap高3个百分点。困难题的瓶颈不仅是选错模型，也包括Strong模型自身无法解决的大量题目。

## 9. 配对统计

| 比较 | Accuracy差 | 95% Bootstrap CI | 双侧p值 |
|---|---:|---:|---:|
| Learned - Task-Aware | -0.125 pp | [-1.250, 1.000] | 0.9300 |
| Reflection - Task-Aware | -0.125 pp | [-1.875, 1.625] | 0.9575 |
| Learned - Always Strong | -0.750 pp | [-1.375, -0.250] | 0.0085 |
| Reflection - Always Strong | -0.750 pp | [-2.500, 1.125] | 0.4510 |
| Combined - Text-only | 0.000 pp | [0.000, 0.000] | 1.0000 |
| Combined - Structured-only | 0.250 pp | [-0.625, 1.250] | 0.6955 |

## 10. 对核心研究问题的回答

1. 扩大到3,200题后，Learned与Reflection仍未稳定超过冻结Task-Aware。
2. V5不能与V4做“只由训练规模导致”的因果比较，因为测试集也不同；V5只能说明大开发集没有自动消除不稳定性。
3. Learned没有胜过matched Random，说明其质量差排序仍未可靠定位模型分歧。
4. Reflection胜过matched Random，但升级行为受固定confidence和阈值边界强烈影响。
5. 13维结构特征没有在Combined中提供Text-only之外的增量。
6. 标签稀疏、Strong回退、弱排序信号和Strong在Hard上的能力上限共同构成瓶颈，不能只归因于数据量。

## 11. 成本与可复现性

- 开发与测试共12,000次逻辑API调用、12,842次实际请求尝试。
- 总token数：6,153,179。
- 按响应usage和冻结单价估算总成本：7.7060572元。
- 25项测试全部通过。
- phase1冻结清单中的21个文件哈希全部复核通过。
- 没有重跑、覆盖或修改V1至V4历史结果。
- 本轮按要求没有提交或推送Git。

## 12. 结论与后续方向

V5的负结果是清晰的：增加开发数据并不足以使轻量Learning-to-Route或response-only Reflection稳定超过简单Task-Aware。当前最稳妥的工程选择仍是冻结Task-Aware；Always Strong适合准确率优先且可接受更高成本的场景。

下一步优先级应是：

1. 主动采样Cheap/Strong有分歧的题，而不是继续等比例增加大量`y=0`样本。
2. 把训练目标改为显式效用：正确性收益减去token成本、延迟和Strong回退风险。
3. 为视觉任务加入图片内容表示，而不只依赖问题文本和`has_image`。
4. 将工具题confidence改成可学习的连续信号，并明确测试`<`与`<=`的边界影响。
5. 使用独立Verifier或一致性信号，降低对模型自报confidence的依赖。
6. 在所有方法、阈值和分析脚本再次冻结后，建立第三批完全未触碰的确认集。

## 13. 产物索引

- [旧版本实现审计](v5_large_scale_audit.zh-CN.md)
- [V5配置](../configs/v5_large_scale.yaml)
- [模型配置](../configs/models.qwen_v5.yaml)
- [数据构建脚本](../scripts/build_v5_data.py)
- [开发与冻结脚本](../scripts/run_v5_phase1.py)
- [冻结测试脚本](../scripts/run_v5_phase2.py)
- [冻结清单](../results/qwen3.5-v5-3200-800/frozen/phase1_freeze.json)
- [逐题测试输出](../results/qwen3.5-v5-3200-800/test_predictions.csv)
- [完整切片汇总](../results/qwen3.5-v5-3200-800/test_summary_all_slices.csv)
- [失败案例](../results/qwen3.5-v5-3200-800/failure_analysis.md)
- [随机与Oracle分析](../results/qwen3.5-v5-3200-800/matched_random_oracle_comparisons.csv)
- [复现说明](../results/qwen3.5-v5-3200-800/reproducibility.md)
