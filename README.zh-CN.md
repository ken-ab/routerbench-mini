# RouterBench-Mini：面向多模态智能体的成本感知模型复用

[English](README.md) | **简体中文**

RouterBench-Mini是一项独立、可复现的成本感知模型路由研究。项目不宣称提出了新的路由算法，而是在统一评测协议下，对固定模型基线、规则路由、学习式路由与回答后Reflection进行受控比较。当前首页介绍最新的V5冻结评测；V1至V4的设计与失败分析保留在文末的[版本索引](#版本索引)中。

RouterBench-Mini研究一个模型选择问题：**什么时候使用便宜模型已经足够，什么时候调用强模型能够带来值得付出成本的质量提升？** V5使用同一Qwen 3.5多模态模型系列、统一提示词和解码配置，在文本、视觉与工具调用任务上比较固定模型、规则路由、学习式路由和回答后Reflection级联。

## 核心结果

V5使用3,200道开发题和800道冻结测试题，覆盖文本、视觉与工具调用。当前实验中证据最充分的成本感知方案是冻结的Task-Aware路由：

| 冻结测试结果 | Always Strong | Task-Aware | 差异 |
|---|---:|---:|---:|
| 准确率 | 72.75% | 72.12% | -0.63个百分点 |
| 平均成本 | 0.00079930 CNY | 0.00068789 CNY | **-13.94%** |
| 平均延迟 | 1,318.2 ms | 1,133.6 ms | **-14.01%** |
| Strong使用率 | 100.00% | 69.88% | -30.12个百分点 |

准确率差异为800题中的5题，在下文的配对统计分析中不显著。冻结清单、配置、逐样本预测、分析脚本、协议测试与CI均已公开，便于检查与复现。

## V5实验设计

### 模型与调用参数

| 角色 | 模型 | 输入/输出价格（CNY/百万tokens） |
|---|---|---:|
| Cheap | `qwen3.5-35b-a3b` | 0.4 / 3.2 |
| Strong | `qwen3.5-397b-a17b` | 1.2 / 7.2 |

两款模型都支持文本、图片和工具调用。所有请求统一使用以下配置：

| 参数 | 设置 |
|---|---|
| `temperature` | `0.2` |
| `top_p` | `0.8` |
| `max_tokens` | `256` |
| Thinking | 关闭 |
| System prompt | 无单独system message |
| Timeout / retries | 120秒 / 最多4次 |
| Cheap与Strong提示词 | 完全一致 |

数学题按最终数值评分，选择题按选项字母评分，开放视觉问答采用规范化文本匹配或5%数值容差，工具调用按函数名和必需参数评分。标准答案只用于离线评分，不提供给Router或模型。

### 数据集与划分

V5从固定revision重新构建数据，不沿用旧版300题、V3 A组或V4 B组。3,200题开发集用于模型输出缓存、五折样本外训练、消融和阈值选择；800题测试集只在所有方法冻结后使用。

| 数据集 | 任务 | 开发Standard | 开发Hard | 开发合计 | 测试Standard | 测试Hard | 测试合计 |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 数学推理 | 400 | 25 | 425 | 80 | 25 | 105 |
| CommonsenseQA | 文本选择 | 300 | 15 | 315 | 60 | 15 | 75 |
| BBH | 逻辑/短答案推理 | 300 | 30 | 330 | 60 | 30 | 90 |
| ScienceQA | 视觉选择 | 400 | 15 | 415 | 80 | 15 | 95 |
| MMMU | 多学科视觉推理 | 200 | 25 | 225 | 40 | 25 | 65 |
| ChartQA | 图表问答 | 200 | 15 | 215 | 40 | 15 | 55 |
| OCR-VQA | OCR视觉问答 | 200 | 15 | 215 | 40 | 15 | 55 |
| BFCL Simple | 单工具调用 | 500 | 20 | 520 | 100 | 20 | 120 |
| BFCL Multiple | 多工具调用 | 500 | 40 | 540 | 100 | 40 | 140 |
| **合计** |  | **3,000** | **200** | **3,200** | **600** | **200** | **800** |

按任务族汇总：

| 集合 | Text | Vision | Tool | 总数 |
|---|---:|---:|---:|---:|
| Development | 1,070 | 1,070 | 1,060 | 3,200 |
| Test | 270 | 270 | 260 | 800 |

manifest逐题保存原始source ID、split、dataset revision、图片ID/SHA-256、模板组、BFCL schema组、难度原因和预分配fold。开发集与测试集在精确指纹、原始ID、图片、模板、BFCL schema和近重复检查下的禁止重叠计数均为0。

## 路由方法

### Always Cheap与Always Strong

- **Always Cheap**：所有请求使用Cheap，提供成本下界。
- **Always Strong**：所有请求使用Strong，提供单模型准确率基线。

### Frozen Task-Aware

Task-Aware只读取推理时可观察的请求特征，不读取数据集名称或标准答案。V5完全冻结V2以来的风险规则和阈值2.0：

- 数学线索`+3`，逻辑线索`+2`，题长至少50词`+1`；
- 图片题含chart/OCR线索`+2`，至少3个数字`+1`，至少5个选项`+1`；
- 工具数至少3个`+2`、等于2个`+1`，必需参数至少4个`+1`，schema深度至少4层`+1`。

风险分`>=2.0`使用Strong，否则使用Cheap。

### Learned Router

每道开发题都由Cheap和Strong回答，并通过确定性评分构造标签：

```text
y = Strong是否正确 - Cheap是否正确
```

`y=+1`表示Strong能修复Cheap，`y=-1`表示升级会导致回退，`y=0`表示两者正确性相同。3,200题中只有288个`+1`和94个`-1`，其余2,818题均为0。

V5比较三种问题表示：

- **Text-only**：TF-IDF unigram/bigram，最多1,500维；
- **Structured-only**：13个可观察特征，包括图片、选项、数字、数学/逻辑/OCR/chart线索以及工具schema复杂度；
- **Combined**：TF-IDF与13维结构特征拼接。

三种表示均使用`Ridge(alpha=0.1)`预测Strong质量优势。五折OOF预测用于选择阈值，随后在全部3,200题上重新训练并冻结。主方法Combined的阈值为`-0.308617`。

### Reflection Router

Reflection先调用Cheap，然后从回答中提取三项信号：自报confidence、格式是否合法、self-check是否通过。`LogisticRegression(C=0.5)`与sigmoid概率校准估计`P(Cheap正确)`；概率低于冻结阈值`0.75`或格式/自检失败时，Strong看到Cheap候选并执行一次`review_and_correct`。系统不允许循环升级。

## V5主结果

| 方法 | Accuracy | Avg Cost（CNY） | Avg Latency | Strong Rate |
|---|---:|---:|---:|---:|
| Always Cheap | 68.25% | 0.00029186 | 722.5 ms | 0.00% |
| Always Strong | **72.75%** | 0.00079930 | 1,318.2 ms | 100.00% |
| Frozen Task-Aware | 72.12% | 0.00068789 | 1,133.6 ms | 69.88% |
| Learned Combined | 72.00% | 0.00078259 | 1,290.0 ms | 95.88% |
| Reflection | 72.00% | 0.00067258 | 1,750.9 ms | 67.38% |

Always Strong以582/800题正确取得最高准确率。Frozen Task-Aware答对577题，Learned Combined与Reflection均答对576题。Task-Aware只比两种学习/级联方法多1题，差异不显著，但其规则简单、Strong使用率为69.88%，并且延迟明显低于Reflection。Learned Combined几乎对所有题都调用Strong，却没有形成有效的成本优势。

## Learned特征消融

| 特征版本 | Accuracy | Strong Rate |
|---|---:|---:|
| Text-only | **72.00%** | 95.12% |
| Structured-only | 71.75% | 80.38% |
| Combined | **72.00%** | 95.88% |

Text-only与Combined都答对576题。Combined额外调用6次Strong却没有增加正确题数，说明13维结构特征在当前线性表示上没有提供Text-only之外的可测增益。

## Random与Oracle控制

| 方法 | Accuracy | Strong Rate | 含义 |
|---|---:|---:|---|
| Random@Learned Rate | 72.52% | 95.88% | 保持Learned调用预算，随机选择Strong |
| Learned Combined | 72.00% | 95.88% | 实际学习式路由 |
| Oracle@Learned Rate | 76.25% | 95.88% | 看过标准答案后的同预算上界 |
| Random@Reflection Rate | 70.69% | 67.38% | 保持Reflection调用预算，随机升级 |
| Reflection | 72.00% | 67.38% | 实际回答后升级 |
| Oracle@Reflection Rate | 73.62% | 67.38% | 同预算Oracle上界 |
| Global Oracle | 76.25% | 8.00% | 两个既有答案间的逐题最优选择 |

Learned比同升级率Random低0.52个百分点。Reflection比同升级率Random高1.31个百分点，随机化检验`p=0.0099`。Global Oracle说明，只要准确找出64道“Cheap错、Strong对”的题，理论上使用8%的Strong即可达到76.25%；当前路由器与这个上界仍有明显差距。

## Reflection诊断

Reflection的Strong使用率按任务族为：

| 任务族 | Strong Rate |
|---|---:|
| Text | 99.63% |
| Vision | 100.00% |
| Tool | 0.00% |

工具调用成功解析后会得到固定confidence `0.75`，而升级条件是`p < 0.75`，因此工具题全部被接受；文本和视觉概率则几乎全部落在阈值下方。当前Reflection更接近由confidence生成方式诱导出的任务族路由，而不是稳定的逐题不确定性判断。

## Standard与Hard

| 方法 | Standard | Hard |
|---|---:|---:|
| Always Cheap | 74.50% | 49.50% |
| Always Strong | 79.50% | 52.50% |
| Frozen Task-Aware | 79.00% | 51.50% |
| Learned Combined | 78.50% | 52.50% |
| Reflection | 78.33% | 53.00% |

所有方法在Hard上都明显下降，且Always Strong只比Always Cheap高3个百分点。困难题因此同时受到路由误差和Strong模型自身能力上限的影响。

## 结论与后续方向

V5没有证明Learned Router或Reflection能够稳定优于简单规则：Combined调用95.88%的Strong，准确率仍低于Task-Aware；Reflection虽然胜过同升级率Random，但其任务族升级分布极端、延迟较高，准确率也比Task-Aware少1题。当前最稳妥的成本感知方案仍是冻结Task-Aware，而不是更复杂的学习式Router。

后续实验优先级：

1. 主动增加Cheap与Strong产生`+1/-1`分歧的样本，而不是继续堆积大量`y=0`题目。
2. 将目标改为显式效用：质量收益减去API成本、延迟和Strong回退风险。
3. 为视觉任务加入图片内容表示，而不只依赖题目文本与`has_image`。
4. 将工具confidence改为连续可学习信号，并测试`<`与`<=`的阈值边界。
5. 引入独立Verifier、一致性或多次采样信号，减少对自报confidence的依赖。
6. 在所有方法与阈值再次冻结后，使用第三批完全未触碰的确认集复现结果。

## 复现与文档

- [V5完整实验报告](docs/v5_large_scale_report.zh-CN.md)
- [V5旧版本实现审计](docs/v5_large_scale_audit.zh-CN.md)
- [V5协议配置](configs/v5_large_scale.yaml)
- [模型配置](configs/models.qwen_v5.yaml)
- [数据构建脚本](scripts/build_v5_data.py)
- [开发与冻结脚本](scripts/run_v5_phase1.py)
- [冻结测试脚本](scripts/run_v5_phase2.py)

## 版本索引

根README只展示最新版。各版本的完整实验记录如下：

| 版本 | 核心变化 | 中文 | English |
|---|---|---|---|
| V1 | 基础规则、数据集标签路由、原始confidence | [README](docs/versions/v1/README.zh-CN.md) | [README](docs/versions/v1/README.md) |
| V2 | 可观察特征、概率校准、Reflection消融 | [README](docs/versions/v2/README.zh-CN.md) | [README](docs/versions/v2/README.md) |
| V3 | TF-IDF/Ridge质量差路由、五折OOF与特征消融 | [README](docs/versions/v3/README.zh-CN.md) | [README](docs/versions/v3/README.md) |
| V4 | 450题开发集与全新B组确认性评测 | [README](docs/versions/v4/README.zh-CN.md) | [README](docs/versions/v4/README.md) |
| V5 | 3,200/800冻结协议、Random/Oracle与难度切片 | [README](docs/versions/v5/README.zh-CN.md) | [README](docs/versions/v5/README.md) |
