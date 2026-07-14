# RouterBench-Mini：面向多模态智能体的成本感知模型复用

[English README](README.md)

RouterBench-Mini 是我从工程实践逐步重新回到科研训练时完成的一项小型实验，目标是在有限规模下完整经历一次多模态模型路由研究的提出、验证、失败与修正过程。

## 摘要

在多模态智能体系统中，将所有请求交给较小模型可以降低调用成本和响应延迟，但可能无法保证复杂任务的准确率；将所有请求交给较强模型虽然更加稳妥，却会持续增加推理成本。RouterBench-Mini围绕这一矛盾展开：面对包含文本推理、视觉理解和工具调用的不同任务，系统能否在回答前或回答后判断当前任务是否真的需要调用更强模型？

实验始终使用Qwen 3.5系列中的两款统一多模态模型，并统一提示词、解码参数、评分规则和API成本统计方式。项目先建立Always Cheap和Always Strong两条基础边界，再依次研究基于任务特征的规则路由、基于回答置信度的Reflection级联，以及基于Cheap和Strong质量差预测的学习式路由。整个实验经历V1至V4四个阶段，每一版都由上一版暴露的问题推动，而不是只保留表现最好的一次结果。

最终确认性实验表明，复杂的学习式路由并未稳定超过简单规则。跨两批共300道无重叠评测题，Task-Aware仅比Always Strong低0.67个百分点，同时平均成本降低22.5%、延迟降低26.6%。相比之下，Learned Router受限于Cheap和Strong真正产生质量差异的样本过少，Reflection则受到自报confidence不可靠以及Strong review可能改坏正确答案等问题影响。

因此，这个项目并不试图提出新的模型路由理论，而是记录一个小型实验如何从基础规则出发，经历标签泄漏、概率校准过拟合、学习式路由和确认性评测，最终得到一个比早期结果更保守、但也更可信的结论。

## 1. 问题背景

面向真实应用的智能体通常不会只处理一种任务。同一个系统可能需要完成数学推理、文本选择、图表理解、图片问答和函数调用，而不同任务对模型能力的要求并不相同。

如果所有请求都交给便宜模型，系统虽然成本较低，但在困难任务上可能出现能力不足；如果所有请求都交给强模型，则大量简单任务也会承担不必要的API费用和延迟。因此，一个实际的多模型系统需要解决以下问题：

> 在不知道标准答案的情况下，如何判断当前请求使用较小模型是否已经足够，以及什么时候调用更强模型能够带来值得付出成本的质量提升？

这一问题可以进一步拆分为三个实验问题：

1. 仅使用推理时可见的任务特征，能否在回答前完成有效路由？
2. Cheap模型生成答案后，能否利用confidence、答案格式和自检结果判断是否需要升级？
3. 能否从历史样本中学习Cheap和Strong的相对质量差，使Router自动决定模型选择？

RouterBench-Mini依次对这三种思路进行了实验。

## 2. 统一实验设置

### 2.1 模型池

实验使用同一Qwen 3.5系列的两款统一多模态模型：

| 角色 | 模型 | 定位 | 输入/输出价格（CNY/百万tokens） |
|---|---|---|---:|
| Cheap | `qwen3.5-35b-a3b` | 较小、调用成本较低 | 0.4 / 3.2 |
| Strong | `qwen3.5-397b-a17b` | 规模更大、能力更强但成本更高 | 1.2 / 7.2 |

两款模型都支持文本、图片和工具调用。因此，本实验研究的不是“文本模型与视觉模型之间的选择”，而是在相同模态能力边界下，根据任务难度选择不同规模的模型。

V1使用`temperature=0`，V2至V4使用`temperature=0.2`。所有版本均关闭thinking，并将最大输出长度设置为256 tokens。

### 2.2 任务构成

实验覆盖三个任务族和五种具体任务形式：

| 任务族 | 任务形式 | 数据来源 | 评分方式 |
|---|---|---|---|
| 文本 | 数学推理 | GSM8K | 最终数值匹配 |
| 文本 | 文本选择题 | CommonsenseQA、BBH逻辑推理 | 选项准确率 |
| 视觉 | 视觉选择题 | ScienceQA、MMMU | 选项准确率 |
| 视觉 | 开放视觉问答 | ChartQA、OCR-VQA | 规范化文本匹配或数值容差 |
| 工具 | 工具调用 | BFCL V4 simple、multiple | 函数名及必需参数匹配 |

标准答案只用于确定性评分，不会提供给Router、Cheap模型或Strong模型。

### 2.3 评测指标

所有方法统一报告以下指标：

- **准确率**：最终回答通过确定性评分的比例；
- **平均成本/题**：根据实际输入和输出token数量计算的API费用；
- **平均延迟**：从发起请求到获得最终结果的观测时间；
- **Strong使用率**：最终调用Strong模型的题目比例。

其中，准确率衡量路由后的任务质量，Strong使用率和平均成本反映模型调用开销。API延迟可能受到远端排队和服务状态影响，因此成本结论通常比延迟结论更加稳定。

## 3. V1：建立基础路由框架

### 3.1 技术路线与数据

V1首先构建一个规模较小但覆盖三类任务的数据集，共包含300道题：

- 文本任务100题：GSM8K 40题、CommonsenseQA 30题、BBH逻辑题30题；
- 视觉任务100题：ScienceQA 80题、ChartQA 10题、OCR-VQA 10题；
- 工具任务100题：BFCL simple 50题、BFCL multiple 50题。

300道题被划分为60道验证题和240道测试题。验证集用于选择Reflection的confidence阈值，测试集用于比较不同策略。

V1设置了四条基础路线：

- **Always Cheap**：所有题目均由Cheap模型回答，用于提供最低成本基线。
- **Always Strong**：所有题目均由Strong模型回答，用于提供模型能力上界。
- **Task-Aware**：根据数据中预先设置的`rule_tier`选择模型，例如GSM8K和逻辑题固定交给Strong，CommonsenseQA固定交给Cheap。
- **Reflection**：先调用Cheap，再根据答案格式、自报confidence和self-check判断是否升级到Strong；验证集最终选择的confidence阈值为0.8。

### 3.2 V1实验结果

| 方法 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 80.00% | 0.00023496 | 707 ms | 0.00% |
| Always Strong | **81.67%** | 0.00063335 | 1,412 ms | 100.00% |
| Task-Aware | 81.25% | 0.00044290 | 1,063 ms | 49.17% |
| Reflection | 78.75% | 0.00057631 | 1,317 ms | 33.33% |

Always Strong取得最高准确率，但相比Always Cheap只提高了1.67个百分点，平均成本却增加约169.6%。Task-Aware只比Strong低0.42个百分点，同时将Strong使用率降低至49.17%，初步表现出了路由的价值。

Reflection的结果没有达到预期。它的平均成本已经接近Always Strong，准确率却只有78.75%，甚至低于Always Cheap。

### 3.3 V1暴露的问题

进一步检查后发现，V1存在两个根本问题。

第一，Task-Aware直接读取数据中预先设置的`rule_tier`。这种规则实际上利用了数据集身份，例如预先知道一道题来自GSM8K或CommonsenseQA，再决定应该调用哪一个模型。真实部署时，Router通常无法获得这种人工标记，因此这一结果存在明显的信息泄漏，不能代表面对新请求时的真实路由能力。

第二，Reflection假设不同任务中的自报confidence具有统一含义，但实验结果并不支持这一点。部分错误数学答案会给出0.95甚至1.0的高confidence，而工具调用只要成功解析，就会被程序固定赋值0.75。在阈值0.8下，工具任务几乎全部升级，一些高置信度错误答案却被直接接受。

因此，V2需要完成两项改进：

1. 删除依赖数据集身份的`rule_tier`，让Router只能使用推理时真正可观察的信息；
2. 不再直接比较原始confidence，而是尝试对Cheap回答正确的概率进行校准。

## 4. V2：可观察特征与概率校准

### 4.1 技术路线与数据调整

V2继续使用300道题和60/240的验证测试划分，但重新平衡了视觉任务：

- ScienceQA 40题；
- ChartQA 20题；
- OCR-VQA 20题；
- MMMU 20题。

这一调整减少了ScienceQA对视觉结果的主导，并加入了难度更高、领域更加多样的MMMU任务。

V2仍然比较四条路线，但重新设计了Task-Aware和Reflection：

- **Always Cheap**：继续作为最低成本基线。
- **Always Strong**：继续作为全量调用强模型的能力基线。
- **Task-Aware**：不再读取数据集标签，而是根据题目长度、数字数量、数学词、逻辑词、是否包含图片、选项数量、候选工具数量、必需参数数量和Schema深度等可观察特征计算风险分数；验证集选择的风险阈值为2.0。
- **Reflection Full**：使用Cheap的原始confidence、答案格式、自检结果以及13个请求侧特征训练逻辑回归，再通过Platt scaling估计`P(Cheap回答正确)`；当概率低于阈值时，Strong会看到原题和Cheap候选答案，并执行review-and-correct。

### 4.2 V2实验结果

| 方法 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 76.67% | 0.00024165 | 1,141 ms | 0.00% |
| Always Strong | 77.92% | 0.00065225 | 1,783 ms | 100.00% |
| **Task-Aware** | **80.00%** | 0.00052408 | 1,610 ms | 68.33% |
| Reflection Full | 76.67% | **0.00025260** | **1,174 ms** | 2.08% |

Task-Aware在这一次240题测试中取得80.00%的准确率，不仅高于Always Cheap，也超过了Always Strong。这说明只使用推理时可见的结构特征，确实可能筛选出一部分更值得交给Strong的任务。

但是，这一优势只来自单次测试划分，尚不足以证明Task-Aware稳定超过Strong。

Reflection Full在验证集上的准确率达到95.00%，但测试集只有76.67%，与Always Cheap完全相同。它在测试阶段只升级了2.08%的请求，说明校准器对大量错误回答给出了过高的正确概率，出现了明显的样本外失效。

### 4.3 Reflection特征消融

为了判断V2 Reflection的效果分别来自哪些特征，实验设计了四种消融版本：Format-only只保留答案是否能够被正确解析的格式信号；Raw confidence只使用Cheap模型自报的原始置信度；Calibrated response-only融合答案格式、原始confidence和self-check，并对Cheap回答正确的概率进行校准；Full则在response-only的基础上进一步加入13个题目结构特征。

| 变体 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Format-only | 76.67% | **0.00024165** | **1,141 ms** | 0.00% |
| Raw confidence | 76.67% | 0.00024524 | 1,149 ms | 0.42% |
| **Calibrated response-only** | **79.17%** | 0.00056005 | 2,060 ms | 59.58% |
| Full：response + 13题目特征 | 76.67% | 0.00025260 | 1,174 ms | 2.08% |

Calibrated response-only取得79.17%的最高准确率，说明对回答格式、confidence和self-check进行联合校准，比直接使用原始confidence更有效。加入13个题目结构特征后，Full版本反而几乎不再升级，说明在当前小样本条件下，额外特征增加了校准器过拟合验证集的风险。

### 4.4 V2缺陷与改进方案

V2的主要问题是验证集规模过小。60道验证题中Cheap只答错5题，却需要同时训练概率校准器和选择升级阈值，导致Reflection在验证集达到95.00%，测试集却下降至76.67%。此外，预测“Cheap是否正确”并不等于预测“调用Strong是否能够带来收益”：当两个模型都会答错，升级只会增加成本；当Cheap正确而Strong错误时，升级还可能降低准确率。

因此，V3将原有300道题全部作为开发集，并将学习目标改为Strong相对于Cheap的质量差。完整流程如下：

```text
300道开发题
  -> Cheap和Strong分别回答
  -> 使用确定性评分得到两个模型是否正确
  -> 构造 y = Strong正确 - Cheap正确
  -> 提取问题TF-IDF和13维可观察结构特征
  -> 使用Ridge预测Strong相对于Cheap的质量差
  -> 通过五折交叉验证分数选择全局路由阈值
  -> 在A组150道新题上回答前选择Cheap或Strong
```

这一改动使Router不再只判断Cheap是否可能出错，而是直接预测调用Strong相对于继续使用Cheap是否具有实际价值。

## 5. V3：学习式质量差路由

### 5.1 方法设计

V3参考了Hybrid LLM中的质量差预测思想和RouteLLM中的成对模型路由框架，将“什么时候使用Strong”从手工规则改写为监督学习问题。FrugalGPT和AutoMix则为回答后升级的Reflection级联提供了方法背景。

原有300道题不再作为测试集，而是全部转为开发集。随后重新构建一套与其数据指纹不重叠的A组测试集，共150题：

- 文本任务50题；
- 视觉任务50题；
- 工具调用50题。

V3加入第五种方法Learned Cost-Aware，并比较以下策略：

- **Always Cheap**：所有请求使用Cheap。
- **Always Strong**：所有请求使用Strong。
- **Task-Aware**：继续使用V2中基于可观察结构特征的风险规则。
- **Learned Cost-Aware**：在回答前预测Strong相对于Cheap的预期质量提升，再根据阈值选择模型。
- **Reflection**：先调用Cheap，再利用回答侧confidence、格式和自检结果预测Cheap是否正确，低于阈值时调用Strong review。

Learned Router的训练流程为：

```text
300道开发题
  -> Cheap和Strong分别回答
  -> 使用确定性评分得到两个模型是否正确
  -> 构造 y = Strong正确 - Cheap正确
  -> 提取问题TF-IDF和13维可观察结构特征
  -> 使用Ridge预测Strong相对于Cheap的质量差
  -> 通过五折交叉验证分数选择全局路由阈值
  -> 在A组150道新题上回答前选择Cheap或Strong
```

在300道开发题中：

- Strong独自正确18题；
- Cheap独自正确14题；
- 两个模型结果相同268题。

这意味着真正能够为模型路由提供监督信号的样本只有32题，其余268题的质量差标签均为0。

V3主实验中的Learned Router使用TF-IDF与13个结构特征组成的Combined表示，阈值为0.04986。Reflection只保留回答侧特征，通过外层五折预测选择0.65作为升级阈值。

### 5.2 V3主实验结果

| 方法 | 准确率 | 95% Bootstrap区间 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 75.33% | [68.00, 82.00] | 0.00024400 | 692 ms | 0.00% |
| Always Strong | **79.33%** | [72.67, 85.33] | 0.00066199 | 1,568 ms | 100.00% |
| Task-Aware | 78.67% | [72.00, 85.33] | 0.00051066 | 1,305 ms | 65.33% |
| Learned Combined | 78.67% | [72.00, 85.33] | **0.00037586** | **1,031 ms** | 36.00% |
| Reflection | 74.00% | [66.67, 81.33] | 0.00063306 | 1,732 ms | 66.00% |

Learned Combined取得78.67%的准确率，只比Always Strong少答对1题，却将Strong使用率从100%降低至36%，共减少96次Strong调用，平均成本下降43.2%。

从单次A组结果看，Learned Combined在准确率和成本之间取得了较好的平衡，也表现出学习式路由相对于固定规则的潜力。

Reflection依然没有达到预期。它在150道题中升级了99题，但Strong review只修复了5个Cheap错误，同时将7个原本正确的Cheap答案修改为错误答案，最终准确率降至74.00%，低于Always Cheap。

这说明“让Strong审查Cheap答案”并不等价于“直接获得Strong模型的能力”。候选答案可能对Strong产生锚定作用，而review提示词本身也可能引入新的回退错误。

### 5.3 Learned Router特征消融

V3进一步对Learned Router的问题表示进行消融。Combined同时融合问题文本的TF-IDF特征和13维结构特征；Structured-only从Combined中去掉TF-IDF，只保留题目长度、数字数量、图片、选项和工具Schema等结构信息；Text-only则去掉全部结构特征，只保留问题文本的TF-IDF表示。

| 特征版本 | 准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Combined：TF-IDF + 13结构特征 | 78.67% | **0.00037586** | **1,031 ms** | **36.00%** |
| Structured-only | 78.67% | 0.00039014 | 1,117 ms | 46.67% |
| **Text-only TF-IDF** | **79.33%** | 0.00044537 | 1,191 ms | 54.67% |

Text-only TF-IDF在A组达到79.33%，与Always Strong准确率相同，同时平均成本降低32.7%。Structured-only与Combined准确率相同，但调用了更多Strong；Combined成本最低，只比Text-only少答对1题。按照“准确率优先、准确率并列时成本优先”的选择规则，Text-only进入V4确认性评测，但这一优势仍需在全新测试集上验证。

### 5.4 从V3到V4

V3看起来得到了一个较好的结果：Text-only Learned Router在准确率上匹配Always Strong，并节省约三分之一成本。

然而，A组已经被用于比较Combined、Structured-only和Text-only，并决定最终采用哪一种特征。因此，A组不再是完全未触碰的最终测试集。如果直接将A组结果作为项目结论，就可能把一次偶然的特征选择优势误认为稳定提升。

因此，V4不再设计新的路由算法，而是进行确认性评测：

1. 冻结V3选出的Text-only方案；
2. 将A组加入开发数据；
3. 重新训练，但不再更改特征模式和方法；
4. 构建一批完全未参与方法选择的新B组测试题；
5. 检验V3中“匹配Strong并降低成本”的结果是否可以复现。

## 6. V4：确认性评测

### 6.1 确认性实验设计

V4将原有300道开发题和A组150题合并，形成450道开发题。随后重新构建与所有已有数据指纹不重叠的B组确认集，共150题。

A组已经参与V3的特征选择，因此不再被视为最终确认数据；B组在模型、特征和阈值冻结后才被使用，是整个实验中唯一完全未触碰的确认集。

V4比较以下方法：

- **Always Cheap**：保持不变；
- **Always Strong**：保持不变；
- **Task-Aware**：保持V2以来的可观察风险规则不变；
- **Learned Text-only**：固定使用V3选出的TF-IDF表示，在450题开发集上重新拟合Ridge和五折阈值，最终阈值为0.02606；
- **Reflection**：固定response-only架构，在450题开发集上重新校准，最终升级阈值为0.75。

B组不参与特征模式、学习器类型或阈值选择。

### 6.2 V4最终结果

![V4准确率与成本权衡](results/qwen3.5-v4-study/pareto.png)

| 方法 | 准确率 | 95% Bootstrap区间 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|---:|
| Always Cheap | 78.67% | [72.00, 84.68] | 0.00023762 | 1,178 ms | 0.00% |
| Always Strong | **83.33%** | [77.33, 89.33] | 0.00064448 | 2,619 ms | 100.00% |
| **Task-Aware** | **82.67%** | [76.67, 88.00] | 0.00050139 | **1,767 ms** | 66.00% |
| Learned Text-only | 80.00% | [73.33, 86.00] | **0.00043693** | 2,391 ms | 50.00% |
| Reflection | 80.00% | [73.33, 86.00] | 0.00047537 | 2,202 ms | 46.00% |

V4中，Always Strong取得83.33%的最高准确率。Task-Aware达到82.67%，只低0.67个百分点，配对准确率差值的95%区间为[-2.67, +1.33]。与此同时，Task-Aware将平均成本降低22.2%，观测延迟降低32.5%。

Learned Text-only达到80.00%，平均成本比Always Strong低32.2%，但准确率低3.33个百分点。V3中“Learned Router匹配Always Strong准确率”的结果没有在新的确认集上复现。

Reflection同样达到80.00%，但它需要先调用Cheap，并对46%的请求继续调用Strong。考虑到级联带来的额外请求过程，它没有表现出相对于回答前路由的稳定优势。

### 6.3 如何解释V3与V4的差异

不能用V3的79.33%和V4的80.00%直接证明增加开发数据提高了Learned Router的准确率，因为两版使用的是不同测试题。

V3的Always Strong准确率为79.33%，V4则为83.33%，说明两批测试题本身的难度和模型表现存在差异。因此，V3与V4之间的绝对准确率不能直接纵向比较。

要研究“300题训练”和“450题训练”之间的差异，必须在同一批全新测试题上同时运行两个训练版本，并保持特征表示、阈值选择和模型API配置完全一致。

V4真正能够支持的结论是：

> 在特征方案和方法固定后，V3中Text-only Learned Router匹配Always Strong的结果没有在B组确认集上复现。

这一结果虽然不如V3亮眼，但比只展示一次最优运行更加可信。

## 7. 两批无重叠评测中的稳定基线

A组和B组各包含150道无重叠题目，但A组参与了V3的特征选择，并随后被加入V4开发集。因此，只有B组属于最终未触碰确认集。

不过，Always Cheap、Always Strong和Task-Aware在V3与V4之间没有发生方法变化，可以将它们在A组和B组上的结果合并，用于观察固定基线的跨批次稳定性。

| 冻结方法 | 两批合计300题准确率 | 平均成本/题（CNY） | 平均延迟 | Strong使用率 |
|---|---:|---:|---:|---:|
| Always Cheap | 77.00% | 0.00024081 | 935 ms | 0.00% |
| Always Strong | **81.33%** | 0.00065324 | 2,094 ms | 100.00% |
| **Task-Aware** | 80.67% | **0.00050602** | **1,536 ms** | 65.67% |

跨两批共300道题，Task-Aware只比Always Strong低0.67个百分点，配对差值95%区间为[-2.33, +1.00]，同时：

- 平均成本降低22.5%；
- 平均延迟降低26.6%；
- Strong调用比例从100%下降至65.67%。

V2中Task-Aware曾在单次测试中超过Always Strong，但两批合并后的结果给出了一个更保守的解释：Task-Aware并未稳定超过Strong，而是在减少约三分之一Strong调用的情况下，保持了接近Strong的准确率。

## 8. 主要发现

### 8.1 简单规则是当前最稳定的方法

最终表现最稳定的并不是学习式Router，而是基于可观察结构特征的Task-Aware规则。

它不需要先调用Cheap，不依赖模型自报confidence，也不需要从少量成对分歧样本中训练质量差预测器。跨两批无重叠评测中，它在准确率接近Always Strong的同时，稳定降低了成本和延迟。

这并不说明手工规则普遍优于学习式路由，而是说明在当前数据规模和模型组合下，简单方法具有更低的估计方差。

### 8.2 Learned Router的关键限制不是模型过于简单，而是有效监督样本不足

在450道开发题中，真正能够区分Cheap和Strong正确性的样本只有52题。其余样本中，两款模型要么都正确，要么都错误。

对于路由器而言，大量`y=0`样本只能说明“两个模型表现相同”，却不能充分告诉模型什么样的任务更值得调用Strong。即使替换更加复杂的分类器或回归器，如果没有增加模型分歧样本，学习式路由仍然很难稳定泛化。

### 8.3 自报confidence不能直接作为跨任务不确定性

不同任务的confidence生成机制并不一致。数学题可能产生高置信度错误答案，工具任务的confidence又可能来自程序规则而不是模型真实不确定性。

V2说明，在极小验证集上对这些信号进行概率校准容易过拟合；V3和V4则说明，即使使用交叉验证，回答侧confidence仍不足以支持稳定的Reflection路由。

### 8.4 Strong review可能产生负向修正

Reflection隐含了一个假设：当Cheap答案不可靠时，让Strong看到Cheap候选并进行修改，最终结果至少不会比Cheap更差。

V3的失败案例否定了这一假设。Strong review修复了部分Cheap错误，也会将部分Cheap正确答案改错。因此，未来研究Reflection时，应同时设置以下两种升级方式：

- **Review Strong**：Strong看到Cheap候选并审查；
- **Blind Strong**：Strong不看到Cheap候选，直接重新回答原题。

只有比较二者，才能区分性能提升来自Strong本身，还是来自review机制。

### 8.5 确认性评测能够推翻偶然的好结果

V3的Text-only方案在A组上匹配Always Strong，看起来是整个项目中最好的结果。但当特征方案被冻结并转移到新的B组后，这一结果没有复现。

这次失败并不是无效实验。相反，它说明了为什么方法选择集和最终确认集需要分离，也说明单次小样本结果不能直接被解释为稳定提升。

## 9. 实验定位

RouterBench-Mini使用的TF-IDF、Ridge回归、逻辑回归、Platt scaling和交叉验证都是经典机器学习方法，不能被包装为新的模型路由理论。

这个项目的主要价值不在于提出了新的Router，而在于完整记录了一次小型研究迭代：

```text
提出准确率与成本问题
  -> 建立Cheap和Strong基础边界
  -> 设计规则路由与Reflection
  -> 发现数据集标签泄漏
  -> 改用推理时可观察特征
  -> 发现小样本概率校准过拟合
  -> 学习Cheap与Strong的质量差
  -> 进行特征消融和方法选择
  -> 构建全新确认集
  -> 推翻一次看起来很好的结果
  -> 得到更加保守但可信的结论
```

因此，它更适合作为一份从工程实践重新进入科研训练的个人实验报告，而不是一项可以独立支撑论文发表的算法贡献。

## 10. 后续实验方向

### 10.1 主动增加有路由价值的样本

当前450道开发题中，只有52题能够区分Cheap和Strong。下一轮不应只随机增加更多简单任务，而应主动寻找：

- Cheap容易失败但Strong能够解决的任务；
- Cheap和Strong采用不同推理路径的任务；
- 视觉内容对难度判断具有决定性作用的任务；
- 工具数量、参数依赖和调用组合更加复杂的任务。

同时应绘制学习曲线，观察路由效果是否会随模型分歧样本增加而稳定提升。

### 10.2 进行受控训练规模实验

为了判断增加开发数据是否有效，应在同一个全新测试集上比较：

- 使用300题训练的Router；
- 使用450题训练的Router；
- 使用更多主动采样分歧题训练的Router。

三种版本需要保持特征、模型、阈值选择和模型API配置完全一致。

### 10.3 改进问题表示

当前TF-IDF只能读取问题文本，无法理解图片本身。对于视觉任务，Router看到的主要是`has_image`等结构信息，而不是图像内容。

后续可以在相同数据划分上比较：

- TF-IDF；
- 固定预训练文本Embedding；
- 图像Embedding；
- 文本与图像联合Embedding；
- 轻量多模态编码器。

重点不是直接使用更大的Router，而是判断视觉内容是否能够提供稳定的模型选择信号。

### 10.4 预先定义成本效用目标

当前阈值选择采用“准确率优先，准确率并列时成本优先”的规则，没有显式定义准确率、成本和延迟之间的交换关系。

后续可以预先固定效用函数：

```text
utility = accuracy - λ × cost - μ × latency
```

或者设置准确率约束：

```text
在准确率最多下降δ的条件下，最小化平均调用成本。
```

这样可以避免在看到测试结果后再决定哪一种成本—准确率权衡更合适。

### 10.5 改进不确定性信号

如果模型API支持，可以进一步比较：

- token-level log probability；
- 输出entropy；
- 多次采样答案一致性；
- 独立Verifier评分；
- Cheap与Verifier之间的语义分歧；
- Cheap内部不同推理路径的一致性。

这些信号比提示词要求模型自报confidence更接近真正的不确定性，但也需要计算额外调用成本。

### 10.6 扩大模型与数据复现

当前实验只使用一个服务商和一个模型系列。后续应加入：

- 更多模型规模档位；
- 不同模型系列；
- 不同供应商；
- 多个随机数据采样批次；
- 更多开放视觉问答和复杂工具调用任务。

在所有方法和超参数冻结后，再建立新的最终确认集，并报告配对置信区间、模型分歧矩阵和典型失败案例。

## 11. 与相关工作的关系

V3的学习式路由主要受到以下工作的启发：

- [Hybrid LLM](https://arxiv.org/abs/2404.14618)：通过预测模型质量差进行路由；
- [RouteLLM](https://arxiv.org/abs/2406.18665)：通过成对偏好和模型相对能力学习路由；
- [FrugalGPT](https://arxiv.org/abs/2305.05176)：研究多个语言模型之间的成本感知级联；
- [AutoMix](https://arxiv.org/abs/2310.12963)：利用不确定性进行模型升级；
- [LLM Routing with Benchmark Datasets](https://arxiv.org/abs/2309.15789)：提供语言模型路由benchmark背景；
- [Deep Model Reassembly](https://arxiv.org/abs/2210.17409)：从性能与资源约束角度支持模型复用的研究动机。

这些工作为实验设计提供了方法背景，但不能直接证明本项目中的手工特征、阈值或学习器具有普遍有效性。

更详细的文献综述和导师视角审查见：

- [`docs/literature_review.md`](docs/literature_review.md)
- [`docs/supervisor_review.zh-CN.md`](docs/supervisor_review.zh-CN.md)

## 12. 复现方法

首先创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[study,test]"

python scripts/build_manifest.py
python scripts/build_v3_data.py
python -m pytest
```

配置以下环境变量：

```bash
export QWEN_API_KEY="YOUR_API_KEY"
export QWEN_BASE_URL="YOUR_API_BASE_URL"
```

运行V3主实验与特征消融：

```bash
python scripts/run_v3_study.py \
  --study-version V3 \
  --workers 8

python scripts/run_v3_ablations.py \
  --workers 8
```

构建V4确认集：

```bash
python scripts/build_v3_data.py \
  --development data/manifest.jsonl data/v3_test.jsonl \
  --out data/v4_test.jsonl \
  --image-dir data/v4_images \
  --seed 20260713 \
  --version v4
```

运行V4确认性实验：

```bash
python scripts/run_v3_study.py \
  --development data/manifest.jsonl data/v3_test.jsonl \
  --test data/v4_test.jsonl \
  --out results/qwen3.5-v4-study \
  --learned-features text \
  --study-version V4 \
  --workers 8
```

聚合跨批次结果：

```bash
python scripts/aggregate_replications.py
```

API Key不会写入仓库。模型响应缓存在：

```text
.cache/routerbench/
```

缓存身份包括任务、模型、提示词版本、solve/review模式、候选答案和解码参数，避免在配置不变时重复产生API调用。

## 13. 局限性

- 当前实验只使用一个服务商和一个模型系列；
- 总计只包含600道采样任务，统计能力有限；
- Cheap和Strong真正产生正确性差异的开发样本较少；
- TF-IDF无法读取图像内容，视觉路由表示较弱；
- 提示词自报confidence不能替代模型内部不确定性；
- Strong review可能受到Cheap候选答案锚定；
- API延迟包含远端排队和服务波动；
- BFCL评分只检查第一个规范函数调用及标准答案要求的参数；
- 公共数据集revision尚未固定，未来重新构建数据时可能需要更新脚本；
- 当前结果只适用于本实验中的模型、任务分布和价格配置，不能直接推广到其他模型池。

## 14. 结果文件

主要实验产物位于：

- V4确认性实验：[`results/qwen3.5-v4-study`](results/qwen3.5-v4-study)
- 跨批次冻结策略：[`results/qwen3.5-confirmatory`](results/qwen3.5-confirmatory)
- V3特征消融：[`results/qwen3.5-v3-ablation`](results/qwen3.5-v3-ablation)

## 最终结论

在当前实验规模下，RouterBench-Mini没有证明学习式路由能够稳定匹配Always Strong，也没有证明回答后的Reflection级联优于回答前路由。

实验中最稳定的结果来自简单的Task-Aware规则：跨两批共300道无重叠题目，它只比Always Strong低0.67个百分点，同时降低22.5%的调用成本和26.6%的观测延迟。

这个结论并不意味着简单规则是模型路由的最终答案。它更准确地说明，在模型分歧样本较少、视觉表示有限且不确定性信号不稳定的条件下，复杂Router带来的估计误差可能超过其理论优势。

RouterBench-Mini最终保留下来的，不是一次最漂亮的实验结果，而是一条可以被复查的实验路径：哪些方法看起来有效、哪些结果没有复现，以及下一步真正需要补充的数据和实验是什么。
