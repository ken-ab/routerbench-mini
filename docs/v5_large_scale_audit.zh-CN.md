# RouterBench-Mini V5 大规模实验：旧版本实现审计

审计日期：2026-07-17
审计基线：`main` / `a9db8efc141682a3bd7db60080b39850bdeccbb8`

## 1. 仓库状态

- 本地分支：`main`
- 远端：`origin = https://github.com/ken-ab/routerbench-mini.git`
- 本地与`origin/main`差异：`0 ahead / 0 behind`
- 审计开始时工作区：干净，无未提交文件
- 历史数据：`data/manifest.jsonl`（300题）、`data/v3_test.jsonl`（A组150题）、`data/v4_test.jsonl`（B组150题）
- 历史结果：`results/qwen3.5-study`、`qwen3.5-v2-study`、`qwen3.5-v3-study`、`qwen3.5-v4-study`等目录

本次V5实验必须使用新的目录和版本化配置，不覆盖上述数据或结果。

## 2. 原300题的真实抽样

当前`data/manifest.jsonl`对应V2后的300题，而V1的原始抽样代码保留在commit `4cf8f44`。

| 数据集 | V1数量 | 当前V2数量 | 实际来源split | 抽样方式 |
|---|---:|---:|---|---|
| GSM8K | 40 | 40 | `openai/gsm8k`, `test` | Hugging Face shuffle(seed=42)后取前N题 |
| CommonsenseQA | 30 | 30 | `tau/commonsense_qa`, `validation` | shuffle(seed=42)后取前N题 |
| BBH | 30 | 30 | BIG-Bench-Hard `logical_deduction_three_objects.json` | Python `random.Random(42).shuffle`后取前N题 |
| ScienceQA | 80 | 40 | `derek-thomas/ScienceQA`, `test` | shuffle(seed=42)，过滤无图片或无选项题后取前N题 |
| ChartQA | 10 | 20 | `docintel/ChartQA`, `test` | shuffle(seed=42)后取前N题 |
| OCR-VQA | 10 | 20 | `pppop7/OCR-VQA`, `train` streaming | 不shuffle，顺序读取首个可用问题；当前版过滤provider阻断短语 |
| MMMU | 0 | 20 | `MMMU/MMMU`, `validation` | 按subject以`seed:subject` shuffle，只保留单图选择题 |
| BFCL Simple | 50 | 50 | Gorilla仓库`BFCL_v4_simple_python.json` | shuffle(seed=42)后取前N个可转换样本 |
| BFCL Multiple | 50 | 50 | Gorilla仓库`BFCL_v4_multiple.json` | shuffle(seed=42)后取前N个可转换样本 |

300题随后只按一级`category`（text/vision/tool）分层，以seed 42划分20%验证集和80%测试集，得到60/240；没有按`source_dataset`联合分层。

## 3. A组和B组的真实抽样

两组均由`scripts/build_v3_data.py`生成，并保持相同的150题分布：

- 文本50：GSM8K 20、CommonsenseQA 15、BBH 15
- 视觉50：ScienceQA 20、ChartQA 10、OCR-VQA 10、MMMU 10
- 工具50：BFCL Simple 25、BFCL Multiple 25

### A组

- 文件：`data/v3_test.jsonl`
- seed：`20260712`
- 排除集：当前300题manifest
- 生成命令的默认version：`v3`

### B组

- 文件：`data/v4_test.jsonl`
- seed：`20260713`
- 排除集：当前300题manifest + A组150题
- README记录的构建命令明确传入`--version v4`

两组通过以下精确指纹去重：规范化后的question、choices、完整tools和图片SHA-256。当前复算未发现300题与A/B组、A组与B组之间的精确指纹重叠。

## 4. 旧数据无法证明的内容

旧manifest没有保存以下信息：

- Hugging Face原始sample ID（除题目内容外无法直接回溯）；
- 原始row index；
- dataset revision / commit；
- BFCL原始row ID；
- BBH原始文件内index；
- source split字段（只能由构建代码反推）；
- image ID；
- license/source note；
- prompt hash；
- 交叉验证fold ID。

因此，旧数据只能复现“使用当时loader和seed重新抽取”的流程，不能证明重新下载后仍会命中完全相同的官方原始记录。A/B组也只做了精确指纹检查，没有字符级、语义级或BFCL prompt/schema近重复检查。V5不能继承这一缺口，必须重建带原始来源字段的新manifest。

## 5. Learned Router真实实现

旧V3/V4使用`LearnedQualityGapEstimator`：

- 标签：`y = is_correct(Strong) - is_correct(Cheap)`，取值`+1/0/-1`；
- 文本：`TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=1500, sublinear_tf=True, strip_accents="unicode")`；
- 结构特征：13维特征经`StandardScaler`拟合；
- 学习器：`Ridge(alpha=0.1)`；
- V3主版本：TF-IDF + 13维结构特征；
- V4确认版本：只使用TF-IDF；
- 阈值：对五折OOF质量差分数的相邻中点逐一模拟，先选开发准确率最高者，再以低成本和低Strong使用率打破并列；
- 最终模型：阈值选定后在全部开发题上重新fit。

旧五折只按一级category分层，`random_state=42`。它没有把`source_dataset`和difficulty一起纳入分层；V5必须改为使用manifest中预先保存的fold ID，并确保TF-IDF和Scaler只在当前训练折fit。

## 6. 13维结构特征

按代码排序后的准确名称为：

1. `chart_cue`
2. `choice_count`
3. `has_image`
4. `is_math`
5. `is_multiple_choice`
6. `is_tool`
7. `logic_cue`
8. `numeric_mentions`
9. `ocr_cue`
10. `question_words`
11. `required_arg_count`
12. `schema_depth`
13. `tool_count`

这些特征只读取推理时可见的question、choices、image presence和tool schema，不读取dataset名称或答案。`is_math`由数学词、显式算式或至少两个数字触发；cue特征由固定词表触发；schema depth递归计算字典/数组最大深度。

## 7. Reflection Router真实实现

Reflection不是纯规则系统，而是包含可训练校准器：

1. Cheap先生成答案；
2. 从Cheap响应提取三项回答侧信号：自报confidence、格式是否合法、self-check是否通过；
3. `StandardScaler + LogisticRegression(C=0.5, random_state=42)`作为基础分类器；
4. `CalibratedClassifierCV(method="sigmoid")`执行Platt式概率校准；
5. 外层五折产生样本外`P(Cheap正确)`，用开发集选择升级阈值；
6. 低于阈值、格式失败或self-check失败时，只升级一次；
7. Strong看到Cheap候选并执行`review_and_correct`，没有无限循环。

V3/V4最终使用response-only三项信号，不包含13维题目特征。阈值候选为0.05至1.00、步长0.05，选择规则同样是准确率优先、成本和Strong使用率打破并列。

## 8. Task-Aware冻结规则

V2以来Task-Aware固定阈值为2.0。风险分规则为：

- 数学线索：+3；逻辑线索：+2；题长至少50词：+1；
- 图片题出现chart/OCR线索：+2；至少3个数字：+1；至少5个选项：+1；
- 工具数至少3：+2，等于2：+1；必需参数至少4：+1；schema depth至少4：+1。

风险分`>=2.0`选择Strong，否则选择Cheap。V5将原样冻结此规则与阈值，不使用新开发集或测试集修改。

## 9. 提示词、模型和API配置

- Cheap：`qwen3.5-35b-a3b`
- Strong：`qwen3.5-397b-a17b`
- temperature：V2至V4为0.2
- max_tokens：256
- thinking：关闭
- top_p：代码未显式发送，使用服务端默认值
- system prompt：没有单独system message；固定指令作为user message发送
- timeout：默认120秒
- retry：默认4次，429和5xx按指数退避重试
- prompt version：`unified-multimodal-v2-review-correct`

缓存键包含完整任务、模型、prompt version、solve/review模式、候选答案和解码参数。

## 10. 评分、成本与延迟

- 数学：提取最终数字后精确比较；
- 选择题：提取选项字母后比较；
- 开放VQA：规范化文本精确匹配，或数值在gold的5%相对容差内；
- 工具：比较首个规范函数名，并检查gold要求的参数值；允许预测包含额外参数；
- Accuracy：通过上述确定性评分的比例；
- API cost：优先使用响应usage按模型输入/输出单价计算的真实估算值；
- latency：直接路由取单次观测延迟，Reflection取Cheap与Strong review延迟之和；
- token：汇总各实际调用的prompt/completion usage。

## 11. README一致性

README对V1至V4的主要模型、数量、阈值、准确率和成本与结果文件一致，也明确承认dataset revision未固定。需要补充的边界是：旧A/B组只能证明精确指纹不重叠，不能证明原始ID和近重复层面的完全独立。

## 12. V5官方来源与配额缺口

V5固定的上游版本：

- `openai/gsm8k`: `740312add88f781978c0658806c59bc2815b9866`
- `tau/commonsense_qa`: `94630fe30dad47192a8546eb75f094926d47e155`
- `derek-thomas/ScienceQA`: `f18b0a70359ebfb41f658fd564208d0355b013f4`
- `docintel/ChartQA`: `8af361fb04d780e9e73e0ac02f3d48f747ec53c4`
- `pppop7/OCR-VQA`: `caf625ea5c5865d494bc19d5be8a52ac3d50d06f`
- `lmms-lab/MMMU`: `364f2e25bdbf23213b3eb20240b3d0b0a1a29590`（V5实际使用并冻结的、与当前loader兼容的LMMs-Lab镜像）
- BIG-Bench-Hard: `9ee07bd481feebf959a6b59d61ea57bdcf30964d`
- Gorilla/BFCL: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`

BFCL静态V4文件实际只有400道`simple_python`和200道`multiple`，不足以提供开发500+500、测试100+100以及额外困难题。禁止重复抽样后，V5采用同一BFCL V4官方仓库内的合法补充文件：

- Simple池：`simple_python` 400、`simple_java` 100、`simple_javascript` 50、`live_simple` 258，共808；
- Multiple池：`multiple` 200、`live_multiple` 1053，共1253。

manifest必须逐题记录具体source file和原始BFCL ID，报告中仍汇总为BFCL Simple / Multiple，并单列各子来源数量。

## 13. 审计结论

旧实验足以确认算法实现和历史结果，但不足以作为V5的数据溯源基础。V5应从固定revision重新构建3200/800 manifest，保存原始ID、split、图片ID、fold、难度原因和hash；在任何最终测试调用前完成开发训练、阈值选择和冻结清单。旧结果只能作为历史比较，不得覆盖或混入新测试选择。
