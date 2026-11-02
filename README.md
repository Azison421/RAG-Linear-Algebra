# RAG 向量检索线性代数大作业

依据上一级目录的 `RAG_向量检索线性代数大作业_V3.1.md` 实施。第八天已补齐两张 PCA 图与七个不同查询的失败案例，方案要求的七份 CSV 和四张 200 dpi 图均已齐全，并同步论文。实验 A/B/C 数值、60 块正文及正式向量、25 条查询与 Gold 保持原样。Q05 的回答覆盖交叠已记入误差分析，未调整标注或成绩。当前等待第八天验收，第九天再整合论文主体。

## 使用环境

在本目录打开 PowerShell，运行：

```powershell
.\.venv\Scripts\python.exe .\verify_environment.py
```

运行其他项目程序时，也使用 `.venv\Scripts\python.exe`。VS Code 的 Python 解释器选择同一路径。

模型从上一级目录的 `模型\bge-small-zh-v1.5` 读取。验证脚本仅使用本地模型文件，三句冒烟模长保存在 `environment-check.json`，不覆盖正式的 `embedding_norms.csv`。

## 安装方式

基础解释器是 `D:\Python\Python312\python.exe`，版本为 3.12.10。该发行版缺少标准 `venv` 模块，因此使用本机已有的 `virtualenv` 创建项目环境。

环境通过 `.venv\Lib\site-packages\base-python-packages.pth` 复用 `D:\Python\Python312\Lib\site-packages` 中的 PyTorch、NumPy、Pandas、scikit-learn 等已有依赖。新下载的包安装在项目的 `.venv` 中；该环境与本机基础 Python 安装有关联。

在已建立的环境中安装直接依赖并检查：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

本次检查依赖已齐全，无需重复下载。

## 已验证的主要版本

| 组件 | 版本 |
| --- | --- |
| PyTorch | 2.12.1+cu132 |
| Sentence Transformers | 6.2.0 |
| Transformers | 5.22.1 |
| NumPy | 2.5.0 |
| Pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| SciPy | 1.18.0 |

`requirements.txt` 列出项目的直接依赖；`requirements-lock.txt` 记录本次 Windows / Python 3.12 环境中这组依赖及其传递依赖的实际版本，包括复用的 CUDA 版 PyTorch。CUDA 版 PyTorch 的安装源需要按设备和官方安装说明选择。

`installation-report.json` 是首次补装依赖时生成的 pip 下载和安装记录。

`dependency-audit-v3.1.json` 是 V3.1 依赖核对记录。PyYAML 和 SciPy 在原环境中已经可用，本次补入直接依赖清单，无需重复下载。SciPy 用于方案中的 Kendall τ 统计。

已使用本机 Microsoft Word 实际导出论文框架预览并逐页检查，封面无页码，后续从 1 编号。

## 验证范围

- 本地模型在 NVIDIA GeForce RTX 5060 Laptop GPU 上加载和编码。
- 三条中文文本生成形状为 `(3, 512)` 的文档向量，一个查询生成形状为 `(1, 512)` 的查询向量。
- 查询使用方案规定的中文检索指令；文档和查询显式开启 L2 归一化。
- 读取 `modules.json` 检查 Normalize 模块，并记录 `normalize_embeddings=False` 时的输出模长及最小值、均值、最大值。
- 验证向量维度、有限数值、单位模长和矩阵相似度计算。
- 验证 Gram 矩阵对称性与半正定性，以及 NumPy SVD 实现的 PCA 投影；用 `eigh` 和 scikit-learn 核对数值。
- 验证 PyYAML 中文配置安全读写，以及 SciPy 的 Kendall τ 统计接口。
- 验证 SQLite 向量写入和读取、Pandas CSV 输出、Matplotlib 中文 PNG 导出。
- `pip check` 未发现依赖冲突。

上述为第一天的三句示例环境检查。正式编码与检索、A/C 实验以及 PCA 维度扫描已分别在第五、六、七天完成，实际结果见对应阶段。

## 第一天产物

- [论文框架](report/course_paper.docx)：沿用老师原封面，A4，正文四号宋体、1.5 倍行距；封面单独一节，后续从 1 编号。Word 导出预览共 6 页，封面 1 页、正文框架 5 页。摘要、指标、结论和参考文献保留待补位置。
- [首批 14 块的核准记录](report/CF-01_CF-27_核准记录.md)：第一天起草 CF-01～CF-14，正文 154–209 个汉字。第三天获用户核准后已入库，原待核稿中的正文已移除。
- [数据库](database/knowledge.db)：只有 `chunks` 表；第一天导入 60 条 ID、分组、主题、标题和对偶关系，当时正文与向量均为 0 条。当前正文数量见本页开头；`content=''` 仍表示仅有目录。
- [配置](experiment_config.yaml)：已填模型路径、实际权重哈希、512 维、固定查询指令、编码参数、归一化、种子与 PCA 维度。`n_chunks=60`、`n_queries=25` 为计划数量。数据冻结哈希和未确认的模型下载日期保留为 NULL。
- [模长诊断](results/embedding_norms.csv)：第一天为三条 smoke 示例及汇总，第五天已替换为 `run_scope=corpus` 的 60 块正式诊断及汇总；第一天示例仍记录在环境 JSON 中。

## 项目结构

```text
RAG-Linear-Algebra/
├── database/
│   └── knowledge.db
├── data/
│   └── queries.csv                  # 25 条已核准查询与唯一 Gold，已冻结
├── src/
│   ├── common.py                    # 配置、冻结检查、本地编码器及 Float32 BLOB
│   ├── build_database.py            # 已实现单表及 60 块目录导入
│   ├── embed.py                     # 正式模长诊断、归一化向量入库
│   ├── search.py                    # NumPy 精确检索与 Top-K 命令行
│   ├── gram_analysis.py             # float64 Gram、秩、谱、统计及绘图
│   ├── pca_analysis.py              # 实验 B：PCA、数值对拍与维度扫描
│   └── evaluate.py                  # 完整排名 MRR、Hit@K、重合率及分层
├── experiments/
│   ├── math_verification.py         # float64 人工数学实验 A1–A5
│   └── run_retrieval.py             # 实验 B：固定查询编码与五种表示检索
├── results/
│   ├── environment-check.json
│   ├── embedding_norms.csv          # 60 块正式诊断及汇总
│   ├── embedding_run.json           # 正式编码参数、模型哈希、版本与设备
│   ├── synthetic_results.csv       # A1–A5 逐项结果
│   ├── gram_matrix.csv             # 按图中 ID 顺序排列的原始 G
│   ├── eigenvalues.csv             # G/Gc 原始特征值与奇异值平方
│   ├── gram_analysis.json          # 秩、阈值、误差及 15 对统计
│   ├── pca_results.csv              # 四维度解释方差、残差、投影和存储
│   ├── pca_analysis.json            # SVD/eigh/sklearn 数值核对及全谱
│   ├── metrics.csv                 # 五方法 × 整体/三类分层，20 行
│   ├── query_metrics.csv           # 125 条逐题 Gold 排名及指标
│   ├── retrieval_rankings.csv      # 全部 7500 个候选得分与名次
│   ├── query_embeddings.npy        # 本次 25×512 Float32 查询输出
│   ├── retrieval_run.json          # 查询编码、文件关联与真实度量核对
│   ├── failure_cases.csv          # 七个不同查询：目标、Top-3、分数及原因
│   └── figures/
│       ├── gram_heatmap.png
│       ├── eigenvalue_spectrum.png
│       ├── pca_variance.png
│       └── pca_comparison.png
├── report/
│   ├── course_paper.docx
│   ├── CF-01_CF-27_核准记录.md       # 正文已入库，只保留核准与来源元数据
│   ├── LA-GR_核准记录.md
│   └── RG-01_RG-10_核准记录.md
├── experiment_config.yaml
├── CHANGELOG.md
├── requirements.txt
├── requirements-lock.txt
├── verify_environment.py
└── README.md
```

本地还有 `.venv`、`.runtime-cache` 和前述依赖安装、核对记录。方案约定的程序均已实现；七份必需 CSV 与四张 200 dpi PNG 图均已生成。完整查询排名、逐题指标和运行 JSON 作为可复算结果一并保留。

## 配置与模型版本

`model_dir` 相对于配置文件所在目录解析，当前为 `../模型/bge-small-zh-v1.5`。验证脚本设置离线模式，以 `local_files_only=True` 加载，并实际计算权重哈希与配置比较。

```text
354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026
```

固定查询指令为 `为这个句子生成表示以用于检索相关文章：`。文档不加指令，文档和查询显式开启归一化。环境结果中保存模型模块、tokenizer、编码参数、设备与依赖版本；本地模型含 Normalize，与原始模长诊断一致。

随机种子为 20261031，PCA 维度为 `[59, 32, 16, 8]`。正文、查询及 Gold 已核准冻结，数据哈希已填写。PCA-59 按方案第 6.6 节验证中心化文档内积的保留，不预设与未中心化基线同排序。

## 建库与正文审阅

在本项目目录执行：

```powershell
.\.venv\Scripts\python.exe .\src\build_database.py --config .\experiment_config.yaml
.\.venv\Scripts\python.exe .\src\common.py --config .\experiment_config.yaml
```

建库命令仅补入缺少的目录记录，保留已存在记录的正文和向量。CF-03 对 CF-04/CF-05、CF-11 对 CF-10/CF-12、CF-22 对 CF-21/CF-23 均已记录。

用户已明确确认“CF-01～CF-27 全部已核准，可入库”。这 27 块按原正文导入 `chunks.content`，写入后逐块读回比对一致。原两份 CF 待核稿已移除，`report/CF-01_CF-27_核准记录.md` 仅保留核准、来源、字数和单块正文哈希，不复制正文。

用户已确认“第三天内容确认无误”，LA、GR 共 23 块按原正文入库，原两份待核稿已移除，来源和核准信息见 `report/LA-GR_核准记录.md`。第四天用户又明确核准 RG 与 25 条查询，可以入库并冻结；RG 正文已原样入库，审阅稿移除正文，来源与核准信息见 `report/RG-01_RG-10_核准记录.md`。知识正文只以 SQLite 为准，查询及唯一 Gold 统一保存在 `data/queries.csv`。没有另建 `corpus.jsonl` 或 `judgments.csv`。

主要来源为方案文献 [9] Strang 第五版，PCA 与协方差补充使用方案 [10] Jolliffe、Cadima。CF-20、CF-23、CF-27、GR-07 的数据库主来源采用 [10]，GR-08 采用方案 [3] Eckart、Young，其余 CF、LA、GR 块采用 [9]；补充来源保留于核准记录。RG 使用方案 [4]、[5]、[6]、[7]、[11] 与 BGE 官方模型卡，均附链接。作者书页直接访问超时时，另列出版方书目；未声称已读取不可访问全文。正文、来源和查询标注已获用户核准。

## 第一天验收

实际执行：

```powershell
.\.venv\Scripts\python.exe .\verify_environment.py
.\.venv\Scripts\python.exe .\src\build_database.py --config .\experiment_config.yaml
.\.venv\Scripts\python.exe .\src\common.py --config .\experiment_config.yaml
```

实际关键输出：

```text
status: passed
device: cuda (NVIDIA GeForce RTX 5060 Laptop GPU)
document_vector_shape: [3, 512]
normalize_embeddings_false_norms: [1.0, 0.9999999403953552, 1.0]
raw norm min / mean / max: 0.9999999403953552 / 1.0 / 1.0
document_vector_norms: [1.0, 1.0, 1.0]
has_normalize_module: true
pip_check: No broken requirements found.
groups: CF=27, LA=15, GR=8, RG=10; total=60
正文已入库=0 向量已入库=0
corpus_sha256=None queries_sha256=None
PRAGMA integrity_check: ok
```

已核对数据库 15 个字段、60 个固定 ID 与标题、15 组双向对偶关系及草稿字数。环境级 NumPy SVD、`eigh`、sklearn、YAML、Kendall τ、SQLite BLOB 和中文绘图检查通过，示例检查不属于正式实验 A/B/C。Word 逐页检查通过，封面无页码，其后为 1–5。

## 第二天产物与验收

- [第二批 13 块的核准记录](report/CF-01_CF-27_核准记录.md)：第二天起草 CF-15～CF-27，正文 184–237 个汉字；第三天获用户核准后已入库，原待核稿中的正文已移除。CF-22 同时记录 CF-21、CF-23 两个对偶块。
- [论文框架](report/course_paper.docx)：补写“三、2、数据来源与文本整理”和“三、3、文本分块与元数据设计”，补入两条已引用且待用户核对的参考文献，区分草稿齐备与用户核准入库，保留原有封面、样式、公式和页码设置。方案文献 [9]、[10] 在当前论文中按首次引用顺序暂编为 [1]、[2]。
- 本轮核对 13 个固定 ID、标题、对偶关系、汉字数、公式分隔符和来源，并已由一名子代理独立复核，未发现阻塞交付的问题。来源核查与代理复核均不代替用户确认。
- 第二天交付时数据库为 60 条目录、正文 0 条、向量 0 条；`queries.csv` 只有表头，数据冻结哈希为 NULL；当日未运行正式编码、检索或实验。

第二天实际验收：使用项目 Python 只读统计并逐项对照数据库目录，13 块的 ID、标题、对偶关系、待核状态、来源、字数及公式分隔符检查通过；SQLite 当时输出 `(60, 0, 0)`（总记录、已有正文、已有向量），完整性检查为 `ok`。第二天验收时数据库、查询、配置和第一天稿件与首个提交一致。DOCX 打包校验通过，本机 Word 当时导出为 7 页，封面无页码，正文页码为 1–6。`git diff --check` 通过。

第二天按方案阶段顺序提前执行，实际执行日期为 2026-10-31，计划日期为 2026-11-01。第二天新稿使用基础排版命令；原两份 CF 审阅稿的正文已在第三天用户核准后移入 SQLite，现通过核准记录查找来源和块 ID。

## 第三天产物与验收

- [LA 组核准记录](report/LA-GR_核准记录.md)：第三天起草 15 块基础知识，正文 166–224 个汉字；第四天获用户核准后原样入库，原稿已移除正文。
- [GR 组核准记录](report/LA-GR_核准记录.md)：第三天起草 8 块数学主线知识，正文 158–225 个汉字；第四天获用户核准后原样入库，保留来源、字数和单块哈希。
- [CF 核准记录](report/CF-01_CF-27_核准记录.md) 与 [数据库](database/knowledge.db)：第三天交付时 27 块已核准入库，来源字段齐全；当时尚无正式向量、未冻结数据。单块核对哈希不作为语料冻结哈希。
- [论文](report/course_paper.docx)：补写“二、6、Gram 矩阵的构造与性质”和“二、7、Gram 矩阵的秩、零空间与半正定性”，使用可编辑公式，同步更新已核准正文和待核稿数量。引用、推导及实验结论分别标明，未预填实际秩或检索指标。

第三天实际执行日期为 2026-10-31，计划日期为 2026-11-02。第三天交付时 LA、GR 的 23 块仍待用户核准；当时未起草 RG、查询或 Gold，也未执行正式向量化、检索及实验 A/B/C。

第三天验收：23 块的数量、固定 ID、标题、待核状态、来源、汉字数、公式分隔符及花括号检查通过；27 块已核准 CF 正文逐块读回一致，核准记录中的 27 个正文哈希全部匹配。查询、实验配置和开工前已有的环境检查记录保持原样。DOCX 打包校验输出 `All validations PASSED!`；论文新增 7 条可编辑公式，共 8 条，封面、校徽、样式、分节、页脚及原公式保留。Word 实际导出 8 页，封面无页码，正文页码 1–7，逐页检查通过。

第三天当时的数据库只读输出为 `(60, 27, 0)`、`ok`，`git diff --check` 通过。三个数分别为目录数、已入库正文数和向量数。

第三天最终由一名子代理独立复核数学表述、23 块新稿、CF 正文与核准记录、论文及项目状态，未发现阻塞交付的问题。LA、GR 后续已由用户核准；代理复核本身不代替用户确认。

## 第四天产物与验收

- [RG-01 至 RG-10 核准记录](report/RG-01_RG-10_核准记录.md)：10 块正文 186–238 个汉字，已获用户核准并原样入库，记录只保留来源、字数与正文哈希。
- [查询与 Gold](data/queries.csv)：25 条，`contrast=15`、`paraphrase=5`、`formula=5`，已获用户核准并冻结。Q01～Q15 对应 P01～P15，每题只问一个概念，notes 写明对偶块为何不作主目标；Q16～Q20 不含目标标题关键词；每题只有一个主 Gold，`acceptable_ids` 当前均为空。
- [LA、GR 核准记录](report/LA-GR_核准记录.md) 与数据库：新增 23 块已按用户确认原样入库；当时共 50 块，随后 RG 核准入库达到 60 块，SQLite 为唯一正式正文源。
- [论文阶段稿](report/course_paper.docx)：同步补写查询设计、Gold 标注与冻结方法，更新正文数量，保持实验结果待测。
- `src/build_database.py --freeze`：全部正文、来源和查询经用户核准后，已执行完整性校验并将两项哈希写入原配置。

第四天实际执行日期为 2026-10-31，计划日期为 2026-11-03。没有运行正式编码、检索或实验，也未依据模型排序调整题目与 Gold。

实际验收：RG 的固定 ID、标题、来源、字数及公式检查通过；25 条查询的六列表头、唯一 ID、15/5/5 分类、Gold 目标、全部概念对覆盖和同义改写关键词检查通过。核准后 60 条正文均有来源，0 条向量；查询核准时仅移除 notes 中的待核前缀，未改动问题、类型、Gold 或 acceptable_ids。DOCX 打包校验通过，Word 导出 9 页，封面无页码、正文页码 1–8；封面、校徽、样式、分节、页脚和原有 8 条可编辑公式均保留。

由一名子代理逐题对照目标正文并复核 RG 内容、冻结代码与项目状态。已修正 Q18 同义改写漏掉一次项条件的问题，保持原 Gold 和标题关键词规避；复查通过后未发现阻塞技术问题。随后用户明确核准，RG 正式入库并完成数据冻结。

核准入库后，同一名子代理独立重算两项冻结哈希，核对全部 60 个正文哈希、来源和 25 条查询；最终状态复核通过。论文结果表保持在同一页，正式实验指标仍待测。

## 数据冻结方法

语料哈希由 `src/common.py` 的 `corpus_text_sha256` 生成：按 `chunk_id` 排序，取 `chunk_id`、`group_code`、`topic`、`title`、`content`、`confusable_with`、`source_title`、`source_author`、`source_year`、`source_url` 十个字段，转为字典列表，以 `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))` 的 UTF-8 字节计算 SHA-256。模型名称、版本、模长、维度和向量不参与，因此第五天写入向量不改变正文指纹。

查询哈希对 `data/queries.csv` 的原始 UTF-8 字节计算；`.gitattributes` 固定此文件使用 LF，避免 Git 换行转换改变哈希。`acceptable_ids` 若填写，使用块 ID 的 JSON 列表，只参与误差分析。

本轮已在用户核准 RG 与 25 条标注后，原样导入 RG 正文并移除审阅稿正文，仅移除查询 notes 的“待核；”前缀，保留其余标注依据，再运行：

```powershell
.\.venv\Scripts\python.exe .\src\build_database.py --config .\experiment_config.yaml --freeze
```

命令拒绝正文不齐、仍待核的查询、无效 Gold、错误的类型数量或对偶组覆盖，也拒绝静默覆盖不匹配的既有冻结哈希。冻结后正文、查询或标注如有变化，先按 CHANGELOG 要求记录原因。

冻结工具已在临时人工数据上验证：写入向量和模型字段后文本哈希不变；修改正文或查询会被发现；重复冻结不重写配置；待核查询不能冻结。临时数据不属于本项目语料或正式实验。

第四天冻结后只读复查数据库的命令：

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('file:database/knowledge.db?mode=ro',uri=True); print(c.execute('SELECT COUNT(*), SUM(length(content)>0), COUNT(embedding) FROM chunks').fetchone()); print(c.execute('PRAGMA integrity_check').fetchone()[0])"
git diff --check
```

第四天实际输出为 `(60, 60, 0)`、`ok`；第五天向量入库后为 `(60, 60, 60)`、`ok`。正式冻结值已写入 `experiment_config.yaml`：

```text
corpus_sha256: f9c6effe0297fb76aca84435a37a39b8a48ff2f567982778cd852070db3e4569
queries_sha256: d09ec555e741b246a30a2915a0bf8fb4b4382dece4507d640229f1d7c4e53293
```

## 第五天产物、运行与验收

第五天实际执行日期为 2026-10-31，计划日期为 2026-11-04，按阶段顺序提前执行。

- `src/embed.py`：冻结检查、全库模长诊断、显式归一化、512 维 Float32 BLOB 入库及逐项回读比对。
- `src/search.py`：一句话命令行检索，返回 ID、标题、分数、正文和来源。按分数降序，完全同分按块 ID 升序；计算接口保留全部 60 个名次，供后续 MRR 使用。
- `results/embedding_norms.csv`：60 个块及一条汇总，`run_scope=corpus`。模长以 float64 统计。
- `results/embedding_run.json`：实际模型权重 SHA、前十二位版本号、Normalize 模块、tokenizer、编码参数、依赖版本、设备、数据冻结哈希及向量大小。未确认的模型下载日期仍为 NULL。
- `report/course_paper.docx`：补写 BGE 向量化、模长诊断、SQLite 向量存储和 NumPy 检索，加入模型卡引用及真实运行示例；整体评价指标仍待测。

在项目目录执行：

```powershell
.\.venv\Scripts\python.exe .\src\embed.py --config .\experiment_config.yaml
.\.venv\Scripts\python.exe .\src\search.py --query "为什么 Gram 矩阵一定是半正定的？"
.\.venv\Scripts\python.exe .\src\search.py --query "怎样降低向量的坐标数量？" --top-k 5
```

`--config`、`--database`、`--queries` 均可显式指定路径；编码另有 `--output-dir`。默认路径以脚本所在项目为准，因此也可用脚本绝对路径从其他目录运行。重新执行编码会更新模型及向量字段、模长 CSV 和运行记录，保留正文、来源和 Gold；检索以只读连接打开数据库。`--top-k` 范围为 1～60，默认 3。

编码只读取 `chunks.content`，不拼接标题、来源和查询指令。查询统一加固定指令，文档和查询均显式设置 `normalize_embeddings=True`；原始模长另以 `False` 单独诊断。实现核对了 [BGE 官方模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5) 与 [Sentence Transformers 编码接口](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)，使用已安装版本的接口。

正式编码实际输出：

```text
raw norm min/mean/max: 0.999999945464 / 1.000000004639 / 1.000000077804
正式向量 shape=(60, 512) dtype=float32 max_norm_error=8.44e-08
已写入 60 条向量
```

本地模型包含 Normalize，因此 `normalize_embeddings=False` 时最终输出也已接近单位模长；这里的 raw_norm 不是未经模型内部归一化的隐藏层模长。60 条 BLOB 每条 2048 字节，合计 122880 字节（120 KiB），只计向量本身。编码设备为 CUDA / NVIDIA GeForce RTX 5060 Laptop GPU。模型版本字段全部为 `354763b9b135`。

上述首个检索命令实际返回：

```text
得分矩阵 shape=(1, 60)，返回 Top-3
1. GR-02 Gram 矩阵的半正定性 | score=0.651128
2. CF-18 半正定矩阵 | score=0.643381
3. CF-19 Gram 矩阵 | score=0.614524
```

命令还逐条输出正文、作者、年份、文献名和来源链接。该例只验证运行链路，不代替 25 题的正式效果评价。

必要验收已检查数据库 `(60, 60, 60)`、`PRAGMA integrity_check: ok`、512 维和 BLOB 字节数、逐块 raw_norm 与 CSV 一致、单位模长、冻结哈希及来源保持。25 条查询批量编码形状为 `(25, 512)`，得分及完整名次矩阵均为 `(25, 60)`，矩阵乘法与逐对内积一致，同分排序规则通过；查询最大模长偏差为 `7.04925433581e-08`。正文最大 token 数为 324，小于 512，均未截断。临时内存副本中改正文与缺失向量均会被拒绝，正式数据库未改动。

本轮未计算 Hit@K、MRR，也未进行正式数学验证、Gram 分析或 PCA 实验。第六、七天程序仍为骨架，数据正文、查询、标注及实验配置保持批准版本。本轮不提交或推送 Git。

DOCX 打包校验通过；本机 Word 实际导出 10 页，封面无页码、正文页码 1–9，逐页视觉检查通过，结果表完整位于同一页。原有校徽、样式、分节、页脚及 8 条可编辑公式保留。修改后的环境检查另在临时输出目录实际运行，状态为 `passed`，未写入模长 CSV，正式模长文件的字节哈希保持一致；原环境记录未改动。

最终由一名子代理只读复核通过，未发现阻塞第五天交付的问题。子代理独立重算模型及两项冻结哈希，核对向量、模长记录和论文，并从项目外目录实际运行 Top-5 CLI，五条结果均有正文和来源，检索前后数据库字节哈希不变。临时验收脚本、环境输出、PDF 和页面预览已在交付前清理。

## 第六天产物、运行与验收

第六天实际执行日期为 2026-11-01，计划日期为 2026-11-05，按阶段顺序提前执行。程序和输入沿用已验收的第五天版本，本轮没有重新编码、修改向量或调整查询和 Gold。

在项目目录执行以下命令，可重新计算数据并生成两张图；也可从其他目录用脚本绝对路径运行：

```powershell
.\.venv\Scripts\python.exe .\experiments\math_verification.py --config .\experiment_config.yaml
.\.venv\Scripts\python.exe .\src\gram_analysis.py --config .\experiment_config.yaml
```

两个程序均支持 `--output-dir`，Gram 程序另支持 `--database` 和 `--queries`。结果默认写入项目 `results/`，Gram 图写入其 `figures/` 子目录；重新运行 Gram 命令即可重建图表，无需 BGE 推理。两者的 `--help` 已实际运行。

- [人工向量结果](results/synthetic_results.csv)：四列为“实验、指标、数值、是否通过”，共 25 条计算或检查记录，覆盖 A1–A5。
- [原始 Gram 矩阵](results/gram_matrix.csv)：含行列 ID 的 60×60 矩阵，顺序与热力图一致；正文不另存副本。
- [特征值](results/eigenvalues.csv)：G、Gc 各 60 条，保存原始特征值、对应奇异值平方和绝对误差。
- [Gram 数值记录](results/gram_analysis.json)：保存两组数值秩、各自阈值、对称与谱误差、均值残差、15 对逐对相似度与整体均值、ID 顺序、依赖版本和冻结哈希。
- [Gram 热力图](results/figures/gram_heatmap.png)、[特征值谱](results/figures/eigenvalue_spectrum.png)：分别为 1800×1600 和 1623×1057 像素，均为 200 dpi。
- [论文](report/course_paper.docx)：补写归一化排序等价、手算反例、紧致 SVD 与 Gram 谱、正交不变性，以及 A/C 实验结果和两张图；PCA 及整体检索指标仍待测。

人工实验固定 `random_seed=20261031`，使用 `numpy.random.default_rng` 和 float64。A2 为 1000 对 512 维单位向量；A3 每次重新生成 60 个模长为 3 的文档和未单位化查询，重复 1000 次。A4 每次生成 60 个单位文档、一个单位查询，再为各文档独立抽取 `U[0.5,2]` 系数，重复 1000 次；Top-3 重合率为交集个数除以 3，Kendall τ 按同一文档在两种完整排序中的名次计算。A5 使用 512 阶随机矩阵的 QR 正交因子，同时右乘查询和文档行向量。A4 的“通过”只表示计算有效，不设置优劣阈值，其数值不属于 BGE 准确率。

实际关键输出：

```text
A1: 7/7 检查通过
A2: 2/2 检查通过
A3: 5/5 检查通过
A4: 4/4 检查通过
A5: 7/7 检查通过
A2 identity_max_abs_error=8.881784197e-16
A3 full_ranking_agreement_rate=1
A4 top1_agreement_rate=0.074
A4 mean_top3_overlap=0.108
A4 mean_kendall_tau=0.0594915254237
A5 inner_product_max_abs_error=1.03028696685e-13
A5 cosine_max_abs_error=2.01227923213e-16
A5 l2_max_abs_error=7.1054273576e-15
```

Gram 分析只读打开 SQLite，将 60×512 的 Float32 向量先转为 float64，计算 `G=X@X.T`，逐列减去文档均值得到 Xc，再计算 `Gc=Xc@Xc.T`。没有将查询得分矩阵当作 Gram 矩阵，也没有把中心化上界 59 用于原始矩阵。

| 检查 | 原始 X / G | 中心化 Xc / Gc |
| --- | --- | --- |
| 矩阵秩 / Gram 秩 | 60 / 60 | 59 / 59 |
| 对称最大误差 | 0 | 0 |
| 最小特征值 | 0.0289246520456 | 2.62227424426e-16 |
| 特征值与奇异值平方最大差 | 7.10542735760e-15 | 3.33066907388e-15 |

`np.linalg.matrix_rank` 与显式 SVD 计数独立核对，采用 [NumPy 默认规则](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)：`sigma_max * max(shape) * float64_eps`。X、G 的阈值分别约为 `7.116e-13`、`5.219e-13`，Xc、Gc 分别约为 `1.796e-13`、`3.325e-14`；这些是各自矩阵的数值判定阈值，不是人为统一为零。半正定容差为 `-1e-10`；谱关系用 `rtol=1e-8, atol=1e-10` 检查，包含中心化的数值零谱。

中心化列和最大残差为 `1.33226762955e-15`。15 个易混淆对的平均余弦为 `0.840857933568`，全体非对角元素平均为 `0.644701165424`；后一统计包括对称的 3540 个单元格，等价于 1770 个无序文档对的均值。前者说明概念对在本向量空间内较近，是否误检留待查询评测判断。

热力图按 CF/LA/GR/RG 分组，CF-04、CF-03、CF-05 连排，使共享 CF-03 的两对也相邻；15 对均已检查相邻。图和矩阵 CSV 使用同一 ID 顺序，与数据库读取顺序的关系记录于 JSON。对数谱图仅将低于约 `5.219e-13` 的数值画在显示下限，不改动 CSV 原始值，秩仍按各矩阵自己的阈值判定。

必要验收已核对 25 条人工记录、120 条特征值记录、Gram CSV 与数据库重新相乘逐项一致，以及两个 PNG 的 200 dpi 元数据和视觉内容。另用已知秩的矩阵、秩亏矩阵和零矩阵核对谱与秩；修正了矩形矩阵比较谱时补齐零特征值的处理。数据库仍为 `(60, 60, 60)`、完整性 `ok`；数据库文件、正文与查询冻结哈希、编码配置、模长文件和第五天程序均与本轮开工前一致。未计算 PCA、Hit@K、MRR，未改动第七天骨架。

论文打包校验通过，本机 Word 导出为 14 页，封面无页码，正文页码为 1–13。逐页视觉检查通过；原有 8 条公式保留，新增 6 条，共 14 条可编辑公式。原校徽、样式、分节及页脚保持，两张内嵌图与当前 PNG 一致，结果表完整在同一页。附录的程序说明已同步到第六天，PCA 与检索评价保留待测。本轮不提交或推送 Git。

最终由一名子代理只读复核通过，未发现阻塞第六天交付的问题。子代理在内存独立重跑 A1–A5，25 条记录精确复现；只读重算 G/Gc、秩、谱和对偶统计，与文件一致。两张图、论文和文档记录一致，13 个基线文件与两项冻结值保持。临时验收脚本、PDF、页面预览及解包目录已在交付前清理。

## 第七天产物与验收

实际执行日期为 2026-11-01，计划日期为 2026-11-06，按用户要求提前执行。只推进第七天，图表补全和失败案例留待第八天。

- [PCA 数值结果](results/pca_results.csv)：59/32/16/8 四行，含累计解释方差、重建残差平方、奇异值尾能量、中心化 Gram 误差、距离检验、子空间对拍、最小投影模长和 Float32 向量存储。
- [PCA 运行记录](results/pca_analysis.json)：文档顺序、冻结值、float64 阈值、完整奇异值/eigh 谱、解释方差及数值检查。
- [整体与分层指标](results/metrics.csv)：五方案 × all/contrast/paraphrase/formula，共 20 行；含样本数、Hit@1/3、完整 MRR、overlap@3、成功与失败题数、累计解释方差。
- [逐题指标](results/query_metrics.csv)：125 行，保存唯一 Gold 的完整名次、倒数排名和命中标记。
- [完整得分与排名](results/retrieval_rankings.csv)：5×25×60=7500 行，支持独立重算指标和下一阶段案例核查。
- [查询向量](results/query_embeddings.npy) 与 [检索运行记录](results/retrieval_run.json)：25×512 Float32 模型输出及对应查询顺序、编码参数、模型哈希、设备、依赖、各表示的度量排序一致率。文件 SHA 将本次排名、PCA CSV 与查询向量关联，避免混用不同运行的产物。
- [论文阶段稿](report/course_paper.docx)：补写 PCA、投影与低秩近似推导、实验 B、结果表、分层观察及存储口径；摘要、失败案例和论文整体整合按后续计划推进。

### 运行命令

在项目目录使用 PowerShell：

```powershell
# 完整实验 B：编码固定查询一次，生成全部 PCA 与检索结果
.\.venv\Scripts\python.exe .\experiments\run_retrieval.py --config .\experiment_config.yaml

# 只重新计算文档 PCA，不加载 BGE、不生成检索排名
.\.venv\Scripts\python.exe .\src\pca_analysis.py --config .\experiment_config.yaml

# 从保存的完整排名重算指标，不加载 BGE
.\.venv\Scripts\python.exe .\src\evaluate.py --config .\experiment_config.yaml
```

三脚本均支持 `--config`、`--database`、`--queries`、`--output-dir`；默认路径基于脚本位置解析。evaluate 另有 `--rankings`、`--pca-results`、`--run-info` 指定同一轮实验的三个输入；改变输出目录不改变默认输入。自定结果目录时，将三输入一并指向对应目录。所有命令先核对冻结正文和查询；数据库始终只读。evaluate 还核对运行记录中的 CSV 哈希和完整排名。

### 数学与编码口径

PCA 主流程为 [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)：库内原 Float32 值转 float64，减文档均值后分解。查询只使用这一个文档均值和右奇异向量矩阵，投影后双方重新按行单位化。零投影不能计算余弦，明确报错；本次四种维度的最小查询投影模长分别约为 0.4292、0.3442、0.2764、0.1754，没有发生零投影。

基线不中心化，将存储/编码的近单位 Float32 输出转 float64 后重新单位化，以精确计算余弦；这一处理不回写数据库。PCA 拟合前不另行单位化，以保持与第六天中心化矩阵相同。五方法都复用本次同一份查询编码，文档只用冻结正文，查询使用固定官方前缀，其他编码参数保持第五天版本。

[eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html) 仅用于对 `Xc.T @ Xc` 的谱作交叉验证；对非零谱比较相对误差 <1e-8，对数值零谱比较绝对值 <1e-10，避免除以零。[sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 采用 `svd_solver="full"`，只对拍 `sigma²/(n-1)` 解释方差和投影子空间，不进入主检索。以投影矩阵比较子空间，避免奇异向量正负号的任意性。固定种子仍为 20261031。

投影基正交性、文档对距离不扩大、Frobenius 重建残差等于奇异值尾能量均在重新归一化前验证。59 维仅保证保留中心化文档 Gram，不能保证与未中心化基线同排序；按方案第 6.6 节的数学约束验收，第 7.2 节的“高度一致”只作预期描述。

### 实际结果

```text
PCA centered_rank=59
eigh_relative_error=3.20326761437e-15
eigh_zero_spectrum_max_abs_error=3.90451977404e-16
sklearn_variance_error=8.32667268469e-17
PCA-59 centered_gram_max_abs_error=5.82867087928e-16
all PCA checks passed=True
saved rankings=7500 metrics=20 per_query=125
```

| 方案 | Hit@1 成功/失败 | Hit@3 成功/失败 | MRR（60 候选） | overlap@3 | 累计解释方差 |
| --- | --- | --- | --- | --- | --- |
| Baseline 512 | 14/11 | 22/3 | 0.71374163 | 1（自身） | 基准表示 |
| PCA-59 | 13/12 | 21/4 | 0.69457143 | 0.81333333 | 1.00000000 |
| PCA-32 | 13/12 | 19/6 | 0.66477193 | 0.78666667 | 0.89566493 |
| PCA-16 | 10/15 | 19/6 | 0.60142857 | 0.58666667 | 0.69827308 |
| PCA-8 | 8/17 | 16/9 | 0.52260991 | 0.53333333 | 0.50256215 |

成功与失败两列均以 25 题为总数；每题对整体 Hit@1 影响四个百分点。主指标只用 `gold_chunk_id`，`acceptable_ids` 不计入。MRR 使用全部 60 个候选，不是 MRR@3；overlap@3 为与基线两个 Top-3 集合交集大小除以 3，指标不要求内部次序一致。

| 方案 | contrast Top-3 | paraphrase Top-3 | formula Top-3 |
| --- | --- | --- | --- |
| Baseline | 13/15 | 4/5 | 5/5 |
| PCA-59 | 13/15 | 4/5 | 4/5 |
| PCA-32 | 12/15 | 4/5 | 3/5 |
| PCA-16 | 14/15 | 2/5 | 3/5 |
| PCA-8 | 11/15 | 3/5 | 2/5 |

PCA-59 没有任何一条查询的完整排名与基线相同，但 Top-3 平均仍重合 81.33%；保留中心化 Gram 不等于保留未中心化余弦几何。四维度累计方差随维度减少而下降，本测试集整体指标也下降；分层和单题不必单调，例如 PCA-16 的概念区分题 Top-3 多命中一题，另两类各少两题。这里报告 25 题上的描述结果，具体错误原因留待第八天逐题核查。

五种表示各自重新单位化后，25 条查询的内积、余弦、直接 L2 完整排序一致率均为 100%，恒等式误差最大 1.77635683940e-15。同分统一按块 ID 升序。这是度量等价性实测，不能写成三种性能不同的方法。

存储只计 Float32 文档向量：512/59/32/16/8 维每条为 2048/236/128/64/32 B，相对基线为 100%/11.5234%/6.25%/3.125%/1.5625%。PCA 在 float64 上计算；这里是存储表示的理论字节数，不含均值、投影矩阵、数据库及结果文件开销。

### 验收记录

PCA 与 evaluate 分别独立运行，产生的 `pca_results.csv`、`metrics.csv`、`query_metrics.csv` 与完整实验输出逐字节一致。独立 NumPy 复算全部 7500 个余弦得分和完整排列、直接按 Gold 重算 20 行整体/分层指标均通过。另用 Gold 在第 4 名的人工例确认 MRR=0.25、Hit@3=0，其他 acceptable ID 不改变主指标；同分按 ID、截断排名拒绝也通过。

17 个开工前文件的字节 SHA 保持一致，包含数据库、配置、查询、第五/六天程序、模长和 A/C 产物；两项冻结值保持，未重新编码文档或修改 Gold。论文新增方案内 Eckart–Young 文献，按首次引用将 BGE 模型卡顺延为 [4]；[出版方书目](https://link.springer.com/article/10.1007/BF02288367) 的作者、1936 年、卷页和 DOI 已核查，新条目仍待用户核对。

DOCX 打包校验通过；本机 Word 导出 17 页，封面无页码，正文 1–16。全部页面预览及新增公式、结果表检查通过；14 条原公式保留，新增 7 条，共 21 条可编辑公式。原样式、分节、页脚及内嵌图字节保持，结果表完整在物理第 14 页。正式 DOCX 与已检查的导出来源逐字节一致，git diff --check 通过。本轮不提交或推送 Git。

最后由一名子代理只读复核通过，未发现阻塞第七天交付的问题。独立重算 SVD/eigh/sklearn 与四维度投影，7500 个得分及完整排名、125 条逐题和 20 行整体/分层指标均一致；核查论文公式、实测结果、版式及阶段边界通过。17 个基线文件及冻结值保持，HEAD 仍为第二次修改（D4），暂存区为空。本轮临时检查输出、DOCX 解包目录、PDF 和页面预览已清理。当前等待用户验收；第八天再生成 PCA 图与至少五个失败案例。

## 第八天产物与验收

实际执行日期为 2026-11-01，计划日期为 2026-11-07，按用户要求提前执行。本轮只补齐图表、失败案例与当天论文内容；第九、十天的论文整合仍待后续阶段。

- [pca_variance.png](results/figures/pca_variance.png)：1840×920、200 dpi，左侧是中心化文档数值秩内的奇异值谱，右侧是累计解释方差，标记 8/16/32/59 维。第六十个奇异值约 7.02e-16，图中注明其值，原始全谱保留在既有 JSON。
- [pca_comparison.png](results/figures/pca_comparison.png)：1700×1440、200 dpi，上方比较五方案的 Hit@1、Hit@3 和完整 MRR，命中柱标注成功题数；下方是四种 PCA 与基线的 Top-3 集合重合率。基线作为独立类别展示，不伪造其中心化解释方差。
- [failure_cases.csv](results/failure_cases.csv)：7 行、7 个不同查询，包含基线全部三条 Top-3 未命中，以及 59/32/16/8 四种 PCA 各一条。每行保存完整题目、唯一 Gold 及标题、Gold 名次/分数、前三块的 ID/标题/分数、对照方法及目标名次、分类与文本分析。
- [论文阶段稿](report/course_paper.docx)：插入图 3/4、七例对照表、逐类解释和局限性。原有公式、图片及样式保留，摘要、绪论与全文压缩在后续阶段完成。

### 重建图表和案例

在项目根目录运行：

```powershell
# 重建原始与中心化 Gram 的两张图，同时重新计算实验 C 数值
.\.venv\Scripts\python.exe .\src\gram_analysis.py --config .\experiment_config.yaml

# 从第七天保存的 PCA CSV/JSON 重建方差图，不重新拟合或重写数值
.\.venv\Scripts\python.exe .\src\pca_analysis.py --plot-only --config .\experiment_config.yaml

# 从完整排名重建指标对照图与七例 CSV，不重写 metrics/query_metrics
.\.venv\Scripts\python.exe .\src\evaluate.py --analyze-only --config .\experiment_config.yaml
```

本轮仅实际执行后两条；Gram 两图及全部 A/B/C 数值复用已验收结果，保持字节不变。绘图与案例均不加载 BGE，不重新编码查询或文档。PCA 默认运行（不加 plot-only）仍重新拟合、数值对拍并输出方差图；evaluate 默认运行仍重算指标，并同时生成对照图和案例。

PCA 的 `--results-dir` 指定 plot-only 的 PCA CSV/JSON 输入目录；`--output-dir` 指定输出目录，图位于其中的 figures 子目录。evaluate 的 `--rankings`、`--pca-results`、`--run-info` 指向同一轮排名实验；`--output-dir` 控制新图与案例位置。两个程序从项目目录外运行也可使用默认输入路径。

### 案例与解释

本阶段的“失败”严格指冻结主目标未进入 Top-3，不将所有返回块一概判为无关。分类由对照查询和库内正文作出，固定选择与人工说明保存在 evaluate.py 的 CASE_SELECTION；数值字段来自完整排名，程序核查每个案例确实仍是 Top-3 未命中。

| 案例 | 查询 / 方法 | Gold 名次 | 对照方法 / Gold 名次 | 分类 |
| --- | --- | --- | --- | --- |
| F01 | Q02 / Baseline | 5 | PCA-16 / 3 | 概念相邻 |
| F02 | Q05 / Baseline | 11 | PCA-16 / 3 | 标注歧义 |
| F03 | Q18 / Baseline | 19 | PCA-16 / 8 | 术语不同 |
| F04 | Q21 / PCA-59 | 5 | Baseline / 1 | 概念相邻 |
| F05 | Q01 / PCA-32 | 4 | Baseline / 1 | 概念相邻 |
| F06 | Q16 / PCA-16 | 7 | Baseline / 2 | 术语不同 |
| F07 | Q17 / PCA-8 | 27 | Baseline / 1 | 术语不同 |

Q02 的返回内容与三角不等式及方向相邻，未完整给出余弦距离的定义和范围；Q21 返回一般半正定内容，未替代所要求的长度平方等式。Q21 的 59 维变化发生在保留全部非零文档主成分时，不能归因于截去非零主成分。Q01 的目标由基线首位降至32维第4名，说明文档解释方差约89.57%不保证该题命中。

Q18、Q16、Q17 都是需要理解表达或条件关系的查询。Q18 各维度均未进入前三；Q16 在16维第7名、8维第1名，显示单题不必随维度单调变化；Q17 在8维落到第27名，返回检索机制块未回答转置反序规则。这里的解释依据文本与排序，不据此断言模型内部具体丢失了哪个语义特征。

**Q05 的标注覆盖交叠：** CF-08 不仅说明一次线性组合，也明确给出“系数遍历全部实数时的所有结果叫线性张成”，并允许系数为零，因而能推得题目询问的集合及零向量成员关系。它与冻结 Gold CF-09 在本题上存在回答覆盖交叠。该例属于单目标标注的歧义线索，不能简单归为纯模型错误。当前 Gold、acceptable_ids、查询和全部指标未改动，也未将 CF-08 事后加入主指标；若未来修订数据，须另行核准并重新冻结。

三条概念相邻、三条术语不同和一条标注歧义仅描述这七个选例，不代表全库错误原因比例。没有证据的“块内容过宽”未强加到案例上。

### 验收记录

实际日志：

```text
saved results/figures/pca_variance.png dpi=200
saved results/figures/pca_comparison.png dpi=200
saved results/failure_cases.csv cases=7 unique_queries=7
```

7 例的原查询、Gold、目标名次/分数、Top-3 的 ID/分数及对照排名均与第七天完整排名逐项匹配；基线三条失败覆盖完整。两张新图及案例 CSV 从项目目录外独立重建，三文件与正式产物逐字节一致。四张 PNG 的 dpi 元数据为 199.9996（200 dpi），中文字与图中标签已视觉检查，修正了累计方差图的标注重叠。

开工前记录的 30 个代码/数据/结果文件中，只有两个计划内程序 pca_analysis.py、evaluate.py 增加绘图与案例功能；其余 28 个保持字节一致。数据库完整性 ok，两项冻结值保持，全部第七天数值结果及 Gram 两图未重写。未改变数据版本，也未根据结果修改查询或 Gold。

两脚本的默认路径也在临时输出目录完整运行，旧 PCA/指标 CSV、新图和案例均与正式产物逐字节一致。README 本地链接与 git diff --check 检查通过。论文打包校验通过，本机 Word 导出21页，封面无页码、正文1–20；21条原有可编辑公式完全保留，原样式、分节、页脚及图片字节不变，新嵌入图与正式PNG一致。逐页预览及新增页面检查通过，七例表与表题完整同在物理第18页。正式DOCX已替换为检查通过的版本。本轮未提交或推送Git。

最后由一名子代理只读复核通过，未发现阻塞第八天交付的问题。7例全部字段和文本分析、Q05覆盖交叠、图表数值及200 dpi均核对一致；独立执行两条重建命令，三份产物逐字节复现，未生成或重写数值CSV。论文21页、21条公式、四张内嵌图、表题同页及阶段范围复核通过；28个原有文件与冻结值保持，HEAD仍为第二次修改（D4），暂存区为空。本轮临时解包、PDF、页面预览、独立重建及复核输出已清理，等待用户验收后进入第九天。

## 后续实施

| 阶段 | 程序或文件 | 目标 |
| --- | --- | --- |
| 第 9–10 天 | `report/course_paper.docx` | 论文主体、引用核对及排版 |

检索实验、PCA 分析及全部图表/案例的重建命令见第五至八天说明。第八天完成后等待用户验收，再进入第九天。课程教材书名、作者、版次及封面个人信息尚待用户补充。
