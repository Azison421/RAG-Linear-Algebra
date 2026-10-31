# RAG 向量检索线性代数大作业

依据上一级目录的 `RAG_向量检索线性代数大作业_V3.1.md` 实施。第四天已完成：全部 60 块正文及来源获用户核准并入库，25 条查询与唯一 Gold 已获用户核准，两项数据哈希已写入配置。当前数据库为 60 条目录、60 条正文、0 条向量，正式数据已冻结；第五天的正式编码与检索尚未开始。环境检查结果见 `results/environment-check.json`。

## 使用环境

在本目录打开 PowerShell，运行：

```powershell
.\.venv\Scripts\python.exe .\verify_environment.py
```

运行其他项目程序时，也使用 `.venv\Scripts\python.exe`。VS Code 的 Python 解释器选择同一路径。

模型从上一级目录的 `模型\bge-small-zh-v1.5` 读取。验证脚本仅使用本地模型文件。

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

上述检查使用三条示例文本，用于验证环境。正式的 60 条知识文本、25 个查询及 PCA 32/16/8 维课程实验尚未开展。

## 第一天产物

- [论文框架](report/course_paper.docx)：沿用老师原封面，A4，正文四号宋体、1.5 倍行距；封面单独一节，后续从 1 编号。Word 导出预览共 6 页，封面 1 页、正文框架 5 页。摘要、指标、结论和参考文献保留待补位置。
- [首批 14 块的核准记录](report/CF-01_CF-27_核准记录.md)：第一天起草 CF-01～CF-14，正文 154–209 个汉字。第三天获用户核准后已入库，原待核稿中的正文已移除。
- [数据库](database/knowledge.db)：只有 `chunks` 表；第一天导入 60 条 ID、分组、主题、标题和对偶关系，当时正文与向量均为 0 条。当前正文数量见本页开头；`content=''` 仍表示仅有目录。
- [配置](experiment_config.yaml)：已填模型路径、实际权重哈希、512 维、固定查询指令、编码参数、归一化、种子与 PCA 维度。`n_chunks=60`、`n_queries=25` 为计划数量。数据冻结哈希和未确认的模型下载日期保留为 NULL。
- [模长诊断](results/embedding_norms.csv)：`run_scope=smoke`，含三条示例和一条汇总；第 5 天生成正式向量时更新为 60 块诊断。

## 项目结构

```text
RAG-Linear-Algebra/
├── database/
│   └── knowledge.db
├── data/
│   └── queries.csv                  # 25 条已核准查询与唯一 Gold，已冻结
├── src/
│   ├── common.py                    # 已实现配置与 Float32 BLOB 转换
│   ├── build_database.py            # 已实现单表及 60 块目录导入
│   ├── embed.py                     # 第 5 天接口骨架
│   ├── search.py                    # 第 5 天接口骨架
│   ├── gram_analysis.py             # 第 6 天接口骨架
│   ├── pca_analysis.py              # 第 7 天接口骨架
│   └── evaluate.py                  # 第 7 天接口骨架
├── experiments/
│   ├── math_verification.py         # 第 6 天接口骨架
│   └── run_retrieval.py             # 第 7 天接口骨架
├── results/
│   ├── environment-check.json
│   ├── embedding_norms.csv          # 当前仅为 smoke
│   └── figures/
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

本地还有 `.venv`、`.runtime-cache` 和前述依赖安装、核对记录。后续模块已预留文件及 `--help` 接口，直接执行会记录“接口骨架”并以状态码 2 退出，正式功能待对应日期实现。正式实验 CSV 和四张 200 dpi PNG 图按方案第 11.3 节生成。

## 配置与模型版本

`model_dir` 相对于配置文件所在目录解析，当前为 `../模型/bge-small-zh-v1.5`。验证脚本设置离线模式，以 `local_files_only=True` 加载，并实际计算权重哈希与配置比较。

```text
354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026
```

固定查询指令为 `为这个句子生成表示以用于检索相关文章：`。文档不加指令，文档和查询显式开启归一化。环境结果中保存模型模块、tokenizer、编码参数、设备与依赖版本；本地模型含 Normalize，与原始模长诊断一致。

随机种子为 20261031，PCA 维度为 `[59, 32, 16, 8]`。正文、查询及 Gold 核准并冻结后才填写数据哈希。PCA-59 按方案第 6.6 节验证中心化文档内积的保留，不预设与未中心化基线同排序。

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
- [LA、GR 核准记录](report/LA-GR_核准记录.md) 与数据库：新增 23 块已按用户确认原样入库，50 块已核准正文为 SQLite 唯一正式数据源。
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

当前只读复查数据库：

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; c=sqlite3.connect('file:database/knowledge.db?mode=ro',uri=True); print(c.execute('SELECT COUNT(*), SUM(length(content)>0), COUNT(embedding) FROM chunks').fetchone()); print(c.execute('PRAGMA integrity_check').fetchone()[0])"
git diff --check
```

实际输出为 `(60, 60, 0)`、`ok`。正式冻结值已写入 `experiment_config.yaml`：

```text
corpus_sha256: f9c6effe0297fb76aca84435a37a39b8a48ff2f567982778cd852070db3e4569
queries_sha256: d09ec555e741b246a30a2915a0bf8fb4b4382dece4507d640229f1d7c4e53293
```

## 后续实施

| 阶段 | 程序或文件 | 目标 |
| --- | --- | --- |
| 第 5 天 | `src/embed.py`、`src/search.py` | 正式模长诊断、向量入库、命令行检索 |
| 第 6 天 | `experiments/math_verification.py`、`src/gram_analysis.py` | 实验 A/C，数学验证与 Gram 图表 |
| 第 7 天 | `src/pca_analysis.py`、`src/evaluate.py`、`experiments/run_retrieval.py` | 实验 B，SVD 与对拍、PCA 扫描、分层指标 |
| 第 8 天 | `results/`、`results/figures/` | 图表与至少五个失败案例 |
| 第 9–10 天 | `report/course_paper.docx` | 论文主体、引用核对及排版 |

向量生成、检索实验、Gram/PCA 分析及图表重建的完整运行参数，在相应模块实现后补充。目前后续骨架文件的 `--help` 只说明计划职责。完整 60 块与 25 条查询已冻结，进入第五天前继续等待用户确认。课程教材书名、作者、版次及封面个人信息尚待用户补充。
