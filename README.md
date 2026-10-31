# RAG 向量检索线性代数大作业

依据上一级目录的 `RAG_向量检索线性代数大作业_V3.1.md` 实施。当前完成第 1 天：环境验证、项目骨架、配置、60 块目录、首批 14 块待核草稿及 Word 论文框架。完整环境检查结果见 `results/environment-check.json`。

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
- [CF-01 至 CF-14 待核稿](report/CF-01_CF-14_待核.md)：14 块，每块正文 154–209 个汉字，包含定义、公式、对偶区别和来源。状态和来源元数据不计入正文汉字数。
- [数据库](database/knowledge.db)：只有 `chunks` 表，含 60 条 ID、分组、主题、标题和对偶关系。`content=''` 表示仅有目录；来源、编码字段为 NULL。正文已入库 0 条、向量已入库 0 条。
- [配置](experiment_config.yaml)：已填模型路径、实际权重哈希、512 维、固定查询指令、编码参数、归一化、种子与 PCA 维度。`n_chunks=60`、`n_queries=25` 为计划数量。数据冻结哈希和未确认的模型下载日期保留为 NULL。
- [模长诊断](results/embedding_norms.csv)：`run_scope=smoke`，含三条示例和一条汇总；第 5 天生成正式向量时更新为 60 块诊断。

## 项目结构

```text
RAG-Linear-Algebra/
├── database/
│   └── knowledge.db
├── data/
│   └── queries.csv                  # 目前只有方案规定的表头
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
│   └── CF-01_CF-14_待核.md
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

待核正文位于 `report/CF-01_CF-14_待核.md`，用户逐块确认后才导入。经核准写库后，知识正文只以 SQLite 为准，审阅稿应移除已入库正文，保留必要审阅记录。没有另建 `corpus.jsonl` 或 `judgments.csv`；查询及唯一 Gold 统一保存在 `data/queries.csv`。

主要来源为方案文献 [9] Strang 第五版，官方书目及目录已核对，具体内容仍待用户对照教材核对。CF-04 补充 scikit-learn `cosine_distances` 官方定义链接并标注待用户核对，尚未列入论文正式参考文献。

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

## 后续实施

| 阶段 | 程序或文件 | 目标 |
| --- | --- | --- |
| 第 2–3 天 | 待核稿与 `knowledge.db` | 补全 CF、LA、GR，逐块核对正文 |
| 第 4 天 | `knowledge.db`、`data/queries.csv`、配置 | 完成 RG、25 条查询与唯一 Gold，核准并冻结哈希 |
| 第 5 天 | `src/embed.py`、`src/search.py` | 正式模长诊断、向量入库、命令行检索 |
| 第 6 天 | `experiments/math_verification.py`、`src/gram_analysis.py` | 实验 A/C，数学验证与 Gram 图表 |
| 第 7 天 | `src/pca_analysis.py`、`src/evaluate.py`、`experiments/run_retrieval.py` | 实验 B，SVD 与对拍、PCA 扫描、分层指标 |
| 第 8 天 | `results/`、`results/figures/` | 图表与至少五个失败案例 |
| 第 9–10 天 | `report/course_paper.docx` | 论文主体、引用核对及排版 |

向量生成、检索实验、Gram/PCA 分析及图表重建的完整运行参数，在相应模块实现后补充。目前骨架文件的 `--help` 只说明计划职责。课程教材书名、作者、版次及封面个人信息尚待用户补充；首批 14 块正文和来源须由用户核准后入库。
