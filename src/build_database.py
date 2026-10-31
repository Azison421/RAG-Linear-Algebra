"""按 V3.1 建立单表数据库，只导入 60 块目录，不导入待核正文。"""

import argparse
import json
import logging
from pathlib import Path
import sqlite3

from common import DEFAULT_CONFIG, ROOT, load_config


CATALOG = {
    "CF": (
        "易混淆概念",
        [
            "向量内积", "余弦相似度", "欧氏距离", "余弦距离", "L2 范数",
            "向量正交", "线性无关", "线性组合", "线性张成", "矩阵阶数",
            "矩阵的秩", "零空间维数（零度）", "特征值", "奇异值", "特征向量",
            "奇异向量", "对称矩阵", "半正定矩阵", "Gram 矩阵", "协方差矩阵",
            "特征值分解", "奇异值分解（SVD）", "主成分分析（PCA）",
            "一般线性投影", "正交投影", "向量归一化", "降维",
        ],
    ),
    "LA": (
        "线性代数基础",
        [
            "实数向量空间 R^d", "向量的坐标表示", "基与维数", "矩阵与线性变换",
            "矩阵乘法及其几何意义", "转置矩阵", "Cauchy–Schwarz 不等式",
            "三角不等式", "向量夹角与单位球面", "行列式", "逆矩阵",
            "线性方程组解的结构", "相似与对角化", "二次型与正定矩阵",
            "Gram–Schmidt 正交化",
        ],
    ),
    "GR": (
        "Gram 矩阵与 PCA 理论",
        [
            "文档向量矩阵 X 的构造", "Gram 矩阵的半正定性",
            "rank(G)=rank(X)（零空间证明）", "实对称矩阵的谱定理",
            "Gram 矩阵特征值与奇异值平方", "数据中心化与中心化矩阵的秩上界",
            "主成分解释方差比", "低秩近似与 Eckart–Young 定理",
        ],
    ),
    "RG": (
        "RAG 与语义检索",
        [
            "RAG 的基本工作流程", "文本分块的目的与方法", "Embedding 模型的作用",
            "稠密向量检索", "BGE 模型的工作方式", "查询指令与文档编码",
            "Top-K 检索机制", "精确检索与矩阵乘法批量检索",
            "Hit@K 与 MRR 指标", "语义检索的局限性与误差来源",
        ],
    ),
}

CONFUSABLE_PAIRS = [
    ("CF-01", "CF-02"), ("CF-03", "CF-04"), ("CF-05", "CF-03"),
    ("CF-06", "CF-07"), ("CF-08", "CF-09"), ("CF-10", "CF-11"),
    ("CF-11", "CF-12"), ("CF-13", "CF-14"), ("CF-15", "CF-16"),
    ("CF-17", "CF-18"), ("CF-19", "CF-20"), ("CF-21", "CF-22"),
    ("CF-22", "CF-23"), ("CF-24", "CF-25"), ("CF-26", "CF-27"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    group_code      TEXT NOT NULL,
    topic           TEXT NOT NULL,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    confusable_with TEXT,
    source_title    TEXT,
    source_author   TEXT,
    source_year     INTEGER,
    source_url      TEXT,
    model_name      TEXT,
    model_revision  TEXT,
    embedding_dim   INTEGER,
    raw_norm        REAL,
    embedding       BLOB
);
"""


def build_database(database_path):
    rows = []
    for group, (topic, titles) in CATALOG.items():
        for number, title in enumerate(titles, 1):
            chunk_id = f"{group}-{number:02d}"
            related = sorted(
                b if a == chunk_id else a
                for a, b in CONFUSABLE_PAIRS if chunk_id in (a, b)
            )
            rows.append((chunk_id, group, topic, title, "", json.dumps(related)))
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(SCHEMA)
        connection.executemany(
            """INSERT INTO chunks
               (chunk_id, group_code, topic, title, content, confusable_with)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(chunk_id) DO NOTHING""",
            rows,
        )
        counts = dict(connection.execute(
            "SELECT group_code, COUNT(*) FROM chunks GROUP BY group_code"
        ))
        text_count, vector_count = connection.execute(
            "SELECT SUM(content != ''), COUNT(embedding) FROM chunks"
        ).fetchone()
    logging.info("database=%s groups=%s total=%d", database_path, counts, sum(counts.values()))
    logging.info("正文已入库=%d 向量已入库=%d；空正文表示仅有目录，尚未核准", text_count, vector_count)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=ROOT / "database" / "knowledge.db")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    counts = build_database(args.database)
    if sum(counts.values()) != config["n_chunks"]:
        raise ValueError("数据库目录总数与配置不一致")


if __name__ == "__main__":
    main()
