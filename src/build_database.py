"""按 V3.1 建立 60 块目录；全部数据获用户核准后可显式冻结。"""

import argparse
from collections import Counter
from contextlib import closing
import csv
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import sqlite3

import yaml

from common import DEFAULT_CONFIG, ROOT, corpus_text_sha256, load_config


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
    with closing(sqlite3.connect(database_path)) as connection, connection:
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


def freeze_dataset(database_path, queries_path, config_path):
    """核验完整数据并写入冻结哈希；调用前须由用户核准正文和查询。"""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    queries_path = Path(queries_path)
    with closing(sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT chunk_id, content, source_title, source_author, source_year, source_url "
            "FROM chunks ORDER BY chunk_id"
        ).fetchall()
        if len(rows) != config["n_chunks"] or any(
            not row[1].strip() or any(value is None or value == "" for value in row[2:])
            for row in rows
        ):
            raise ValueError("全部知识正文及来源尚未齐全，不能冻结")
        with queries_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = ["query_id", "query_text", "query_type", "gold_chunk_id", "acceptable_ids", "notes"]
            if reader.fieldnames != fields:
                raise ValueError("查询 CSV 表头与方案不一致")
            queries = list(reader)
        if len(queries) != config["n_queries"] or Counter(
            query["query_type"] for query in queries
        ) != {"contrast": 15, "paraphrase": 5, "formula": 5}:
            raise ValueError("查询必须为 25 条，类型数量为 15/5/5")
        ids = {row[0] for row in rows}
        query_ids, query_texts, pair_numbers = set(), set(), []
        for query in queries:
            if None in query or any(value is None for value in query.values()):
                raise ValueError("查询 CSV 存在缺列或多余列")
            if not all(query[field].strip() for field in fields if field != "acceptable_ids"):
                raise ValueError("查询的必填字段不能留空")
            if query["notes"].startswith("待核"):
                raise ValueError("查询仍标为待核，须经用户确认后再冻结")
            if query["query_id"] in query_ids or query["query_text"] in query_texts:
                raise ValueError("查询 ID 和问题正文均须唯一")
            if query["gold_chunk_id"] not in ids:
                raise ValueError("查询的唯一 Gold 不存在于知识库")
            acceptable = json.loads(query["acceptable_ids"]) if query["acceptable_ids"] else []
            if not isinstance(acceptable, list) or any(not isinstance(item, str) or item not in ids for item in acceptable):
                raise ValueError("acceptable_ids 须为空或由有效块 ID 构成的 JSON 列表")
            if query["query_type"] == "contrast":
                matches = re.findall(r"\bP(0[1-9]|1[0-5])\b", query["notes"])
                if len(matches) != 1 or query["gold_chunk_id"] not in CONFUSABLE_PAIRS[int(matches[0]) - 1]:
                    raise ValueError("概念区分题的 notes 须注明一个 P01～P15 编号，Gold 须属于该对")
                pair_numbers.append(int(matches[0]))
            query_ids.add(query["query_id"])
            query_texts.add(query["query_text"])
        if sorted(pair_numbers) != list(range(1, 16)):
            raise ValueError("概念区分查询必须完整覆盖 P01～P15，每对一题")
        hashes = (corpus_text_sha256(connection), sha256(queries_path.read_bytes()).hexdigest())
    old_hashes = (config["corpus_sha256"], config["queries_sha256"])
    if old_hashes != (None, None) and old_hashes != hashes:
        raise ValueError("数据与既有冻结哈希不一致，须先按 CHANGELOG 规则记录变更，不能直接覆盖")
    if old_hashes != hashes:
        config["corpus_sha256"], config["queries_sha256"] = hashes
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    logging.info("corpus_sha256=%s queries_sha256=%s", *hashes)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=ROOT / "database" / "knowledge.db")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.csv")
    parser.add_argument("--freeze", action="store_true", help="仅在用户核准全部正文和查询后执行，写入配置中的冻结哈希")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    counts = build_database(args.database)
    if sum(counts.values()) != config["n_chunks"]:
        raise ValueError("数据库目录总数与配置不一致")
    if args.freeze:
        freeze_dataset(args.database, args.queries, args.config)


if __name__ == "__main__":
    main()
