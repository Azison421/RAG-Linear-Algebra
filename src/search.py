"""NumPy 精确检索；输入一句话，返回 Top-K 知识块正文与来源。"""

import argparse
from contextlib import closing
import logging
from pathlib import Path
import sqlite3

import numpy as np

from common import (
    DEFAULT_CONFIG, ROOT, encode_texts, load_config, load_encoder,
    read_vector_corpus, verify_frozen_data,
)


def rank_scores(scores, chunk_ids):
    """分数降序，完全同分时按块 ID 升序；保留完整排序供后续 MRR 使用。"""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(chunk_ids) or not np.isfinite(scores).all():
        raise ValueError("得分矩阵须为有限的 查询数×文档数 矩阵")
    return np.lexsort((np.broadcast_to(np.asarray(chunk_ids), scores.shape), -scores), axis=1)


def exact_search(query_vectors, document_vectors, chunk_ids):
    """通过 Q @ X.T 一次计算全部查询对文档的得分与完整名次。"""
    query_vectors = np.asarray(query_vectors)
    document_vectors = np.asarray(document_vectors)
    if query_vectors.ndim != 2 or document_vectors.ndim != 2 or query_vectors.shape[1] != document_vectors.shape[1]:
        raise ValueError("查询与文档必须为相同列数的二维矩阵")
    scores = query_vectors @ document_vectors.T
    return scores, rank_scores(scores, chunk_ids)


def search(query, config, database_path, queries_path, top_k=None):
    top_k = config["top_k"] if top_k is None else top_k
    if not query.strip():
        raise ValueError("查询不能为空")
    if not 1 <= top_k <= config["n_chunks"]:
        raise ValueError(f"top-k 必须在 1～{config['n_chunks']} 之间")
    with closing(sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        verify_frozen_data(connection, config, queries_path)
        records, vectors = read_vector_corpus(connection, config)
    model, _ = load_encoder(config)
    query_vectors = encode_texts(model, [query], config, query=True, normalize=True)
    scores, order = exact_search(query_vectors, vectors, [row["chunk_id"] for row in records])
    results = []
    for rank, index in enumerate(order[0, :top_k], 1):
        row = {key: value for key, value in records[index].items() if key != "embedding"}
        results.append({"rank": rank, "score": float(scores[0, index]), **row})
    logging.info("得分矩阵 shape=%s，返回 Top-%d", scores.shape, top_k)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=ROOT / "database" / "knowledge.db")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.csv")
    parser.add_argument("--query", required=True, help="待检索的一句话；程序自动添加固定查询指令")
    parser.add_argument("--top-k", type=int, default=None, help="默认读取配置 top_k=3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if not args.query.strip() or (args.top_k is not None and not 1 <= args.top_k <= config["n_chunks"]):
        parser.error(f"query 不能为空，top-k 必须在 1～{config['n_chunks']} 之间")
    results = search(args.query, config, args.database, args.queries, args.top_k)
    for result in results:
        print(f"\n{result['rank']}. {result['chunk_id']} {result['title']} | score={result['score']:.6f}")
        print(result["content"])
        print(f"来源：{result['source_author']} ({result['source_year']}). {result['source_title']}")
        print(result["source_url"])


if __name__ == "__main__":
    main()
