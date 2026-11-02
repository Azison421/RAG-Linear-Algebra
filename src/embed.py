"""本地 BGE 模长诊断、正式归一化向量入库。"""

import argparse
from contextlib import closing
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np

from common import (
    DEFAULT_CONFIG, ROOT, check_vectors, encode_texts, load_config,
    load_encoder, read_vector_corpus, vector_to_blob, verify_frozen_data,
)


def embed_corpus(config, database_path, queries_path, output_dir):
    database_path = Path(database_path).resolve()
    with closing(sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)) as connection:
        hashes = verify_frozen_data(connection, config, queries_path)
        rows = connection.execute("SELECT chunk_id, content FROM chunks ORDER BY chunk_id").fetchall()
    documents = [row[1] for row in rows]
    model, run_info = load_encoder(config)
    raw = encode_texts(model, documents, config, normalize=False)
    raw_norms = check_vectors(raw, len(rows), config["embedding_dim"], unit=False)
    vectors = encode_texts(model, documents, config, normalize=True)
    norms = check_vectors(vectors, len(rows), config["embedding_dim"])
    logging.info("raw norm min/mean/max: %.12f / %.12f / %.12f", raw_norms.min(), raw_norms.mean(), raw_norms.max())
    logging.info("正式向量 shape=%s dtype=%s max_norm_error=%.3g", vectors.shape, vectors.dtype, np.max(np.abs(norms - 1)))
    updates = [
        (config["model_name"], run_info["model_revision"], config["embedding_dim"],
         float(raw_norms[i]), vector_to_blob(vectors[i], config["embedding_dim"]), row[0])
        for i, row in enumerate(rows)
    ]
    with closing(sqlite3.connect(database_path.as_uri() + "?mode=rw", uri=True)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        verify_frozen_data(connection, config, queries_path)
        connection.executemany(
            "UPDATE chunks SET model_name=?, model_revision=?, embedding_dim=?, raw_norm=?, embedding=? WHERE chunk_id=?",
            updates,
        )
        stored_records, stored_vectors = read_vector_corpus(connection, config)
        if [row["chunk_id"] for row in stored_records] != [row[0] for row in rows] or not np.array_equal(stored_vectors, vectors):
            raise ValueError("写入数据库后的向量与编码输出不一致")
        verify_frozen_data(connection, config, queries_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["run_scope", "record_type", "text_id", "raw_norm", "normalized_norm",
              "raw_norm_min", "raw_norm_mean", "raw_norm_max",
              "normalized_norm_min", "normalized_norm_mean", "normalized_norm_max"]
    with (output_dir / "embedding_norms.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({"run_scope": "corpus", "record_type": "sample", "text_id": row[0],
                         "raw_norm": float(raw_norms[i]), "normalized_norm": float(norms[i])}
                        for i, row in enumerate(rows))
        writer.writerow({"run_scope": "corpus", "record_type": "summary",
                         "raw_norm_min": float(raw_norms.min()), "raw_norm_mean": float(raw_norms.mean()),
                         "raw_norm_max": float(raw_norms.max()), "normalized_norm_min": float(norms.min()),
                         "normalized_norm_mean": float(norms.mean()), "normalized_norm_max": float(norms.max())})
    run_info.update({
        "encoded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "run_scope": "corpus",
        "encoded_field": "chunks.content",
        "document_order": [row[0] for row in rows],
        "document_vector_shape": list(vectors.shape),
        "dtype": str(vectors.dtype),
        "vector_bytes": int(vectors.nbytes),
        "raw_norm_summary": {"min": float(raw_norms.min()), "mean": float(raw_norms.mean()), "max": float(raw_norms.max())},
        "normalized_norm_max_error": float(np.max(np.abs(norms - 1))),
        **hashes,
    })
    (output_dir / "embedding_run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logging.info("已写入 %d 条向量；模长记录及运行参数：%s", len(rows), output_dir)
    return run_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=ROOT / "database" / "knowledge.db")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    embed_corpus(load_config(args.config), args.database, args.queries, args.output_dir)


if __name__ == "__main__":
    main()
