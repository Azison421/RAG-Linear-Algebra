"""实验 B：固定 60 文档与 25 查询，扫描 Baseline / PCA-59/32/16/8。"""

import argparse
import csv
from datetime import datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import DEFAULT_CONFIG, encode_texts, load_config, load_encoder
from evaluate import evaluate_rankings, load_queries, save_metrics
from pca_analysis import fit_pca, normalize_rows, read_documents, save_pca
from search import exact_search, rank_scores


def metric_equivalence(queries, documents, ids):
    inner = queries @ documents.T
    cosine = inner / (np.linalg.norm(queries, axis=1)[:, None] * np.linalg.norm(documents, axis=1)[None, :])
    distances = np.linalg.norm(queries[:, None, :] - documents[None, :, :], axis=2)
    order = rank_scores(inner, ids)
    cosine_equal = np.all(order == rank_scores(cosine, ids), axis=1)
    l2_equal = np.all(order == rank_scores(-distances, ids), axis=1)
    return {
        "cosine_inner_full_ranking_agreement_count": int(cosine_equal.sum()),
        "l2_inner_full_ranking_agreement_count": int(l2_equal.sum()),
        "n_queries": len(queries), "full_ranking_agreement_rate": float(np.mean(cosine_equal & l2_equal)),
        "unit_distance_identity_max_abs_error": float(np.max(np.abs(distances ** 2 - (2 - 2 * inner)))),
    }


def run(config, database, queries_path, output_dir):
    output_dir = Path(output_dir)
    records, documents, hashes = read_documents(config, database, queries_path)
    ids = [row["chunk_id"] for row in records]
    queries = load_queries(queries_path, config, ids)
    model, encoder_info = load_encoder(config)
    encoded = encode_texts(model, [q["query_text"] for q in queries], config, query=True, normalize=True)
    query_vectors = encoded.astype(np.float64)
    mean, vt, pca_rows, pca_summary = fit_pca(documents, config["pca_dims"], config["random_seed"])
    save_pca(output_dir, pca_rows, pca_summary, config, hashes, ids)
    # 基线精确计算余弦；PCA 拟合前仅转换精度，不改变库内文档值。
    representations = {"Baseline": (normalize_rows(query_vectors), normalize_rows(documents))}
    projection_query_norms = {}
    for r in config["pca_dims"]:
        q = (query_vectors - mean) @ vt[:r].T
        x = (documents - mean) @ vt[:r].T
        projection_query_norms[f"PCA-{r}"] = float(np.linalg.norm(q, axis=1).min())
        representations[f"PCA-{r}"] = (normalize_rows(q), normalize_rows(x))
    orders, score_matrices, equivalence = {}, {}, {}
    for method, (q, x) in representations.items():
        score_matrices[method], orders[method] = exact_search(q, x, ids)
        equivalence[method] = metric_equivalence(q, x, ids)
        if equivalence[method]["full_ranking_agreement_rate"] != 1.0:
            raise ValueError(f"{method} 单位化后真实度量排序不一致，请检查数值")
    with (output_dir / "retrieval_rankings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "dimension", "query_id", "rank", "chunk_id", "score"])
        writer.writeheader()
        for method, order in orders.items():
            dimension = representations[method][0].shape[1]
            for i, query in enumerate(queries):
                for rank, j in enumerate(order[i], 1):
                    writer.writerow({"method": method, "dimension": dimension, "query_id": query["query_id"],
                                     "rank": rank, "chunk_id": ids[j], "score": float(score_matrices[method][i, j])})
    np.save(output_dir / "query_embeddings.npy", encoded, allow_pickle=False)
    metrics, details = evaluate_rankings(orders, queries, ids, pca_rows, config["embedding_dim"])
    save_metrics(output_dir, metrics, details)
    run_info = {
        "run_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **hashes,
        "random_seed": config["random_seed"], "encoder": encoder_info,
        "chunk_ids": ids, "query_ids": [q["query_id"] for q in queries],
        "query_embeddings_shape": list(encoded.shape), "query_embeddings_dtype": str(encoded.dtype),
        "query_unit_norm_max_abs_error": float(np.max(np.abs(np.linalg.norm(query_vectors, axis=1) - 1))),
        "computation_dtype": "float64", "baseline": "row-normalized float64 copies; no centering",
        "pca_input": "original stored/encoded float32 values cast to float64; document mean only",
        "tie_rule": "score descending, chunk_id ascending", "mrr_candidates": len(ids),
        "metric_equivalence": equivalence, "minimum_query_projection_norms": projection_query_norms,
        "pca_59_vs_baseline_full_ranking_agreement_count": int(np.all(orders["PCA-59"] == orders["Baseline"], axis=1).sum()),
        "result_sha256": {name: sha256((output_dir / name).read_bytes()).hexdigest() for name in
                          ("query_embeddings.npy", "retrieval_rankings.csv", "pca_results.csv")},
    }
    (output_dir / "retrieval_run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("saved rankings=%d metrics=%d per_query=%d", len(queries) * len(ids) * len(orders), len(metrics), len(details))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=ROOT / "database" / "knowledge.db")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config), args.database, args.queries, args.output_dir)


if __name__ == "__main__":
    main()
