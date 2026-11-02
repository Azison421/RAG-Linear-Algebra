"""实验 A1–A5：用 float64 人工向量验证度量、独立缩放和正交不变性。"""

import argparse
import csv
import logging
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import DEFAULT_CONFIG, load_config


TRIALS = 1000
DIMENSION = 512
N_DOCUMENTS = 60


def metric_scores(query, documents):
    """直接计算三种度量，L2 不借用待验证的恒等式。"""
    inner = documents @ query
    cosine = inner / (np.linalg.norm(documents, axis=1) * np.linalg.norm(query))
    distance = np.linalg.norm(documents - query, axis=1)
    return inner, cosine, distance


def rankings(scores):
    inner, cosine, distance = scores
    return (np.argsort(-inner, kind="stable"), np.argsort(-cosine, kind="stable"),
            np.argsort(distance, kind="stable"))


def run_verification(seed):
    rng = np.random.default_rng(seed)
    rows = []

    def record(experiment, metric, value, passed):
        rows.append({"实验": experiment, "指标": metric, "数值": float(value), "是否通过": bool(passed)})

    query = np.array([1.0, 0.0], dtype=np.float64)
    documents = np.array([[0.2, 0.0], [np.sqrt(3) / 2, 0.5]], dtype=np.float64)
    scores = metric_scores(query, documents)
    expected = ([0.2, np.sqrt(3) / 2], [1.0, np.sqrt(3) / 2], [0.8, np.sqrt(2 - np.sqrt(3))])
    for name, actual, reference in zip(("inner_product", "cosine", "l2"), scores, expected):
        for i, (value, target) in enumerate(zip(actual, reference), 1):
            record("A1", f"{name}_document_{i}", value, abs(value - target) < 1e-12)
    match = all(np.array_equal(order, target) for order, target in
                zip(rankings(scores), ([1, 0], [0, 1], [1, 0])))
    record("A1", "all_three_rankings_match_hand_example", int(match), match)

    left = rng.normal(size=(TRIALS, DIMENSION))
    right = rng.normal(size=(TRIALS, DIMENSION))
    left /= np.linalg.norm(left, axis=1, keepdims=True)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    error = float(np.max(np.abs(np.sum((left - right) ** 2, axis=1) -
                                (2 - 2 * np.sum(left * right, axis=1)))))
    record("A2", "pairs", TRIALS, True)
    record("A2", "identity_max_abs_error", error, error < 1e-12)

    agreements = 0
    query_norms = []
    document_norm_error = 0.0
    for _ in range(TRIALS):
        documents = rng.normal(size=(N_DOCUMENTS, DIMENSION))
        documents *= 3.0 / np.linalg.norm(documents, axis=1, keepdims=True)
        query = rng.normal(size=DIMENSION)
        query_norms.append(float(np.linalg.norm(query)))
        document_norm_error = max(document_norm_error, float(np.max(np.abs(np.linalg.norm(documents, axis=1) - 3))))
        inner_order, cosine_order, l2_order = rankings(metric_scores(query, documents))
        agreements += np.array_equal(inner_order, cosine_order) and np.array_equal(inner_order, l2_order)
    record("A3", "trials", TRIALS, True)
    record("A3", "full_ranking_agreement_rate", agreements / TRIALS, agreements == TRIALS)
    record("A3", "document_norm_max_error_from_3", document_norm_error, document_norm_error < 1e-12)
    record("A3", "query_norm_min", min(query_norms), min(query_norms) > 0)
    record("A3", "query_norm_max", max(query_norms), not np.allclose(query_norms, 1))

    top1, overlaps, taus = [], [], []
    for _ in range(TRIALS):
        documents = rng.normal(size=(N_DOCUMENTS, DIMENSION))
        documents /= np.linalg.norm(documents, axis=1, keepdims=True)
        query = rng.normal(size=DIMENSION)
        query /= np.linalg.norm(query)
        scales = rng.uniform(0.5, 2.0, size=N_DOCUMENTS)
        scaled = documents * scales[:, None]
        _, cosine_order, l2_order = rankings(metric_scores(query, scaled))
        top1.append(cosine_order[0] == l2_order[0])
        overlaps.append(len(set(cosine_order[:3]) & set(l2_order[:3])) / 3)
        # 同一文档的两种名次逐项对应，不能直接相关两条文档 ID 排列。
        taus.append(float(kendalltau(np.argsort(cosine_order), np.argsort(l2_order)).statistic))
    record("A4", "trials", TRIALS, True)
    record("A4", "top1_agreement_rate", np.mean(top1), True)
    record("A4", "mean_top3_overlap", np.mean(overlaps), True)
    record("A4", "mean_kendall_tau", np.mean(taus), np.isfinite(taus).all())

    rotation, _ = np.linalg.qr(rng.normal(size=(DIMENSION, DIMENSION)))
    documents = rng.normal(size=(N_DOCUMENTS, DIMENSION))
    query = rng.normal(size=DIMENSION)
    original = metric_scores(query, documents)
    transformed = metric_scores(query @ rotation, documents @ rotation)
    for name, old, new, old_order, new_order in zip(
        ("inner_product", "cosine", "l2"), original, transformed,
        rankings(original), rankings(transformed),
    ):
        error = float(np.max(np.abs(old - new)))
        record("A5", f"{name}_max_abs_error", error, error < 1e-10)
        record("A5", f"{name}_full_ranking_unchanged", int(np.array_equal(old_order, new_order)),
               np.array_equal(old_order, new_order))
    error = float(np.max(np.abs(rotation.T @ rotation - np.eye(DIMENSION))))
    record("A5", "orthogonality_max_abs_error", error, error < 1e-12)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    logging.info("实验 A：float64，seed=%d；A2/A3/A4 各 %d 次，随机文档=%d×%d",
                 config["random_seed"], TRIALS, N_DOCUMENTS, DIMENSION)
    rows = run_verification(config["random_seed"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "synthetic_results.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["实验", "指标", "数值", "是否通过"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for experiment in ("A1", "A2", "A3", "A4", "A5"):
        subset = [row for row in rows if row["实验"] == experiment]
        logging.info("%s: %d/%d 检查通过", experiment, sum(row["是否通过"] for row in subset), len(subset))
    for row in rows:
        if row["指标"] in ("identity_max_abs_error", "full_ranking_agreement_rate", "top1_agreement_rate",
                           "mean_top3_overlap", "mean_kendall_tau") or row["指标"].endswith("_max_abs_error"):
            logging.info("%s %s=%.12g", row["实验"], row["指标"], row["数值"])
    logging.info("结果：%s；A4 是人工缩放下的观测，不属于 BGE 检索指标", output)
    if not all(row["是否通过"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
