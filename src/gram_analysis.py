"""实验 C：从正式向量库只读计算 float64 Gram 矩阵、数值秩、谱与图表。"""

import argparse
from contextlib import closing
import csv
from datetime import datetime
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np

from build_database import CONFUSABLE_PAIRS
from common import DEFAULT_CONFIG, ROOT, load_config, read_vector_corpus, verify_frozen_data


def matrix_analysis(matrix):
    """使用 NumPy 默认阈值规则，同时核对 matrix_rank 与显式 SVD 计数。"""
    gram = matrix @ matrix.T
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    gram_singular_values = np.linalg.svd(gram, compute_uv=False)
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    squared_singular_values = np.zeros(len(gram), dtype=np.float64)
    squared_singular_values[:len(singular_values)] = singular_values ** 2
    epsilon = np.finfo(np.float64).eps
    matrix_tolerance = float(singular_values[0] * max(matrix.shape) * epsilon)
    gram_tolerance = float(gram_singular_values[0] * max(gram.shape) * epsilon)
    matrix_rank = int(np.linalg.matrix_rank(matrix))
    gram_rank = int(np.linalg.matrix_rank(gram))
    matrix_rank_svd = int(np.count_nonzero(singular_values > matrix_tolerance))
    gram_rank_svd = int(np.count_nonzero(gram_singular_values > gram_tolerance))
    symmetry_error = float(np.max(np.abs(gram - gram.T)))
    spectrum_error = float(np.max(np.abs(eigenvalues - squared_singular_values)))
    checks = {
        "symmetric": symmetry_error < 1e-12,
        "positive_semidefinite": bool(eigenvalues[-1] >= -1e-10),
        "matrix_rank_matches_svd": matrix_rank == matrix_rank_svd,
        "gram_rank_matches_svd": gram_rank == gram_rank_svd,
        "gram_rank_equals_matrix_rank": gram_rank == matrix_rank,
        "eigenvalues_equal_squared_singular_values": bool(np.allclose(
            eigenvalues, squared_singular_values, rtol=1e-8, atol=1e-10)),
    }
    summary = {
        "matrix_shape": list(matrix.shape), "gram_shape": list(gram.shape),
        "matrix_rank": matrix_rank, "matrix_rank_svd": matrix_rank_svd,
        "matrix_rank_tolerance": matrix_tolerance,
        "gram_rank": gram_rank, "gram_rank_svd": gram_rank_svd,
        "gram_rank_tolerance": gram_tolerance,
        "symmetry_max_abs_error": symmetry_error,
        "minimum_eigenvalue": float(eigenvalues[-1]),
        "maximum_eigenvalue": float(eigenvalues[0]),
        "spectrum_max_abs_error": spectrum_error,
        "checks": checks,
    }
    return gram, eigenvalues, singular_values, summary


def heatmap_order(records):
    """按 CF/LA/GR/RG 分组，CF-04、03、05 连排使两个共享对均相邻。"""
    cf = ["CF-01", "CF-02", "CF-04", "CF-03", "CF-05"]
    cf += [f"CF-{i:02d}" for i in range(6, 28)]
    ids = cf + [f"{group}-{i:02d}" for group, count in (("LA", 15), ("GR", 8), ("RG", 10))
                for i in range(1, count + 1)]
    indices = {row["chunk_id"]: i for i, row in enumerate(records)}
    if set(indices) != set(ids) or len(indices) != len(ids):
        raise ValueError("Gram 图需要方案固定的 60 个块 ID")
    return ids, [indices[chunk_id] for chunk_id in ids]


def plot_gram(gram, spectra, summary, records, figure_dir):
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".runtime-cache" / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.findfont(font_manager.FontProperties(family="Microsoft YaHei"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    figure_dir.mkdir(parents=True, exist_ok=True)
    ids, order = heatmap_order(records)
    shown = gram[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = ax.imshow(shown, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=9)
    ax.set_yticks(range(len(ids)), ids, fontsize=9)
    for boundary in (26.5, 41.5, 49.5):
        ax.axhline(boundary, color="white", linewidth=0.8)
        ax.axvline(boundary, color="white", linewidth=0.8)
    for group, center in (("CF", 13), ("LA", 34), ("GR", 45.5), ("RG", 54.5)):
        ax.text(center, -1.6, group, ha="center", va="bottom", fontsize=10)
    ax.set_title("60 个知识块的原始单位向量 Gram 矩阵", pad=28)
    ax.set_xlabel("知识块 ID（CF 内共享对相邻）")
    ax.set_ylabel("知识块 ID")
    fig.colorbar(image, ax=ax, shrink=0.78, label="单位向量内积（余弦相似度）")
    fig.savefig(figure_dir / "gram_heatmap.png", dpi=200)
    plt.close(fig)

    floor = summary["spectrum_plot_floor"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, label, color in (("G", "G：原始向量", "#215b9a"), ("Gc", "Gc：中心化向量", "#c65d21")):
        eigenvalues = spectra[name]
        rank = summary[name]["gram_rank"]
        ax.plot(np.arange(1, len(eigenvalues) + 1), np.maximum(eigenvalues, floor),
                marker="o", markersize=3, linewidth=1.3, color=color, label=f"{label}，数值秩 {rank}")
    ax.set_yscale("log")
    ax.axhline(floor, color="gray", linestyle=":", linewidth=1)
    ax.set_title("原始与中心化 Gram 矩阵的特征值谱")
    ax.set_xlabel("特征值序号（降序）")
    ax.set_ylabel("特征值（对数坐标）")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.5, -0.02, f"低于 {floor:.2e} 的值仅在图中置于显示下限；CSV 保留原始特征值。",
             ha="center", fontsize=8)
    fig.savefig(figure_dir / "eigenvalue_spectrum.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_corpus(config, database_path, queries_path, output_dir):
    with closing(sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        hashes = verify_frozen_data(connection, config, queries_path)
        records, vectors = read_vector_corpus(connection, config)
    matrix = vectors.astype(np.float64)
    centered = matrix - matrix.mean(axis=0)
    gram, eigenvalues, singular_values, raw_summary = matrix_analysis(matrix)
    _, centered_eigenvalues, centered_singular_values, centered_summary = matrix_analysis(centered)
    index = {row["chunk_id"]: i for i, row in enumerate(records)}
    pair_values = [{"pair_id": f"P{i:02d}", "chunk_a": a, "chunk_b": b,
                    "cosine": float(gram[index[a], index[b]])}
                   for i, (a, b) in enumerate(CONFUSABLE_PAIRS, 1)]
    off_diagonal = gram[~np.eye(len(gram), dtype=bool)]
    center_error = float(np.max(np.abs(centered.sum(axis=0))))
    diagonal_error = float(np.max(np.abs(np.diag(gram) - 1)))
    summary = {
        "computed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dtype": str(matrix.dtype), "random_seed": config["random_seed"],
        "numpy_version": np.__version__, "matplotlib_version": metadata.version("matplotlib"),
        **hashes,
        "document_order": [row["chunk_id"] for row in records],
        "heatmap_order": heatmap_order(records)[0],
        "rank_threshold_rule": "sigma_max * max(matrix.shape) * float64_eps",
        "G": raw_summary, "Gc": centered_summary,
        "centered_column_sum_max_abs_error": center_error,
        "gram_diagonal_max_abs_error_from_1": diagonal_error,
        "confusable_pair_count": len(pair_values),
        "confusable_pair_mean_cosine": float(np.mean([row["cosine"] for row in pair_values])),
        "all_off_diagonal_mean_cosine": float(off_diagonal.mean()),
        "confusable_pairs": pair_values,
        "spectrum_plot_floor": max(raw_summary["gram_rank_tolerance"], centered_summary["gram_rank_tolerance"]),
        "figure_dpi": 200,
        "checks": {
            "raw_rank_bound": raw_summary["matrix_rank"] <= min(matrix.shape),
            "centered_rank_bound": centered_summary["matrix_rank"] <= min(matrix.shape[0] - 1, matrix.shape[1]),
            "centered_gram_rank_bound": centered_summary["gram_rank"] <= matrix.shape[0] - 1,
            "centered_column_sums_zero": center_error < 1e-12,
            "unit_vector_gram_diagonal": diagonal_error < 1e-5,
        },
    }
    passed = all(summary["checks"].values()) and all(raw_summary["checks"].values()) and all(centered_summary["checks"].values())
    summary["passed"] = passed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "gram_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["chunk_id", *summary["heatmap_order"]])
        order = heatmap_order(records)[1]
        for chunk_id, row in zip(summary["heatmap_order"], gram[np.ix_(order, order)]):
            writer.writerow([chunk_id, *row])
    with (output_dir / "eigenvalues.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["matrix", "component", "eigenvalue", "singular_value_squared", "absolute_error"])
        for name, values, singular in (("G", eigenvalues, singular_values), ("Gc", centered_eigenvalues, centered_singular_values)):
            for i, (value, squared) in enumerate(zip(values, singular ** 2), 1):
                writer.writerow([name, i, value, squared, abs(value - squared)])
    (output_dir / "gram_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_gram(gram, {"G": eigenvalues, "Gc": centered_eigenvalues}, summary, records, output_dir / "figures")
    for name in ("G", "Gc"):
        result = summary[name]
        logging.info("%s: rank(X)=%d rank(G)=%d symmetry_error=%.3g min_eigenvalue=%.12g spectrum_error=%.3g",
                     name, result["matrix_rank"], result["gram_rank"], result["symmetry_max_abs_error"],
                     result["minimum_eigenvalue"], result["spectrum_max_abs_error"])
    logging.info("15 对平均余弦=%.12f；全体非对角平均=%.12f",
                 summary["confusable_pair_mean_cosine"], summary["all_off_diagonal_mean_cosine"])
    logging.info("实验 C 检查通过=%s；结果：%s", passed, output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=ROOT / "database" / "knowledge.db")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = analyze_corpus(load_config(args.config), args.database, args.queries, args.output_dir)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
