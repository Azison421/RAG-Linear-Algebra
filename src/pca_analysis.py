"""实验 B：文档均值与 NumPy SVD 实现 PCA，eigh 和 sklearn 仅用于对拍。"""

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
from sklearn.decomposition import PCA

from common import DEFAULT_CONFIG, ROOT, load_config, read_vector_corpus, verify_frozen_data


def normalize_rows(matrix):
    """统一以 float64 计算；数值零投影不能定义余弦，直接报告。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("存在非有限向量或数值零投影，不能计算余弦相似度")
    return matrix / norms[:, None]


def read_documents(config, database, queries):
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        hashes = verify_frozen_data(connection, config, queries)
        records, vectors = read_vector_corpus(connection, config)
    return records, vectors.astype(np.float64), hashes


def fit_pca(documents, dimensions, seed):
    """只拟合文档，返回均值、右奇异向量及数值验收结果。"""
    documents = np.asarray(documents, dtype=np.float64)
    mean = documents.mean(axis=0)
    centered = documents - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = singular_values[0] * max(centered.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    if not dimensions or any(r < 1 or r > rank for r in dimensions):
        raise ValueError(f"PCA 维度必须介于 1 和中心化数值秩 {rank} 之间")
    squared = singular_values ** 2
    eigenvalues, _ = np.linalg.eigh(centered.T @ centered)
    eigenvalues = eigenvalues[::-1]
    nonzero_error = float(np.max(np.abs(eigenvalues[:rank] - squared[:rank]) / squared[:rank]))
    zero_error = float(np.max(np.abs(eigenvalues[rank:]))) if rank < len(eigenvalues) else 0.0
    reference = PCA(n_components=max(dimensions), svd_solver="full", random_state=seed).fit(documents)
    variance = squared / (len(documents) - 1)
    sklearn_error = float(np.max(np.abs(reference.explained_variance_ - variance[:max(dimensions)])))
    original_distances = np.linalg.norm(centered[:, None, :] - centered[None, :, :], axis=2)
    gram = centered @ centered.T
    rows = []
    for r in dimensions:
        basis = vt[:r].T
        projected = centered @ basis
        reconstruction_error = float(np.linalg.norm(centered - projected @ basis.T, "fro") ** 2)
        tail_energy = float(squared[r:].sum())
        gram_error = float(np.max(np.abs(projected @ projected.T - gram)))
        distance_increase = float(np.max(
            np.linalg.norm(projected[:, None, :] - projected[None, :, :], axis=2) - original_distances))
        reference_basis = reference.components_[:r].T
        projector_error = float(np.max(np.abs(basis @ basis.T - reference_basis @ reference_basis.T)))
        orthogonality_error = float(np.max(np.abs(basis.T @ basis - np.eye(r))))
        checks = (orthogonality_error < 1e-12, distance_increase < 1e-12,
                  np.isclose(reconstruction_error, tail_energy, rtol=1e-8, atol=1e-12),
                  projector_error < 1e-8, r != rank or gram_error < 1e-8)
        normalize_rows(projected)
        rows.append({
            "method": f"PCA-{r}", "dimension": r,
            "cumulative_explained_variance": float(squared[:r].sum() / squared.sum()),
            "reconstruction_squared_frobenius_error": reconstruction_error,
            "discarded_singular_value_energy": tail_energy,
            "centered_gram_max_abs_error": gram_error,
            "projection_distance_max_increase": distance_increase,
            "basis_orthogonality_max_abs_error": orthogonality_error,
            "sklearn_projector_max_abs_error": projector_error,
            "minimum_document_projection_norm": float(np.linalg.norm(projected, axis=1).min()),
            "float32_bytes_per_vector": 4 * r,
            "relative_vector_storage": r / documents.shape[1], "passed": bool(all(checks)),
        })
    checks = {
        "eigh_nonzero_relative_error_below_1e-8": nonzero_error < 1e-8,
        "eigh_zero_spectrum_absolute_error_below_1e-10": zero_error < 1e-10,
        "sklearn_explained_variance_matches": bool(np.allclose(
            reference.explained_variance_, variance[:max(dimensions)], rtol=1e-8, atol=1e-12)),
        "all_dimensions_passed": all(row["passed"] for row in rows),
    }
    summary = {
        "centered_rank": rank, "rank_tolerance": float(tolerance),
        "eigh_nonzero_max_relative_error": nonzero_error,
        "eigh_zero_spectrum_max_abs_error": zero_error,
        "sklearn_explained_variance_max_abs_error": sklearn_error,
        "singular_values": singular_values.tolist(), "explained_variance": variance.tolist(),
        "cumulative_explained_variance": (np.cumsum(squared) / squared.sum()).tolist(),
        "eigh_eigenvalues": eigenvalues.tolist(), "checks": checks, "passed": all(checks.values()),
    }
    if not summary["passed"]:
        raise ValueError(f"PCA 数值核对失败：{checks}")
    return mean, vt, rows, summary


def save_pca(output_dir, rows, summary, config, hashes, chunk_ids):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pca_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "analyzed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dtype": "float64", "fit_on": "documents_only", "svd_full_matrices": False,
        "sklearn_solver": "full", "random_seed": config["random_seed"],
        "dimensions": config["pca_dims"], "chunk_ids": chunk_ids, **hashes, **summary,
        "package_versions": {name: metadata.version(name) for name in ("numpy", "scikit-learn")},
    }
    (output_dir / "pca_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("PCA centered_rank=%d eigh_relative_error=%.12g sklearn_variance_error=%.12g passed=%s",
                 summary["centered_rank"], summary["eigh_nonzero_max_relative_error"],
                 summary["sklearn_explained_variance_max_abs_error"], summary["passed"])
    for row in rows:
        logging.info("%s cumulative_variance=%.12g gram_error=%.12g passed=%s", row["method"],
                     row["cumulative_explained_variance"], row["centered_gram_max_abs_error"], row["passed"])


def configure_plotting():
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".runtime-cache" / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont(font_manager.FontProperties(family="Microsoft YaHei"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 11})
    return plt


def plot_pca(rows, summary, figure_dir):
    """只展示中心化文档谱；512 维基线不参与累计解释方差曲线。"""
    plt = configure_plotting()
    from matplotlib.ticker import PercentFormatter
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rank = summary["centered_rank"]
    singular = np.asarray(summary["singular_values"], dtype=np.float64)
    cumulative = np.asarray(summary["cumulative_explained_variance"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), constrained_layout=True)
    positions = np.arange(1, rank + 1)
    axes[0].plot(positions, singular[:rank], color="#187C80", linewidth=1.8)
    axes[0].set(xlabel="主成分序号", ylabel="奇异值（对数刻度）", title="中心化文档的非零奇异值")
    axes[0].set_yscale("log")
    axes[0].text(0.97, 0.96, f"数值秩 {rank}\n第 {len(singular)} 个奇异值 {singular[-1]:.2e}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=10)
    axes[1].plot(positions, cumulative[:rank], color="#C54E36", linewidth=1.8)
    for row in sorted(rows, key=lambda row: int(row["dimension"])):
        dimension = int(row["dimension"])
        value = float(row["cumulative_explained_variance"])
        axes[1].scatter(dimension, value, color="#C54E36", zorder=3)
        axes[1].annotate(f"{dimension}维：{value:.2%}", (dimension, value),
                         xytext=(-4 if dimension == rank else 6, 10 if dimension == rank else 7),
                         textcoords="offset points", ha="right" if dimension == rank else "left", fontsize=10)
    axes[1].set(xlabel="保留维度 r", ylabel="累计解释方差", ylim=(0, 1.08), title="文档变差的保留比例")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle("PCA 文档谱与累计解释方差（60 个文档，float64）", fontsize=13)
    path = figure_dir / "pca_variance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logging.info("saved %s dpi=200", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=ROOT / "database" / "knowledge.db")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--plot-only", action="store_true", help="只用已保存的 PCA 结果重建图，不重写数值结果")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="plot-only 的 PCA CSV/JSON 输入目录")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    records, documents, hashes = read_documents(config, args.database, args.queries)
    if args.plot_only:
        summary = json.loads((args.results_dir / "pca_analysis.json").read_text(encoding="utf-8"))
        with (args.results_dir / "pca_results.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if (any(summary[key] != value for key, value in hashes.items())
                or summary["chunk_ids"] != [row["chunk_id"] for row in records]
                or summary["dimensions"] != config["pca_dims"]
                or [int(row["dimension"]) for row in rows] != config["pca_dims"]):
            raise ValueError("PCA 绘图输入与冻结数据、文档顺序或维度不一致")
        squared = np.asarray(summary["singular_values"]) ** 2
        for row in rows:
            if not np.isclose(float(row["cumulative_explained_variance"]),
                              squared[:int(row["dimension"])].sum() / squared.sum(), rtol=0, atol=1e-12):
                raise ValueError("PCA CSV 与奇异值谱不一致")
    else:
        _, _, rows, summary = fit_pca(documents, config["pca_dims"], config["random_seed"])
        save_pca(args.output_dir, rows, summary, config, hashes, [row["chunk_id"] for row in records])
    plot_pca(rows, summary, args.output_dir / "figures")


if __name__ == "__main__":
    main()
