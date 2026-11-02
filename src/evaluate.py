"""实验 B：从完整候选排名重算单 Gold 指标、成功失败题数与分层结果。"""

import argparse
import csv
from hashlib import sha256
import json
import logging
from pathlib import Path

import numpy as np

from common import DEFAULT_CONFIG, ROOT, load_config
from pca_analysis import configure_plotting, read_documents
from search import rank_scores


def load_queries(path, config, chunk_ids):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        queries = list(csv.DictReader(stream))
    if (len(queries) != config["n_queries"] or len({q["query_id"] for q in queries}) != len(queries)
            or any(q["gold_chunk_id"] not in chunk_ids or q["query_type"] not in
                   ("contrast", "paraphrase", "formula") for q in queries)):
        raise ValueError("查询数、唯一 ID、类型或 Gold 不符合配置与知识库")
    return queries


def evaluate_rankings(rankings, queries, chunk_ids, pca_rows, baseline_dimension=512):
    """主指标只认 gold_chunk_id，MRR 不截断为 @3。"""
    expected = np.arange(len(chunk_ids))
    for order in rankings.values():
        if (order.shape != (len(queries), len(chunk_ids)) or
                not np.array_equal(np.sort(order, axis=1), np.broadcast_to(expected, order.shape))):
            raise ValueError("指标必须使用覆盖所有候选、每个 ID 恰好一次的完整排名")
    baseline = rankings["Baseline"]
    by_method = {row["method"]: row for row in pca_rows}
    indices = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
    details, metrics = [], []
    for method, order in rankings.items():
        method_details = []
        for i, query in enumerate(queries):
            gold_rank = int(np.flatnonzero(order[i] == indices[query["gold_chunk_id"]])[0] + 1)
            method_details.append({
                "method": method, "query_id": query["query_id"], "query_type": query["query_type"],
                "gold_chunk_id": query["gold_chunk_id"], "gold_rank": gold_rank,
                "hit_at_1": int(gold_rank == 1), "hit_at_3": int(gold_rank <= 3),
                "reciprocal_rank": 1.0 / gold_rank,
                "overlap_at_3": len(set(order[i, :3]) & set(baseline[i, :3])) / 3,
            })
        details.extend(method_details)
        for stratum in ("all", "contrast", "paraphrase", "formula"):
            subset = [row for row in method_details if stratum == "all" or row["query_type"] == stratum]
            if not subset:
                continue
            count = len(subset)
            hit1 = sum(row["hit_at_1"] for row in subset)
            hit3 = sum(row["hit_at_3"] for row in subset)
            metrics.append({
                "method": method,
                "dimension": baseline_dimension if method == "Baseline" else int(by_method[method]["dimension"]),
                "query_type": stratum, "n_queries": count,
                "hit_at_1": hit1 / count, "hit_at_3": hit3 / count,
                "mrr": float(np.mean([row["reciprocal_rank"] for row in subset])),
                "overlap_at_3": float(np.mean([row["overlap_at_3"] for row in subset])),
                "hit_at_1_successes": hit1, "hit_at_1_failures": count - hit1,
                "hit_at_3_successes": hit3, "hit_at_3_failures": count - hit3,
                "cumulative_explained_variance": "" if method == "Baseline" else by_method[method]["cumulative_explained_variance"],
            })
    return metrics, details


def save_metrics(output_dir, metrics, details):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("metrics.csv", metrics), ("query_metrics.csv", details)):
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in metrics:
        if row["query_type"] == "all":
            logging.info("%s Hit@1=%d/%d Hit@3=%d/%d MRR=%.12g overlap@3=%.12g", row["method"],
                         row["hit_at_1_successes"], row["n_queries"], row["hit_at_3_successes"],
                         row["n_queries"], row["mrr"], row["overlap_at_3"])


def read_rankings(path, queries, chunk_ids, config):
    methods = {"Baseline": config["embedding_dim"], **{f"PCA-{r}": r for r in config["pca_dims"]}}
    orders = {method: np.full((len(queries), len(chunk_ids)), -1, dtype=int) for method in methods}
    scores = {method: np.full((len(queries), len(chunk_ids)), np.nan) for method in methods}
    query_indices = {q["query_id"]: i for i, q in enumerate(queries)}
    chunk_indices = {chunk_id: i for i, chunk_id in enumerate(chunk_ids)}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            method, query_id, chunk_id = row["method"], row["query_id"], row["chunk_id"]
            rank = int(row["rank"]) - 1
            if (method not in methods or query_id not in query_indices or chunk_id not in chunk_indices
                    or int(row["dimension"]) != methods[method] or not 0 <= rank < len(chunk_ids)):
                raise ValueError("排名记录包含未知方法、查询、候选、维度或无效名次")
            i, j = query_indices[query_id], chunk_indices[chunk_id]
            if orders[method][i, rank] != -1 or np.isfinite(scores[method][i, j]):
                raise ValueError("排名记录重复")
            orders[method][i, rank] = j
            scores[method][i, j] = float(row["score"])
    for method in methods:
        if not np.isfinite(scores[method]).all() or not np.array_equal(rank_scores(scores[method], chunk_ids), orders[method]):
            raise ValueError("排名缺失，或未按分数降序、同分 ID 升序排列")
    return orders


CASE_SELECTION = (
    ("Baseline", "Q02", "PCA-16", "概念相邻",
     "目标要求一减余弦的范围和三角性质。第一名 LA-08 讨论三角不等式，也提到一减余弦并非严格度量，但没有完整给出目标定义与范围；CF-02、LA-09 讨论方向。相关术语接近，仍不足以替代 CF-04。PCA-16 将目标提至第3名。"),
    ("Baseline", "Q05", "PCA-16", "标注歧义",
     "题目问遍历所有系数形成的集合及其是否含零向量。首位 CF-08 已说明所有系数遍历所得集合叫线性张成，并允许零系数，因此也能推得题目所问；与 Gold CF-09 存在回答覆盖交叠。这是单目标标注的歧义线索，不能简单归为纯模型错误。冻结 Gold 和主指标保持不变；PCA-16 的第3名、PCA-8 的第2名仅表示主目标被检出。"),
    ("Baseline", "Q18", "PCA-16", "术语不同",
     "查询用一次项等式、特定可行结果、右端为零等表达，避开线性方程组、特解和齐次解术语。LA-12 完整给出特解加齐次解的结构，返回的 GR-02、GR-03、CF-07 虽涉及零值或零空间，却没有回答非齐次解集的平移结构。各维度均未进入Top-3。"),
    ("PCA-59", "Q21", "Baseline", "概念相邻",
     "该题需要把 z 的二次型改写为 X 转置乘 z 的长度平方。CF-13 讲特征值，CF-18、LA-14 讲一般半正定和二次型，与目标相邻但未完整替代 GR-02 的该等式。基线目标第1名，59维时第5名；此维度保留全部非零文档主成分，不能把变化归因于截去非零主成分。"),
    ("PCA-32", "Q01", "Baseline", "概念相邻",
     "查询问坐标乘积求和及其线性缩放，CF-01 直接回答。32维前三项转为坐标表示、降维、归一化，均有关联的坐标或长度表述，但没有完整回答该内积缩放关系。基线目标第1名变为第4名，说明保留约89.57%的文档方差仍不保证本题命中。"),
    ("PCA-16", "Q16", "Baseline", "术语不同",
     "查询用无冗余、表示每个成员、成员数不变来改写基与维数。LA-03 同时给出无关、张成及基的成员数不变；前三项为余弦、张成、组合，后两项只涉及部分条件。目标由基线第2名降至第7名，而8维又回到第1名，显示单题排序不随维度单调变化。"),
    ("PCA-8", "Q17", "Baseline", "术语不同",
     "查询以数表、行列互换、分别处理来表达转置，并问乘积处理时次序反转。LA-06 直接给出转置乘积规则；8维前三项是批量矩阵检索、Top-K及评价指标，缺少该反序法则。目标由基线第1名降至第27名，反映本次低维表示未保住该改写题的目标排序。"),
)


def save_failure_cases(path, rankings_path, queries, records):
    """案例数值全部查自完整排名；原因是对照查询和正文后的人工分析。"""
    query_map = {row["query_id"]: row for row in queries}
    document_map = {row["chunk_id"]: row for row in records}
    rankings = {}
    with Path(rankings_path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rankings.setdefault((row["method"], row["query_id"]), []).append(row)
    cases = []
    for i, (method, query_id, comparison, category, reason) in enumerate(CASE_SELECTION, 1):
        query = query_map[query_id]
        ranked = rankings[method, query_id]
        target = next(row for row in ranked if row["chunk_id"] == query["gold_chunk_id"])
        counterpart = rankings[comparison, query_id]
        counterpart_target = next(row for row in counterpart if row["chunk_id"] == query["gold_chunk_id"])
        if int(target["rank"]) <= 3:
            raise ValueError(f"{method}/{query_id} 已非 Top-3 失败，请重新核查案例解释")
        row = {
            "case_id": f"F{i:02d}", "method": method, "dimension": int(target["dimension"]),
            "query_id": query_id, "query_type": query["query_type"], "query_text": query["query_text"],
            "gold_chunk_id": query["gold_chunk_id"], "gold_title": document_map[query["gold_chunk_id"]]["title"],
            "gold_rank": int(target["rank"]), "gold_score": float(target["score"]),
        }
        for rank, retrieved in enumerate(ranked[:3], 1):
            row[f"top{rank}_id"] = retrieved["chunk_id"]
            row[f"top{rank}_title"] = document_map[retrieved["chunk_id"]]["title"]
            row[f"top{rank}_score"] = float(retrieved["score"])
        row.update({
            "comparison_method": comparison, "comparison_gold_rank": int(counterpart_target["rank"]),
            "comparison_hit_at_3": int(int(counterpart_target["rank"]) <= 3),
            "comparison_top3_ids": json.dumps([r["chunk_id"] for r in counterpart[:3]]),
            "failure_category": category, "reason_analysis": reason,
        })
        cases.append(row)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    logging.info("saved %s cases=%d unique_queries=%d", path, len(cases), len({r["query_id"] for r in cases}))


def plot_comparison(metrics, figure_dir):
    plt = configure_plotting()
    from matplotlib.ticker import PercentFormatter
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    overall = [row for row in metrics if row["query_type"] == "all"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.2), constrained_layout=True)
    positions = np.arange(len(overall))
    for offset, key, label, color in ((-0.24, "hit_at_1", "Hit@1", "#3D6EA8"),
                                      (0, "hit_at_3", "Hit@3", "#187C80"),
                                      (0.24, "mrr", "MRR（完整60候选）", "#C58032")):
        values = [row[key] for row in overall]
        bars = axes[0].bar(positions + offset, values, 0.24, label=label, color=color)
        labels = [f"{row[key + '_successes']}/{row['n_queries']}"
                  if key != "mrr" else f"{row[key]:.3f}" for row in overall]
        axes[0].bar_label(bars, labels=labels, padding=3, fontsize=9)
    labels = ["基线\n512维"] + [f"PCA\n{row['dimension']}维" for row in overall[1:]]
    axes[0].set_xticks(positions, labels)
    axes[0].set(ylim=(0, 1.13), ylabel="命中比例 / 平均倒数排名", title="25 条冻结查询的检索表现")
    axes[0].legend(loc="upper right", ncol=3, fontsize=10)
    projected = overall[1:]
    bars = axes[1].bar(np.arange(len(projected)), [row["overlap_at_3"] for row in projected],
                       width=0.55, color="#54778A")
    axes[1].bar_label(bars, labels=[f"{row['overlap_at_3']:.2%}" for row in projected], padding=4)
    axes[1].set_xticks(np.arange(len(projected)), [f"PCA-{row['dimension']}" for row in projected])
    axes[1].set(ylim=(0, 1.12), ylabel="平均 Top-3 集合重合率", title="相对于未中心化 512 维基线的 Top-3 保留程度")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.set_yticks(np.arange(0, 1.01, .2))
    path = figure_dir / "pca_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logging.info("saved %s dpi=200", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=ROOT / "database" / "knowledge.db")
    parser.add_argument("--queries", type=Path, default=ROOT / "data" / "queries.csv")
    parser.add_argument("--rankings", type=Path, default=ROOT / "results" / "retrieval_rankings.csv")
    parser.add_argument("--pca-results", type=Path, default=ROOT / "results" / "pca_results.csv")
    parser.add_argument("--run-info", type=Path, default=ROOT / "results" / "retrieval_run.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--analyze-only", action="store_true", help="生成对照图与案例，不重写第七天指标 CSV")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    records, _, hashes = read_documents(config, args.database, args.queries)
    run_info = json.loads(args.run_info.read_text(encoding="utf-8"))
    if any(run_info[key] != value for key, value in hashes.items()):
        raise ValueError("排名实验的冻结数据与当前配置不一致")
    for name, path in (("retrieval_rankings.csv", args.rankings), ("pca_results.csv", args.pca_results)):
        if sha256(path.read_bytes()).hexdigest() != run_info["result_sha256"][name]:
            raise ValueError(f"{name} 不匹配本次实验运行记录")
    ids = [row["chunk_id"] for row in records]
    queries = load_queries(args.queries, config, ids)
    orders = read_rankings(args.rankings, queries, ids, config)
    with args.pca_results.open(encoding="utf-8", newline="") as stream:
        pca_rows = list(csv.DictReader(stream))
    metrics, details = evaluate_rankings(orders, queries, ids, pca_rows, config["embedding_dim"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        save_metrics(args.output_dir, metrics, details)
    plot_comparison(metrics, args.output_dir / "figures")
    save_failure_cases(args.output_dir / "failure_cases.csv", args.rankings, queries, records)


if __name__ == "__main__":
    main()
