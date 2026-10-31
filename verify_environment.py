"""验证本地模型及课程所需依赖；示例数据不属于正式课程实验。"""

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(ROOT / ".runtime-cache" / "huggingface")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".runtime-cache" / "matplotlib")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiment_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_path = (config_path.parent / config["model_dir"]).resolve()
    with (model_path / "model.safetensors").open("rb") as model_file:
        model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    assert model_sha256 == config["model_file_sha256"], "模型权重与配置哈希不一致"
    assert config["normalize_embeddings"] is True
    torch.manual_seed(config["random_seed"])
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading local model on {device} ...", flush=True)
    model = SentenceTransformer(
        str(model_path), device=device, local_files_only=True
    )
    model.max_seq_length = config["max_seq_length"]
    encode_options = {
        "batch_size": config["encode_batch_size"],
        "precision": config["embedding_precision"],
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    documents = [
        "向量内积是对应坐标乘积之和。单位向量的内积等于余弦相似度。",
        "Gram 矩阵由向量之间的内积组成，具有实对称和半正定性质。",
        "主成分分析 PCA 对数据中心化后，通过正交投影将高维向量降到低维。",
    ]
    query = config["query_instruction"] + "怎样用 PCA 降低向量维度？"
    modules = json.loads((model_path / "modules.json").read_text(encoding="utf-8"))
    has_normalize_module = any(
        module["type"].rsplit(".", 1)[-1] == "Normalize" for module in modules
    )
    raw_vectors = model.encode(
        documents,
        normalize_embeddings=False,
        **encode_options,
    )
    assert raw_vectors.shape == (3, 512) and np.isfinite(raw_vectors).all()
    raw_norms = np.linalg.norm(raw_vectors, axis=1)
    vectors = model.encode(
        documents,
        normalize_embeddings=True,
        **encode_options,
    )
    query_vectors = model.encode(
        [query],
        normalize_embeddings=True,
        **encode_options,
    )
    assert vectors.shape == (3, 512), vectors.shape
    assert query_vectors.shape == (1, 512), query_vectors.shape
    assert np.isfinite(vectors).all() and np.isfinite(query_vectors).all()
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    np.testing.assert_allclose(
        np.linalg.norm(query_vectors, axis=1), 1.0, atol=1e-5
    )
    scores = query_vectors @ vectors.T
    gram = vectors.astype(np.float64) @ vectors.astype(np.float64).T
    np.testing.assert_allclose(gram, gram.T, atol=1e-12)
    assert np.linalg.eigvalsh(gram).min() >= -1e-10
    vectors64 = vectors.astype(np.float64)
    mean = vectors64.mean(axis=0)
    centered = vectors64 - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    covariance_eigenvalues, _ = np.linalg.eigh(centered.T @ centered)
    np.testing.assert_allclose(
        covariance_eigenvalues[::-1][:len(singular_values)],
        singular_values ** 2,
        rtol=1e-8,
        atol=1e-10,
    )
    projection = vt[:2].T
    reduced = centered @ projection
    reduced_query = (query_vectors.astype(np.float64) - mean) @ projection
    assert reduced.shape == (3, 2) and reduced_query.shape == (1, 2)
    reference_pca = PCA(n_components=2, svd_solver="full").fit(vectors64)
    np.testing.assert_allclose(
        singular_values[:2] ** 2 / (len(vectors64) - 1),
        reference_pca.explained_variance_,
        rtol=1e-8,
        atol=1e-10,
    )
    sample_config = {
        "model_name": "BAAI/bge-small-zh-v1.5",
        "model_dir": str(model_path),
        "query_instruction": "为这个句子生成表示以用于检索相关文章：",
        "pca_dims": [59, 32, 16, 8],
        "random_seed": 20261031,
    }
    yaml_text = yaml.safe_dump(sample_config, allow_unicode=True)
    assert yaml.safe_load(yaml_text) == sample_config
    tau = kendalltau([1, 2, 3], [1, 2, 3])
    np.testing.assert_allclose(tau.statistic, 1.0, atol=1e-12)
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE test_vectors (id INTEGER, vector BLOB)")
        connection.execute(
            "INSERT INTO test_vectors VALUES (?, ?)",
            (1, vectors[0].astype(np.float32).tobytes()),
        )
        blob = connection.execute("SELECT vector FROM test_vectors").fetchone()[0]
        np.testing.assert_array_equal(np.frombuffer(blob, dtype=np.float32), vectors[0])
    csv_buffer = StringIO()
    pd.DataFrame(scores, columns=["inner_product", "gram", "pca"]).to_csv(
        csv_buffer, index=False
    )
    assert "pca" in csv_buffer.getvalue()
    chinese_font = font_manager.findfont(
        font_manager.FontProperties(family="SimSun"), fallback_to_default=False
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([512, 32, 16, 8], [2048, 128, 64, 32])
    ax.set_title("环境检查：向量存储", fontproperties=font_manager.FontProperties(fname=chinese_font))
    image = BytesIO()
    fig.savefig(image, format="png", dpi=100)
    plt.close(fig)
    assert image.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    packages = [
        "torch", "sentence-transformers", "transformers", "huggingface-hub",
        "tokenizers", "safetensors", "numpy", "pandas", "scikit-learn", "matplotlib",
        "PyYAML", "scipy",
    ]
    result = {
        "status": "passed",
        "purpose": "环境验证；三条示例文本不属于正式课程实验",
        "specification": "RAG_向量检索线性代数大作业_V3.1.md",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "model_path": str(model_path),
        "model_name": config["model_name"],
        "model_file_sha256": model_sha256,
        "encoding_parameters": {
            "batch_size": config["encode_batch_size"],
            "precision": config["embedding_precision"],
            "max_seq_length": model.max_seq_length,
            "document_instruction": "",
            "query_instruction": config["query_instruction"],
            "normalize_embeddings": True,
            "random_seed": config["random_seed"],
        },
        "tokenizer_class": type(model.tokenizer).__name__,
        "model_modules": [module["type"] for module in modules],
        "local_files_only": True,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "document_vector_shape": list(vectors.shape),
        "query_vector_shape": list(query_vectors.shape),
        "document_vector_norms": norms.tolist(),
        "has_normalize_module": has_normalize_module,
        "normalize_embeddings_false_norms": raw_norms.tolist(),
        "normalize_embeddings_false_norm_summary": {
            "min": float(raw_norms.min()),
            "mean": float(raw_norms.mean()),
            "max": float(raw_norms.max()),
        },
        "sample_similarity_scores": scores[0].tolist(),
        "sample_top_document": int(scores[0].argmax()) + 1,
        "pca_output_shape": list(reduced.shape),
        "pca_svd_eigh_sklearn": "passed",
        "yaml_unicode_roundtrip": "passed",
        "scipy_kendalltau": "passed",
        "gram_checks": "对称性和半正定性通过",
        "sqlite_roundtrip": "passed",
        "sqlite_version": sqlite3.sqlite_version,
        "pandas_csv": "passed",
        "matplotlib_png": "passed",
        "chinese_font": chinese_font,
        "pip_check": pip_check.stdout.strip(),
        "package_versions": {name: metadata.version(name) for name in packages},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    norm_rows = [
        {
            "run_scope": "smoke",
            "record_type": "sample",
            "text_id": f"SMOKE-{i + 1:02d}",
            "raw_norm": float(raw_norms[i]),
            "normalized_norm": float(norms[i]),
        }
        for i in range(len(documents))
    ]
    norm_rows.append({
        "run_scope": "smoke",
        "record_type": "summary",
        "raw_norm_min": float(raw_norms.min()),
        "raw_norm_mean": float(raw_norms.mean()),
        "raw_norm_max": float(raw_norms.max()),
    })
    pd.DataFrame(norm_rows).to_csv(
        args.output_dir / "embedding_norms.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
