"""读取实验配置，转换方案规定的 Float32 向量 BLOB。"""

import argparse
from hashlib import file_digest, sha256
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import random
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiment_config.yaml"
CORPUS_TEXT_FIELDS = (
    "chunk_id", "group_code", "topic", "title", "content", "confusable_with",
    "source_title", "source_author", "source_year", "source_url",
)


def corpus_text_sha256(connection):
    """按固定 ID 顺序散列文本与来源，排除后续写入的模型和向量字段。"""
    rows = connection.execute(
        f"SELECT {', '.join(CORPUS_TEXT_FIELDS)} FROM chunks ORDER BY chunk_id"
    ).fetchall()
    records = [dict(zip(CORPUS_TEXT_FIELDS, row)) for row in rows]
    payload = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def load_config(path=DEFAULT_CONFIG):
    """相对模型路径以配置文件所在目录为基准。"""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["model_dir"] = str((path.parent / config["model_dir"]).resolve())
    return config


def vector_to_blob(vector, dimension=512):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(f"需要含 {dimension} 个有限数值的一维向量")
    return vector.tobytes()


def blob_to_vector(blob, dimension=512):
    if blob is None:
        raise ValueError("尚无正式向量，请先运行 embed.py")
    if len(blob) != dimension * np.dtype(np.float32).itemsize:
        raise ValueError(f"BLOB 长度应为 {dimension * 4} 字节")
    vector = np.frombuffer(blob, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("向量包含非有限数值")
    return vector


def verify_frozen_data(connection, config, queries_path):
    """编码和检索共用冻结检查，不改动正文或 Gold。"""
    hashes = {
        "corpus_sha256": corpus_text_sha256(connection),
        "queries_sha256": sha256(Path(queries_path).read_bytes()).hexdigest(),
    }
    for field, actual in hashes.items():
        if not config.get(field) or actual != config[field]:
            raise ValueError(f"{field} 与冻结值不一致，请核对数据及 CHANGELOG")
    count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if count != config["n_chunks"]:
        raise ValueError("知识块数量与配置不一致")
    return hashes


def check_vectors(vectors, count, dimension, unit=True):
    """检查 Float32 编码输出；用 float64 统计模长。"""
    vectors = np.asarray(vectors)
    if vectors.shape != (count, dimension) or vectors.dtype != np.float32:
        raise ValueError(f"需要形状 {(count, dimension)} 的 Float32 向量")
    if not np.isfinite(vectors).all():
        raise ValueError("向量含非有限数值")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.any(norms <= 0):
        raise ValueError("编码产生零向量，不能计算余弦相似度")
    if unit and np.max(np.abs(norms - 1.0)) > 1e-5:
        raise ValueError("归一化向量模长偏离 1 超过 1e-5")
    return norms


def load_encoder(config):
    """核对本地权重并加载唯一编码器；同时返回复现所需运行信息。"""
    if config["normalize_embeddings"] is not True or config["embedding_precision"] != "float32":
        raise ValueError("正式编码必须显式归一化并使用 float32")
    model_dir = Path(config["model_dir"])
    with (model_dir / "model.safetensors").open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if digest != config["model_file_sha256"]:
        raise ValueError("本地模型权重与配置 SHA-256 不一致")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / ".runtime-cache" / "huggingface")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer
    import torch

    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("加载本地模型 model=%s device=%s", config["model_name"], device)
    model = SentenceTransformer(str(model_dir), device=device, local_files_only=True)
    model.float()
    model.max_seq_length = config["max_seq_length"]
    if model.get_embedding_dimension() != config["embedding_dim"]:
        raise ValueError("模型输出维度与配置不一致")
    modules = json.loads((model_dir / "modules.json").read_text(encoding="utf-8"))
    packages = ("torch", "sentence-transformers", "transformers", "numpy", "tokenizers", "safetensors", "PyYAML")
    run_info = {
        "python": sys.version,
        "model_name": config["model_name"],
        "model_dir": str(model_dir),
        "model_file_sha256": digest,
        "model_revision": digest[:12],
        "model_download_date": config["model_download_date"],
        "local_files_only": True,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "tokenizer_class": type(model.tokenizer).__name__,
        "model_modules": [module["type"] for module in modules],
        "has_normalize_module": any(module["type"].rsplit(".", 1)[-1] == "Normalize" for module in modules),
        "encoding_parameters": {
            "batch_size": config["encode_batch_size"],
            "max_seq_length": model.max_seq_length,
            "precision": config["embedding_precision"],
            "document_instruction": "",
            "query_instruction": config["query_instruction"],
            "normalize_embeddings": True,
            "diagnostic_normalize_embeddings": False,
            "random_seed": config["random_seed"],
        },
        "package_versions": {name: metadata.version(name) for name in packages},
    }
    return model, run_info


def encode_texts(model, texts, config, *, query=False, normalize=True):
    """文档只输入正文，查询只在这里添加一次固定前缀。"""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("编码文本不能为空")
    inputs = [config["query_instruction"] + text for text in texts] if query else texts
    vectors = model.encode(
        inputs,
        prompt="",
        batch_size=config["encode_batch_size"],
        precision=config["embedding_precision"],
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    check_vectors(vectors, len(texts), config["embedding_dim"], unit=normalize)
    return vectors


def read_vector_corpus(connection, config):
    """按固定 ID 读取已入库向量，供检索及后续矩阵实验使用。"""
    cursor = connection.execute("SELECT * FROM chunks ORDER BY chunk_id")
    names = [column[0] for column in cursor.description]
    records = [dict(zip(names, row)) for row in cursor.fetchall()]
    if len(records) != config["n_chunks"]:
        raise ValueError("向量库数量与配置不一致")
    for row in records:
        if (row["model_name"], row["model_revision"], row["embedding_dim"]) != (
            config["model_name"], config["model_file_sha256"][:12], config["embedding_dim"]
        ):
            raise ValueError(f"{row['chunk_id']} 的向量未生成或模型版本不匹配，请运行 embed.py")
        if row["raw_norm"] is None or not np.isfinite(row["raw_norm"]) or row["raw_norm"] <= 0:
            raise ValueError(f"{row['chunk_id']} 缺少有效原始模长")
    vectors = np.stack([blob_to_vector(row["embedding"], config["embedding_dim"]) for row in records])
    check_vectors(vectors, len(records), config["embedding_dim"])
    return records, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    logging.info("model=%s dimension=%s", config["model_name"], config["embedding_dim"])
    logging.info("model_dir=%s", config["model_dir"])
    logging.info("corpus_sha256=%s queries_sha256=%s", config["corpus_sha256"], config["queries_sha256"])


if __name__ == "__main__":
    main()
