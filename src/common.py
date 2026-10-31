"""读取实验配置，转换方案规定的 Float32 向量 BLOB。"""

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path

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
    if len(blob) != dimension * np.dtype(np.float32).itemsize:
        raise ValueError(f"BLOB 长度应为 {dimension * 4} 字节")
    vector = np.frombuffer(blob, dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("向量包含非有限数值")
    return vector


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
