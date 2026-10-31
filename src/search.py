"""第 5 天实现：NumPy 精确检索并按 ID 返回 Top-K 正文与来源。"""

import argparse
import logging
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_config.yaml")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.error("当前为第 1 天接口骨架；正式向量生成后于第 5 天实现。")
    raise SystemExit(2)
