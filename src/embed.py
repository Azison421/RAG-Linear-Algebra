"""第 5 天实现：本地 BGE 编码、模长诊断与归一化向量入库。"""

import argparse
import logging
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_config.yaml")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.error("当前为第 1 天接口骨架；正文核准后于第 5 天实现。")
    raise SystemExit(2)
