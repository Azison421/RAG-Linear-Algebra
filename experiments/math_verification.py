"""第 6 天实现：使用 float64 人工向量完成实验 A1 至 A5。"""

import argparse
import logging
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_config.yaml")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.error("当前为第 1 天接口骨架；正式数学实验于第 6 天实现。")
    raise SystemExit(2)
