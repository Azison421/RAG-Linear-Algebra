"""第 7 天实现：单 Gold 的 Hit@K、MRR、overlap@3 与分层指标。"""

import argparse
import logging
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_config.yaml")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.error("当前为第 1 天接口骨架；查询和 Gold 冻结后于第 7 天实现。")
    raise SystemExit(2)
