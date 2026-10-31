"""第 7 天实现：NumPy SVD 主流程、eigh 与 sklearn 对拍及维度扫描。"""

import argparse
import logging
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "experiment_config.yaml")
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.error("当前为第 1 天接口骨架；数据冻结和编码后于第 7 天实现。")
    raise SystemExit(2)
