"""Đọc cấu hình dùng chung và các tiện ích nhỏ (đường dẫn, seed)."""
import random
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def load_config(path=CONFIG_PATH):
    """Đọc configs/config.yaml thành dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_from(cfg, key):
    """Đổi khóa trong cfg['paths'] thành đường dẫn tuyệt đối."""
    return ROOT_DIR / cfg["paths"][key]


def set_seed(seed=None, cfg=None):
    """Cố định seed cho random và numpy.

    Khi đã chọn framework học sâu, hãy thêm vào đây lệnh đặt seed tương ứng
    (ví dụ torch.manual_seed hoặc tf.random.set_seed).
    """
    if seed is None:
        seed = (cfg or load_config())["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    return seed
