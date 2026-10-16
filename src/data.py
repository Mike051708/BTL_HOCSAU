"""Đọc dữ liệu theo giờ và chia train/validation/test theo thời gian.

Dùng chung cho mọi thành viên:
    from src.data import load_hourly_data, split_by_time, load_splits
Chạy trực tiếp để tạo lại train.csv / validation.csv / test.csv:
    python src/data.py
Mốc chia lấy từ configs/config.yaml (mục split), không đặt cứng trong code.
"""
import sys
from pathlib import Path

import pandas as pd

try:  # chạy dạng module (from src.data import ...)
    from src.config import ROOT_DIR, load_config, path_from
except ImportError:  # chạy trực tiếp: python src/data.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ROOT_DIR, load_config, path_from

SPLIT_NAMES = ("train", "validation", "test")


def load_hourly_data(cfg=None, customers=None):
    """Đọc series_hourly.csv, kiểm tra chất lượng và trả về DataFrame (chỉ số = timestamp).

    Giá trị là kW trung bình mỗi giờ; nhãn thời gian là mốc kết thúc của giờ.
    Không tự điền dữ liệu thiếu và không xóa số 0: chỉ kiểm tra và báo lỗi.
    """
    cfg = cfg or load_config()
    path = path_from(cfg, "hourly_file")
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy chạy: python src/data_preprocessing.py"
        )

    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    if customers is not None:
        missing = [c for c in customers if c not in df.columns]
        if missing:
            raise ValueError(f"Không có các cột: {missing}")
        df = df[list(customers)]

    if df.index.has_duplicates:
        raise ValueError("Có timestamp trùng lặp.")
    if df.isna().any().any():
        raise ValueError("Dữ liệu theo giờ có giá trị thiếu.")
    if (df < 0).any().any():
        raise ValueError("Dữ liệu theo giờ có giá trị âm.")
    gaps = df.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta("1h")).all():
        raise ValueError("Chỉ số thời gian không đều 1 giờ (thiếu hoặc nhảy giờ).")
    return df


def split_by_time(df, cfg=None):
    """Chia theo mốc thời gian trong config (không xáo trộn). Trả về (train, validation, test).

    train      : timestamp <= split.train_end
    validation : split.train_end < timestamp <= split.val_end
    test       : timestamp > split.val_end
    """
    cfg = cfg or load_config()
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    if not train_end < val_end:
        raise ValueError("Cần train_end < val_end trong config.")

    train = df.loc[df.index <= train_end]
    validation = df.loc[(df.index > train_end) & (df.index <= val_end)]
    test = df.loc[df.index > val_end]

    expected = cfg["split"].get("expected_rows")
    if expected:
        got = {"train": len(train), "validation": len(validation), "test": len(test)}
        if got != expected:
            raise ValueError(f"Số dòng từng tập {got} khác kỳ vọng {expected}.")
    if not (train.index.max() < validation.index.min()
            and validation.index.max() < test.index.min()):
        raise ValueError("Các tập bị chồng lấn thời gian.")
    return train, validation, test


def load_splits(cfg=None, customers=None):
    """Tiện ích: đọc dữ liệu giờ và trả về dict {'train','validation','test'}."""
    cfg = cfg or load_config()
    df = load_hourly_data(cfg, customers)
    return dict(zip(SPLIT_NAMES, split_by_time(df, cfg)))


def split_summary(parts):
    """Bảng tóm tắt từng tập: số giờ, tỷ lệ, thời điểm đầu/cuối, trung bình từng chuỗi."""
    total = sum(len(p) for p in parts.values())
    rows = []
    for name, part in parts.items():
        row = {
            "split": name,
            "rows": len(part),
            "percent": round(100 * len(part) / total, 2),
            "start": part.index.min(),
            "end": part.index.max(),
        }
        for c in part.columns:
            row[f"mean_{c}"] = round(part[c].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    out_dir = path_from(cfg, "processed_dir")
    tables_dir = path_from(cfg, "tables_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_hourly_data(cfg)
    parts = dict(zip(SPLIT_NAMES, split_by_time(df, cfg)))

    for name, part in parts.items():
        part.reset_index().to_csv(out_dir / f"{name}.csv", index=False)

    summary = split_summary(parts)
    summary.to_csv(tables_dir / "split_summary.csv", index=False)

    print(f"Tổng số giờ: {len(df):,} ({df.index.min()} -> {df.index.max()})")
    print(summary.iloc[:, :5].to_string(index=False))
    print(f"\nĐã lưu: {out_dir}/train.csv, validation.csv, test.csv")
    print(f"Đã lưu: {tables_dir / 'split_summary.csv'}")
    print("Lưu ý: scaler (nếu dùng) phải được fit CHỈ trên tập train.")


if __name__ == "__main__":
    main()
