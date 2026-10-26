"""Liệt kê các giờ có thể bất thường quanh ngày chuyển giờ (DST) trong dữ liệu theo giờ.

Chạy: python src/dst_check.py
Kết quả: results/tables/dst_hours.csv

Theo mô tả bộ dữ liệu, ngày chuyển giờ tháng 3 (23 giờ) có giờ 01:00-02:00 bị ghi
bằng 0, còn ngày chuyển giờ tháng 10 (25 giờ) thì giờ này gộp hai giờ. Vì nhãn thời
gian là mốc kết thúc, giờ bị ảnh hưởng có nhãn 02:00. Script KHÔNG sửa dữ liệu, chỉ
thống kê để nhóm giải thích các lỗi dự báo lớn khi phân tích lỗi.

Cột ratio_vs_neighbors = giá trị giờ 02:00 / trung bình giờ 01:00 và 03:00.
Cột suggest_check = True nếu ratio < 0.5 hoặc > 1.3 (chỉ là gợi ý: với các chuỗi
biến động mạnh như MT_158 thì ngày thường cũng có lúc đạt ngưỡng này).
"""
import sys
from pathlib import Path

import pandas as pd

try:
    from src.config import load_config, path_from
    from src.data import load_hourly_data
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import load_config, path_from
    from data import load_hourly_data


def dst_timestamps(cfg=None):
    """Trả về list[(Timestamp giờ 02:00, 'march'|'october')] cho mọi ngày chuyển giờ."""
    cfg = cfg or load_config()
    items = []
    for kind in ("march", "october"):
        for d in cfg["dst"][kind]:
            items.append((pd.Timestamp(d) + pd.Timedelta(hours=2), kind))
    return sorted(items)


def dst_table(df, cfg=None):
    cfg = cfg or load_config()
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    rows = []
    for ts, kind in dst_timestamps(cfg):
        split = "train" if ts <= train_end else "validation" if ts <= val_end else "test"
        for c in df.columns:
            prev, cur, nxt = (df.loc[ts + pd.Timedelta(hours=k), c] for k in (-1, 0, 1))
            ratio = cur / ((prev + nxt) / 2)
            rows.append({
                "date": ts.date(), "kind": kind, "hour_label": ts, "split": split,
                "customer": c, "value_01h": round(prev, 2), "value_02h": round(cur, 2),
                "value_03h": round(nxt, 2), "ratio_vs_neighbors": round(ratio, 2),
                "suggest_check": bool(ratio < 0.5 or ratio > 1.3),
            })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    df = load_hourly_data(cfg)
    table = dst_table(df, cfg)
    out = path_from(cfg, "tables_dir")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "dst_hours.csv", index=False)
    main_c = cfg["customers"]["main"]
    print(f"Đã lưu: {out / 'dst_hours.csv'}")
    print(f"\nChuỗi chính {main_c}:")
    print(table[table.customer == main_c].drop(columns="customer").to_string(index=False))
    print("\nSố ngày chuyển giờ được gợi ý kiểm tra (theo chuỗi):")
    print(table.groupby("customer")["suggest_check"].sum().to_string())
    print("\nChưa sửa dữ liệu: ghi lại các mốc này trong báo cáo khi phân tích lỗi.")


if __name__ == "__main__":
    main()
