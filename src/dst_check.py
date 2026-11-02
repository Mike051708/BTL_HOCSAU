"""Liệt kê các giờ có thể bất thường quanh ngày chuyển giờ (DST) trong dữ liệu theo giờ.

Chạy: python src/dst_check.py
Kết quả: results/tables/dst_hours.csv

Theo mô tả bộ dữ liệu, ngày chuyển giờ tháng 3 (23 giờ) có giờ 01:00-02:00 bị ghi
bằng 0, còn ngày chuyển giờ tháng 10 (25 giờ) thì giờ này gộp hai giờ. Vì nhãn thời
gian là mốc kết thúc, giờ bị ảnh hưởng có nhãn 02:00. Script KHÔNG sửa dữ liệu, chỉ
thống kê để nhóm giải thích các lỗi dự báo lớn khi phân tích lỗi.


CÁCH ĐÁNH DẤU BẤT THƯỜNG - VÀ VÌ SAO LÀM THẾ

Chỉ số dùng để so sánh:

    ratio = giá trị giờ nhãn 02:00 / trung bình giá trị giờ 01:00 và 03:00

Bản trước đặt cứng một cặp ngưỡng chung cho mọi chuỗi (ratio < 0,5 hoặc > 1,3).
Cách đó có hai vấn đề:

  1. Không có cơ sở nào nói phải là 0,5 và 1,3. Hai con số do người viết tự nghĩ ra.
  2. Áp một ngưỡng chung cho mọi chuỗi là sai về nguyên tắc: chuỗi biến động mạnh
     thì ngày thường cũng vượt ngưỡng (báo nhầm), còn chuỗi rất đều thì một bất
     thường thật lại không vượt ngưỡng (bỏ sót).

Bản này lấy ngưỡng TỪ CHÍNH CHUỖI ĐÓ:

  - Tính `ratio` cho MỌI ngày trong chuỗi, không chỉ ngày chuyển giờ, được một
    phân bố khoảng 1.460 giá trị mô tả "giờ 02:00 của chuỗi này thường dao động
    quanh hàng xóm của nó như thế nào".
  - Bỏ chính các ngày chuyển giờ ra khỏi phân bố tham chiếu, nếu không chúng tự
    kéo giãn hai đuôi và tự làm mình trông bình thường.
  - Lấy phân vị p và (100 - p) của phân bố đó làm cận dưới và cận trên.
  - Ngày chuyển giờ nào có `ratio` nằm ngoài khoảng đó mới bị đánh dấu.

Nhờ vậy mỗi chuỗi có ngưỡng riêng theo độ biến động của chính nó, và câu trả lời
cho "tại sao đánh dấu giờ này" là: *vì nó bất thường so với chính chuỗi này*, chứ
không phải so với một con số nhóm tự đặt.

VÌ SAO p = 1 (phân vị 1 và 99), đặt trong configs/config.yaml mục dst.ref_percentile:

    Phân bố tham chiếu có khoảng 1.453 ngày (1.461 ngày trừ 8 ngày chuyển giờ).
    Theo định nghĩa phân vị, p% số ngày THƯỜNG cũng nằm ngoài khoảng - đó là tỷ lệ
    báo nhầm mà ta chấp nhận trước. Mỗi chuỗi có 8 ngày chuyển giờ, nên số lần báo
    nhầm kỳ vọng trên một chuỗi là:

        8 x 2p/100     (2p vì tính cả hai đuôi)

    p = 5  ->  8 x 0,10 = 0,80 lần  (gần 1, quá lỏng)
    p = 1  ->  8 x 0,02 = 0,16 lần  (dưới 1/6 lần, đủ chặt)

    Chọn p = 1 để số lần báo nhầm kỳ vọng trên mỗi chuỗi ở dưới 1. Đây là tiêu chí
    rút ra được từ số ngày có trong dữ liệu, không phải con số chọn theo cảm tính.

Trường hợp giá trị đúng bằng 0 được đánh dấu riêng, không qua phân vị: đó là biểu
hiện mà mô tả bộ dữ liệu nói rõ cho ngày chuyển giờ tháng 3, và khi hai giờ lân cận
cũng bằng 0 thì `ratio` không tính được.

Cột `reason` ghi rõ lý do đánh dấu, cùng quy ước với stage_a_reason và
stage_c_reason của src/select_customers.py: nói được "đánh dấu vì cái gì", không
chỉ True/False.
"""
import sys
from pathlib import Path

import numpy as np
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


def hour02_ratios(s: pd.Series) -> pd.Series:
    """Tỷ lệ giờ nhãn 02:00 so với trung bình giờ 01:00 và 03:00, cho MỌI ngày.

    Trả về Series có chỉ số là các mốc 02:00. Ngày nào không tính được tỷ lệ
    (thiếu giờ lân cận, hoặc hai giờ lân cận đều bằng 0) thì để NaN.
    """
    at02 = s[s.index.hour == 2]
    prev = s.reindex(at02.index - pd.Timedelta(hours=1)).to_numpy(dtype=float)
    nxt = s.reindex(at02.index + pd.Timedelta(hours=1)).to_numpy(dtype=float)
    denom = (prev + nxt) / 2.0
    ratio = np.divide(
        at02.to_numpy(dtype=float), denom,
        out=np.full(len(at02), np.nan), where=denom > 0,
    )
    return pd.Series(ratio, index=at02.index)


def reference_bounds(s: pd.Series, cfg) -> tuple:
    """Cận dưới/cận trên lấy từ phân bố tỷ lệ của CHÍNH chuỗi này.

    Bỏ các ngày chuyển giờ ra khỏi phân bố tham chiếu: chúng là thứ ta đang đi tìm,
    để lại thì chúng tự nới rộng hai đuôi và tự che mình.
    """
    p = float(cfg["dst"]["ref_percentile"])
    ratios = hour02_ratios(s)
    dst_ts = {ts for ts, _ in dst_timestamps(cfg)}
    ref = ratios.drop(index=[t for t in dst_ts if t in ratios.index]).dropna()
    if len(ref) < 100:
        # Quá ít ngày tham chiếu thì phân vị không đáng tin; không đánh dấu theo tỷ lệ.
        return np.nan, np.nan, len(ref)
    return float(np.percentile(ref, p)), float(np.percentile(ref, 100 - p)), len(ref)


def dst_table(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])

    bounds = {c: reference_bounds(df[c].astype(float), cfg) for c in df.columns}

    rows = []
    for ts, kind in dst_timestamps(cfg):
        split = "train" if ts <= train_end else "validation" if ts <= val_end else "test"
        for c in df.columns:
            low, high, n_ref = bounds[c]
            prev, cur, nxt = (
                float(df.loc[ts + pd.Timedelta(hours=k), c]) for k in (-1, 0, 1)
            )
            denom = (prev + nxt) / 2.0
            ratio = cur / denom if denom > 0 else np.nan

            # Giá trị đúng bằng 0: biểu hiện mà mô tả bộ dữ liệu nói rõ cho tháng 3.
            # Xét riêng vì khi hai giờ lân cận cũng bằng 0 thì ratio không tính được.
            if cur == 0:
                reason = "giá trị bằng 0"
            elif np.isnan(ratio) or np.isnan(low):
                reason = ""
            elif ratio < low:
                reason = "thấp bất thường so với chính chuỗi này"
            elif ratio > high:
                reason = "cao bất thường so với chính chuỗi này"
            else:
                reason = ""

            rows.append({
                "date": ts.date(), "kind": kind, "hour_label": ts, "split": split,
                "customer": c,
                "value_01h": round(prev, 2), "value_02h": round(cur, 2),
                "value_03h": round(nxt, 2),
                "ratio_vs_neighbors": round(ratio, 2) if not np.isnan(ratio) else np.nan,
                # In kèm ngưỡng để người đọc tự kiểm chứng, không phải tin suông
                "ref_low": round(low, 2) if not np.isnan(low) else np.nan,
                "ref_high": round(high, 2) if not np.isnan(high) else np.nan,
                "n_ngay_tham_chieu": n_ref,
                "suggest_check": bool(reason),
                "reason": reason,
            })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    df = load_hourly_data(cfg)
    table = dst_table(df, cfg)

    out = path_from(cfg, "tables_dir")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "dst_hours.csv", index=False, encoding="utf-8-sig")

    p = cfg["dst"]["ref_percentile"]
    main_c = cfg["customers"]["main"]
    print(f"Đã lưu: {out / 'dst_hours.csv'}")
    print(f"\nNgưỡng lấy từ phân vị {p} và {100 - p} của chính từng chuỗi "
          f"(số lần báo nhầm kỳ vọng mỗi chuỗi: 8 x {2 * p / 100:.2f} = "
          f"{8 * 2 * p / 100:.2f} lần).")

    print(f"\nChuỗi chính {main_c}:")
    cols = ["date", "kind", "split", "value_01h", "value_02h", "value_03h",
            "ratio_vs_neighbors", "ref_low", "ref_high", "reason"]
    print(table[table.customer == main_c][cols].to_string(index=False))

    print("\nSố ngày chuyển giờ được gợi ý kiểm tra (theo chuỗi):")
    print(table.groupby("customer")["suggest_check"].sum().to_string())

    print("\nChưa sửa dữ liệu: ghi lại các mốc này trong báo cáo khi phân tích lỗi.")


if __name__ == "__main__":
    main()
