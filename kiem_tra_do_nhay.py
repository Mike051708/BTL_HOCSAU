"""Kiểm tra độ nhạy của các ngưỡng chọn chuỗi.

VÌ SAO CẦN SCRIPT NÀY

Sáu ngưỡng ở tầng C và D1 (max_const_run_h, cv_min, cv_max, min_ac_lag168,
max_year_drift, max_split_shift, main_max_ac_lag24) là những con số do nhóm
chọn. Không có cơ sở lý thuyết nào nói phải là 0,80 chứ không phải 0,75.

Khi không chứng minh được một con số là ĐÚNG, cách làm khoa học là chứng minh
KẾT LUẬN KHÔNG PHỤ THUỘC vào con số đó. Script này đổi từng ngưỡng một, giữ
nguyên các ngưỡng còn lại, rồi xem chuỗi chính được chọn có đổi theo không.

    - Chuỗi chính giữ nguyên ở hầu hết các bộ ngưỡng
      -> lựa chọn vững, con số cụ thể không quan trọng.
    - Chuỗi chính đổi liên tục
      -> lựa chọn nhạy với ngưỡng, PHẢI ghi điều này vào phần hạn chế
         của báo cáo thay vì giấu đi.

Đây là phép thử một-lần-một-biến (one-at-a-time). Nó không bao phủ mọi tổ hợp,
nhưng đủ để trả lời câu hỏi "nếu thầy cô đổi ngưỡng thì kết quả có đổi không".

Chạy từ thư mục gốc dự án, sau khi đã có cache:
    python kiem_tra_do_nhay.py
"""
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config                       # noqa: E402
from src.select_customers import (                       # noqa: E402
    apply_filters,
    hourly_features,
    pick_main,
)

# Mỗi ngưỡng thử các giá trị quanh giá trị đang dùng, rộng hẹp hơn rõ rệt
BIEN_THE = {
    "max_const_run_h":    [3, 4, 5, 6, 8, 12],
    "cv_min":             [0.15, 0.20, 0.25, 0.30, 0.35],
    "cv_max":             [0.70, 0.80, 0.85, 0.95, 1.20],
    "min_ac_lag168":      [0.65, 0.70, 0.75, 0.80, 0.85],
    "max_year_drift":     [0.25, 0.30, 0.35, 0.45, 0.60],
    "max_split_shift":    [0.15, 0.20, 0.25, 0.35, 0.50],
    "main_max_ac_lag24":  [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
}


def main():
    cfg = load_config()
    cache = ROOT / cfg["paths"]["candidates_hourly_file"]
    if not cache.exists():
        sys.exit(f"Chưa có {cache}. Chạy: python src/select_customers.py")

    print(f"Đọc cache: {cache.name}")
    hourly = pd.read_csv(cache, parse_dates=["timestamp"]).set_index("timestamp")
    print(f"  {hourly.shape[1]} chuỗi, {hourly.shape[0]:,} giờ")

    # Đặc trưng chỉ phụ thuộc dữ liệu, không phụ thuộc ngưỡng -> tính MỘT lần
    print("Tính đặc trưng tầng B...")
    feat = hourly_features(hourly, cfg)

    goc = cfg["selection"]
    chuoi_goc, _ = pick_main(
        apply_filters(feat, cfg).pipe(lambda d: d[d.stage_c_pass]), cfg
    )
    print(f"\nChuỗi chính với bộ ngưỡng đang dùng: {chuoi_goc}\n")

    ket_qua = []
    for ten_nguong, gia_tri_list in BIEN_THE.items():
        for gt in gia_tri_list:
            cfg_thu = copy.deepcopy(cfg)
            cfg_thu["selection"][ten_nguong] = gt

            loc = apply_filters(feat, cfg_thu)
            shortlist = loc[loc["stage_c_pass"]].sort_values(
                "ac_lag168", ascending=False
            )
            if shortlist.empty:
                chuoi, n_ung_vien = "(rỗng)", 0
            else:
                chuoi, du_dieu_kien = pick_main(shortlist, cfg_thu)
                n_ung_vien = len(du_dieu_kien)

            ket_qua.append({
                "ngưỡng": ten_nguong,
                "giá trị": gt,
                "đang dùng": "<--" if gt == goc[ten_nguong] else "",
                "qua tầng C": len(shortlist),
                "ứng viên chính": n_ung_vien,
                "chuỗi chính": chuoi,
                "khớp gốc": "có" if chuoi == chuoi_goc else "KHÁC",
            })

    bang = pd.DataFrame(ket_qua)
    for ten in BIEN_THE:
        print(bang[bang["ngưỡng"] == ten].to_string(index=False))
        print()

    n_khop = int((bang["chuỗi chính"] == chuoi_goc).sum())
    print("=" * 62)
    print(f"{n_khop}/{len(bang)} bộ ngưỡng cho ra chuỗi chính {chuoi_goc}")
    khac = bang[bang["chuỗi chính"] != chuoi_goc]
    if khac.empty:
        print("Không bộ ngưỡng nào làm đổi chuỗi chính -> lựa chọn KHÔNG nhạy với ngưỡng.")
    else:
        print("\nCác bộ ngưỡng cho ra chuỗi khác:")
        print(khac[["ngưỡng", "giá trị", "chuỗi chính"]].to_string(index=False))
        print("\nGhi những trường hợp này vào phần hạn chế của báo cáo.")

    out = ROOT / cfg["paths"]["tables_dir"] / "do_nhay_nguong.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    bang.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu bảng đầy đủ: {out}")


if __name__ == "__main__":
    main()
