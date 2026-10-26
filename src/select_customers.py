"""Chọn chuỗi khách hàng theo tiêu chí công khai, tái lập được.

Thay thế cho src/check_customers.py (bảng top10 cũ sắp xếp theo ba cột đều
bằng 0 nên không sắp xếp được gì, danh sách 5 chuỗi là ngẫu nhiên theo thứ tự
cột trong file).

Quy trình 4 tầng, mọi ngưỡng đặt trong configs/config.yaml (mục `selection`):

    Tầng A  370 -> ?   Chất lượng dữ liệu thô (thiếu, âm, bắt đầu muộn, 0 lẻ)
    Tầng B  ?   -> ?   Tính đặc trưng chuỗi theo giờ cho các chuỗi qua tầng A
    Tầng C  ?   -> N   Lọc theo đặc trưng (đoạn phẳng, cv, chu kỳ, dịch mức)
    Tầng D1 N   -> 1   Chọn CHUỖI CHÍNH trên toàn bộ shortlist (làm TRƯỚC)
    Tầng D2 N   -> 4   Chọn chuỗi phụ trải đều theo mức tải

Thứ tự D1 trước D2 là bắt buộc: phần lớn khách hàng trong bộ dữ liệu có chu kỳ
ngày lặp gần như máy móc (ac_lag24 ~ 0,95), nên nếu chọn 5 chuỗi theo mức tải
trước thì hầu như chắc chắn không chuỗi nào trong đó còn dư địa cho mô hình học
sâu.

Chạy:
    python src/select_customers.py            # đọc data/raw, tạo cache theo giờ
    python src/select_customers.py --cached   # dùng lại cache, chạy lại tầng C-D

Đầu ra (results/tables/):
    customer_quality_all.csv       370 chuỗi, kết quả tầng A
    customer_features_hourly.csv   các chuỗi qua tầng A, đặc trưng tầng B
    customer_shortlist.csv         các chuỗi qua tầng C, đã xếp hạng
    customer_main_candidates.csv   các chuỗi đủ điều kiện làm chuỗi chính (tầng D1)
    customer_selection.csv         5 chuỗi được chọn + vai trò (chính/phụ)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import ROOT_DIR, load_config, path_from
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ROOT_DIR, load_config, path_from


CACHE_PATH = "data/processed/candidates_hourly.csv"


# ==========================================================================
# TẦNG A - CHẤT LƯỢNG DỮ LIỆU THÔ (trên dữ liệu 15 phút)
# ==========================================================================

def leading_zero_count(values: np.ndarray) -> int:
    """Số mốc bằng 0 liên tiếp ở ĐẦU chuỗi.

    Theo mô tả của bộ dữ liệu, đoạn 0 ở đầu là giai đoạn khách hàng chưa có
    dữ liệu, không phải tiêu thụ bằng 0. Tách riêng khỏi 'số giá trị 0 nằm
    giữa chuỗi' vì hai hiện tượng này có ý nghĩa khác nhau.
    """
    nonzero = np.flatnonzero(values != 0)
    return int(len(values)) if nonzero.size == 0 else int(nonzero[0])


def quality_report(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    """Tầng A: thống kê chất lượng cho toàn bộ 370 chuỗi 15 phút."""
    sel = cfg["selection"]
    rows = []

    for col in raw.columns:
        values = raw[col].to_numpy(dtype="float64")
        lead = leading_zero_count(values)
        zeros_total = int((values == 0).sum())

        rows.append({
            "customer": col,
            "records_15min": len(values),
            "missing_count": int(np.isnan(values).sum()),
            "negative_count": int((values < 0).sum()),
            "leading_zero_steps": lead,
            "leading_zero_days": round(lead / 96, 1),
            # 0 nằm giữa chuỗi: khác hẳn với 0 do bắt đầu muộn
            "zeros_after_start": zeros_total - lead,
            "mean_15min": float(np.nanmean(values)),
        })

    rep = pd.DataFrame(rows).set_index("customer")

    rep["pass_no_missing"] = rep["missing_count"] == 0
    rep["pass_no_negative"] = rep["negative_count"] == 0
    rep["pass_full_from_2011"] = rep["leading_zero_steps"] <= sel["max_leading_zero_steps"]
    rep["pass_no_inner_zero"] = rep["zeros_after_start"] <= sel["max_zeros_after_start"]
    rep["pass_positive_mean"] = rep["mean_15min"] > 0

    pass_cols = [c for c in rep.columns if c.startswith("pass_")]
    rep["stage_a_pass"] = rep[pass_cols].all(axis=1)
    # Lý do trượt, để giải thích trong báo cáo thay vì chỉ nói "bị loại"
    rep["stage_a_reason"] = rep[pass_cols].apply(
        lambda r: "" if r.all() else ", ".join(c[5:] for c in pass_cols if not r[c]),
        axis=1,
    )
    return rep


# ==========================================================================
# TẦNG B - ĐẶC TRƯNG CHUỖI THEO GIỜ
# ==========================================================================

def to_hourly(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    """Gộp 15 phút -> 1 giờ ĐÚNG quy ước của src/data_preprocessing.py.

    Trung bình 4 mốc, nhãn là mốc kết thúc của giờ (closed='right',
    label='right'). Giữ nguyên quy ước để mọi bảng so sánh được với nhau.
    """
    hourly = raw.resample("1h", closed="right", label="right").mean()
    counts = raw.resample("1h", closed="right", label="right").count()
    incomplete = int((counts != 4).any(axis=1).sum())
    if incomplete:
        print(f"  Cảnh báo: {incomplete} giờ không đủ 4 mốc 15 phút.")
    return hourly


def longest_constant_run(s: pd.Series):
    """(đoạn không đổi dài nhất, số đoạn không đổi >= 6 giờ).

    Chuỗi có đoạn phẳng dài là chuỗi ít thông tin để mô hình học (ví dụ
    đồng hồ hỏng, hoặc giá trị bị giữ nguyên).
    """
    same = s.diff() == 0
    if not same.any():
        return 0, 0
    runs = same.groupby((~same).cumsum()).sum()
    return int(runs.max()), int((runs >= 6).sum())


def hourly_features(hourly: pd.DataFrame, cfg) -> pd.DataFrame:
    """Tầng B: đặc trưng mô tả hình dạng chuỗi (chưa lọc, chưa chấm điểm)."""
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])
    # Nhãn là mốc kết thúc nên giờ 00:00 ngày 01/01/2015 vẫn thuộc năm 2014
    year = (hourly.index - pd.Timedelta(hours=1)).year

    rows = []
    for col in hourly.columns:
        s = hourly[col].astype(float)
        run_max, run6 = longest_constant_run(s)

        by_year = s.groupby(year).mean()
        drift = float((by_year.max() - by_year.min()) / s.mean())

        tr = s.loc[s.index <= train_end]
        va = s.loc[(s.index > train_end) & (s.index <= val_end)]
        te = s.loc[s.index > val_end]

        is_weekend = s.index.dayofweek >= 5
        wd, we = s[~is_weekend].mean(), s[is_weekend].mean()

        rows.append({
            "customer": col,
            "mean_kW": s.mean(),
            "std_kW": s.std(),
            "cv": s.std() / s.mean(),
            "max_kW": s.max(),
            "const_run_max_h": run_max,
            "const_runs_ge_6h": run6,
            "ac_lag1": s.autocorr(1),
            "ac_lag24": s.autocorr(24),
            "ac_lag168": s.autocorr(168),
            "year_drift": drift,
            # Lệch mức giữa các tập: ảnh hưởng trực tiếp tới sai số test
            "shift_val_vs_train": abs(va.mean() - tr.mean()) / tr.mean(),
            "shift_test_vs_train": abs(te.mean() - tr.mean()) / tr.mean(),
            "weekday_weekend_ratio": wd / we if we > 0 else np.nan,
        })

    return pd.DataFrame(rows).set_index("customer")


# ==========================================================================
# TẦNG C - LỌC THEO NGƯỠNG CÔNG KHAI
# ==========================================================================

def apply_filters(feat: pd.DataFrame, cfg) -> pd.DataFrame:
    """Tầng C: áp ngưỡng trong config. Mỗi ngưỡng thành một cột pass_* riêng
    để báo cáo nói được 'loại vì tiêu chí nào', không chỉ 'bị loại'."""
    sel = cfg["selection"]
    out = feat.copy()

    out["pass_no_flat"] = out["const_run_max_h"] <= sel["max_const_run_h"]
    out["pass_cv"] = out["cv"].between(sel["cv_min"], sel["cv_max"])
    out["pass_weekly_cycle"] = out["ac_lag168"] >= sel["min_ac_lag168"]
    out["pass_stable_level"] = out["year_drift"] <= sel["max_year_drift"]
    out["pass_small_shift"] = out["shift_test_vs_train"] <= sel["max_split_shift"]

    pass_cols = [c for c in out.columns if c.startswith("pass_")]
    out["stage_c_pass"] = out[pass_cols].all(axis=1)
    out["stage_c_reason"] = out[pass_cols].apply(
        lambda r: "" if r.all() else ", ".join(c[5:] for c in pass_cols if not r[c]),
        axis=1,
    )
    return out


# ==========================================================================
# TẦNG D - CHỌN 5 CHUỖI ĐẠI DIỆN VÀ CHUỖI CHÍNH
# ==========================================================================

def pick_main(shortlist: pd.DataFrame, cfg):
    """Chọn chuỗi CHÍNH trên TOÀN BỘ shortlist (phải gọi TRƯỚC pick_secondary).

    Quy tắc: trong số các chuỗi mà Seasonal Naive 24 giờ CHƯA gần như tối ưu
    (ac_lag24 < ngưỡng), lấy chuỗi có chu kỳ tuần rõ nhất (ac_lag168 cao nhất).

    Lý do: phần lớn khách hàng trong bộ dữ liệu có ac_lag24 rất cao (0,93-0,96),
    nghĩa là "giá trị cùng giờ hôm qua" đã bám sát chuỗi và mô hình học sâu gần
    như không còn dư địa cải thiện - bài toán mất ý nghĩa thực nghiệm. Chuỗi
    chính phải là chuỗi còn chỗ cho mô hình thể hiện.

    Quy tắc đặt TRƯỚC khi chạy mô hình, không dựa trên kết quả dự báo.

    Trả về (tên_chuỗi, bảng các chuỗi đủ điều kiện làm chuỗi chính).
    """
    cap = cfg["selection"]["main_max_ac_lag24"]
    eligible = shortlist[shortlist["ac_lag24"] < cap].sort_values(
        "ac_lag168", ascending=False
    )
    if eligible.empty:
        print(f"  Cảnh báo: không chuỗi nào có ac_lag24 < {cap}. "
              f"Lấy chuỗi có ac_lag24 thấp nhất và GHI RÕ ngoại lệ này trong báo cáo.")
        eligible = shortlist.nsmallest(1, "ac_lag24")
    return eligible.index[0], eligible


def pick_secondary(shortlist: pd.DataFrame, main: str, k: int) -> pd.DataFrame:
    """Chọn k-1 chuỗi phụ trải đều theo mức tải, KHÔNG tính chuỗi chính.

    Lấy các phân vị đều nhau của mean_kW rồi chọn chuỗi gần nhất mỗi phân vị.
    Mục đích: các chuỗi phụ đại diện cho những quy mô khách hàng khác nhau
    (kế hoạch yêu cầu "2 chuỗi phụ khác mức tải để kiểm tra tính tổng quát"),
    không phải k chuỗi giống hệt nhau. Quy tắc cố định nên chạy lại cho cùng
    kết quả.
    """
    pool = shortlist.drop(index=main, errors="ignore")
    n = k - 1
    if n <= 0 or pool.empty:
        return pool.iloc[:0]
    if len(pool) <= n:
        return pool.copy()

    ranked = pool.sort_values("mean_kW")
    targets = ranked["mean_kW"].quantile(np.linspace(0.1, 0.9, n))

    chosen = []
    for t in targets:
        rest = ranked.drop(index=chosen)
        chosen.append((rest["mean_kW"] - t).abs().idxmin())
    return shortlist.loc[chosen].sort_values("mean_kW")


# ==========================================================================
# CHẠY CHÍNH
# ==========================================================================

def load_raw(cfg) -> pd.DataFrame:
    """Đọc LD2011_2014.txt (~700 MB). Dùng float32 để giảm bộ nhớ."""
    path = path_from(cfg, "raw_file")
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Chạy: python src/download_data.py"
        )
    print(f"Đang đọc {path.name} (~700 MB), vui lòng chờ...")
    raw = pd.read_csv(
        path,
        sep=cfg["data"]["raw_separator"],
        decimal=cfg["data"]["raw_decimal"],
        index_col=0,
        parse_dates=True,
        low_memory=False,
    )
    raw.index.name = "timestamp"
    raw.columns = raw.columns.str.strip().str.strip('"')
    return raw.apply(pd.to_numeric, errors="coerce").astype("float32").sort_index()


def main(use_cache=False):
    cfg = load_config()
    tables = path_from(cfg, "tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    cache = ROOT_DIR / CACHE_PATH

    if use_cache:
        if not cache.exists():
            raise FileNotFoundError(f"Chưa có cache {cache}. Chạy lại không có --cached.")
        print(f"Dùng cache: {CACHE_PATH}")
        hourly = pd.read_csv(cache, parse_dates=["timestamp"]).set_index("timestamp")
        quality = pd.read_csv(
            tables / "customer_quality_all.csv", encoding="utf-8-sig"
        ).set_index("customer")
    else:
        raw = load_raw(cfg)
        print(f"Dữ liệu thô: {len(raw):,} mốc x {len(raw.columns)} khách hàng")

        print("\n[Tầng A] Chất lượng dữ liệu thô...")
        quality = quality_report(raw, cfg)
        quality.to_csv(tables / "customer_quality_all.csv", encoding="utf-8-sig")
        passers = quality.index[quality["stage_a_pass"]].tolist()
        print(f"  {len(raw.columns)} -> {len(passers)} chuỗi qua tầng A")
        for reason, n in quality.loc[~quality["stage_a_pass"], "stage_a_reason"] \
                                .value_counts().head(5).items():
            print(f"    loại {n:>3} chuỗi vì: {reason}")
        if not passers:
            raise RuntimeError("Không chuỗi nào qua tầng A; xem lại ngưỡng trong config.")

        print("\n[Tầng B] Gộp theo giờ các chuỗi qua tầng A...")
        hourly = to_hourly(raw[passers], cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        hourly.reset_index().to_csv(cache, index=False)
        print(f"  Đã lưu cache: {CACHE_PATH} ({hourly.shape[0]:,} giờ x {hourly.shape[1]})")

    feat = hourly_features(hourly, cfg)
    feat.to_csv(tables / "customer_features_hourly.csv", encoding="utf-8-sig")

    print("\n[Tầng C] Lọc theo đặc trưng chuỗi...")
    sel = cfg["selection"]
    print(f"  Ngưỡng: đoạn phẳng <= {sel['max_const_run_h']}h, "
          f"cv trong [{sel['cv_min']}, {sel['cv_max']}], "
          f"ac_lag168 >= {sel['min_ac_lag168']},")
    print(f"          dịch mức theo năm <= {sel['max_year_drift']}, "
          f"lệch test/train <= {sel['max_split_shift']}")

    filtered = apply_filters(feat, cfg)
    shortlist = filtered[filtered["stage_c_pass"]].sort_values("ac_lag168", ascending=False)
    shortlist.to_csv(tables / "customer_shortlist.csv", encoding="utf-8-sig")
    print(f"  {len(feat)} -> {len(shortlist)} chuỗi qua tầng C")
    for reason, n in filtered.loc[~filtered["stage_c_pass"], "stage_c_reason"] \
                             .value_counts().head(5).items():
        print(f"    loại {n:>3} chuỗi vì: {reason}")
    if shortlist.empty:
        raise RuntimeError("Không chuỗi nào qua tầng C; nới ngưỡng trong config.")

    cols = ["role", "mean_kW", "cv", "const_run_max_h",
            "ac_lag24", "ac_lag168", "year_drift", "shift_test_vs_train"]

    # --- Tầng D1: chuỗi chính, chọn trên TOÀN BỘ shortlist ---
    cap = sel["main_max_ac_lag24"]
    print(f"\n[Tầng D1] Chuỗi chính: ac_lag168 cao nhất trong nhóm ac_lag24 < {cap}...")
    main_series, eligible = pick_main(shortlist, cfg)
    print(f"  {len(eligible)}/{len(shortlist)} chuỗi đủ điều kiện làm chuỗi chính"
          f" (phần lớn chuỗi có chu kỳ ngày lặp gần như máy móc).")
    print(eligible.head(5)[["mean_kW", "cv", "ac_lag24", "ac_lag168"]].round(3).to_string())
    eligible[["mean_kW", "cv", "ac_lag24", "ac_lag168", "year_drift",
              "shift_test_vs_train"]].to_csv(
        tables / "customer_main_candidates.csv", encoding="utf-8-sig")
    print(f"  -> Chuỗi chính: {main_series}")

    # --- Tầng D2: các chuỗi phụ, trải đều theo mức tải ---
    k = sel["n_representatives"]
    print(f"\n[Tầng D2] {k - 1} chuỗi phụ trải đều theo mức tải...")
    secondary = pick_secondary(shortlist, main_series, k)

    selected = pd.concat([shortlist.loc[[main_series]], secondary])
    selected = selected.copy()
    selected["role"] = "phụ"
    selected.loc[main_series, "role"] = "CHÍNH"
    selected = selected.sort_values("mean_kW")
    selected[cols].to_csv(tables / "customer_selection.csv", encoding="utf-8-sig")
    print(selected[cols].round(3).to_string())

    # --- Đối chiếu với lựa chọn đang ghi trong config ---
    cur_main = cfg["customers"]["main"]
    cur_all = cfg["customers"]["hourly_file_columns"]
    print("\n[Đối chiếu với configs/config.yaml]")
    print(f"  Đang dùng: chính={cur_main}, 5 chuỗi={cur_all}")
    if cur_main in eligible.index:
        rank = list(eligible.index).index(cur_main) + 1
        print(f"  {cur_main} đủ điều kiện làm chuỗi chính "
              f"(hạng {rank}/{len(eligible)} theo ac_lag168 trong nhóm ac_lag24 < {cap})"
              f" -> giữ được, nêu tiêu chí này trong báo cáo.")
    elif cur_main in shortlist.index:
        print(f"  {cur_main} qua tầng C nhưng ac_lag24 = "
              f"{filtered.loc[cur_main, 'ac_lag24']:.3f} >= {cap}: Seasonal Naive 24h"
              f" đã gần tối ưu -> cân nhắc đổi sang {main_series}.")
    else:
        why = filtered.loc[cur_main, "stage_c_reason"] if cur_main in filtered.index else "trượt tầng A"
        print(f"  {cur_main} KHÔNG qua tầng C ({why}) -> cần đổi chuỗi chính"
              f" hoặc giải trình rõ lý do giữ lại.")
    for c in cur_all:
        if c in filtered.index and not filtered.loc[c, "stage_c_pass"]:
            print(f"    {c}: loại vì {filtered.loc[c, 'stage_c_reason']}")

    print(f"\nĐã lưu 4 bảng vào {tables}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cached", action="store_true",
                    help="dùng lại data/processed/candidates_hourly.csv, bỏ qua tầng A")
    main(use_cache=ap.parse_args().cached)
