"""Biểu đồ cho phần HUẤN LUYỆN của một mô hình (mặc định LSTM).

Khác với src/error_analysis.py (so sánh giữa các mô hình, phân tích lỗi),
script này chỉ nói về QUÁ TRÌNH HUẤN LUYỆN của một mô hình:
learning curve, learning rate, hội tụ, dò tham số, độ ổn định giữa các seed.

Dùng cho mục "Mô hình LSTM" trong báo cáo. Bảo Quốc dùng lại y hệt cho MLP
bằng cách thêm --model MLP.

CHẠY:
    python src/training_figures.py                  # LSTM
    python src/training_figures.py --model MLP      # MLP
    python src/training_figures.py --model LSTM --smooth 3

ĐẦU VÀO:
    results/dl/<model>/tables/history_<chuoi>_s<seed>.csv
    results/dl/<model>/tables/tuning_log.csv          (nếu có)
    results/experiments.csv

ĐẦU RA:
    results/analysis_train/<model>/figures/*.png
    results/analysis_train/<model>/tables/*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.config import ROOT_DIR, load_config
    from src.error_analysis import chi_giu_cau_hinh_da_do
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import ROOT_DIR, load_config
    from src.error_analysis import chi_giu_cau_hinh_da_do

DPI = 130
plt.rcParams.update({
    "figure.autolayout": True, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "font.size": 10,
})


# ======================================================================
# ĐỌC DỮ LIỆU
# ======================================================================

def doc_history(model, cac_chuoi):
    """Đọc mọi history_<chuoi>_s<seed>.csv -> DataFrame dài."""
    thu_muc = ROOT_DIR / "results" / "dl" / model.lower() / "tables"
    phan = []
    for c in cac_chuoi:
        for t in sorted(thu_muc.glob(f"history_{c}_s*.csv")):
            d = pd.read_csv(t, encoding="utf-8-sig")
            d["customer"] = c
            d["seed"] = int(t.stem.split("_s")[-1])
            phan.append(d)
        # tương thích với tên file cũ không có hậu tố seed
        t = thu_muc / f"history_{c}.csv"
        if t.exists():
            d = pd.read_csv(t, encoding="utf-8-sig")
            d["customer"] = c
            d["seed"] = 42
            phan.append(d)
    if not phan:
        return None
    h = pd.concat(phan, ignore_index=True)
    if "lr" not in h.columns:        # file cũ chưa ghi learning rate
        h["lr"] = np.nan
    return h


# ======================================================================
# HÌNH 01 - Learning curve gộp các seed
# ======================================================================

def hinh_01(model, h, cac_chuoi, smooth=1):
    """Train/validation loss trung bình ± độ lệch chuẩn qua các seed.

    Vẽ dải ± std thay vì 5 đường rời: nhìn ra ngay mức dao động giữa các seed,
    và đây là hình nên đưa vào báo cáo thay cho 20 hình đơn lẻ.
    """
    for c in cac_chuoi:
        d = h[h.customer == c]
        if d.empty:
            continue
        g = (d.groupby("epoch")
               .agg(tr=("train_mse_scaled", "mean"), tr_sd=("train_mse_scaled", "std"),
                    va=("validation_mse_scaled", "mean"), va_sd=("validation_mse_scaled", "std"),
                    lr=("lr", "mean"), n=("seed", "count")).reset_index())
        # chỉ giữ các epoch mà ĐA SỐ seed còn chạy, tránh đuôi nhiễu do early stopping
        g = g[g.n >= max(1, d.seed.nunique() // 2)]
        if smooth > 1:
            for col in ("tr", "va"):
                g[col] = g[col].rolling(smooth, center=True, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(g.epoch, g.tr, lw=1.8, label="Train loss (MSE)")
        ax.fill_between(g.epoch, g.tr - g.tr_sd, g.tr + g.tr_sd, alpha=0.18)
        ax.plot(g.epoch, g.va, lw=1.8, label="Validation loss (MSE)")
        ax.fill_between(g.epoch, g.va - g.va_sd, g.va + g.va_sd, alpha=0.18)

        # best epoch trung bình qua các seed
        be = d.loc[d.groupby("seed")["validation_mse_scaled"].idxmin(), "epoch"]
        ax.axvline(be.mean(), ls="--", c="grey", lw=1.3,
                   label=f"best epoch TB = {be.mean():.0f} ± {be.std():.0f}")
        ax.set(xlabel="Epoch", ylabel="MSE (thang chuẩn hoá)",
               title=f"{model} — {c}: learning curve, trung bình ± độ lệch chuẩn "
                     f"qua {d.seed.nunique()} seed")

        if g.lr.notna().any():
            ax2 = ax.twinx()
            ax2.plot(g.epoch, g.lr, c="tab:green", alpha=0.5, lw=1.2)
            ax2.set_yscale("log")
            ax2.set_ylabel("Learning rate (trục log)", color="tab:green")
            ax2.grid(False)
        ax.legend(fontsize=9, loc="upper right")
        fig.savefig(FIG / f"01_learning_curve_{c}.png", dpi=DPI)
        plt.close(fig)


# ======================================================================
# HÌNH 02 - Bốn chuỗi trên cùng một hình
# ======================================================================

def hinh_02(model, h, cac_chuoi, smooth=1):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, c in zip(axes.ravel(), cac_chuoi):
        d = h[h.customer == c]
        if d.empty:
            ax.set_visible(False); continue
        g = (d.groupby("epoch")
               .agg(tr=("train_mse_scaled", "mean"),
                    va=("validation_mse_scaled", "mean"),
                    n=("seed", "count")).reset_index())
        g = g[g.n >= max(1, d.seed.nunique() // 2)]
        if smooth > 1:
            for col in ("tr", "va"):
                g[col] = g[col].rolling(smooth, center=True, min_periods=1).mean()
        ax.plot(g.epoch, g.tr, lw=1.5, label="Train")
        ax.plot(g.epoch, g.va, lw=1.5, label="Validation")
        ty = g.va.iloc[-1] / g.tr.iloc[-1]
        ax.set(xlabel="Epoch", ylabel="MSE (chuẩn hoá)",
               title=f"{c}  —  val/train cuối = {ty:.2f}")
        ax.legend(fontsize=8)
    fig.suptitle(f"{model} — learning curve 4 chuỗi "
                 f"(val/train > 1,3 là dấu hiệu overfitting)", y=1.0, fontsize=12)
    fig.savefig(FIG / "02_learning_curve_tong_hop.png", dpi=DPI)
    plt.close(fig)


# ======================================================================
# HÌNH 03 - Quá trình dò tham số
# ======================================================================

def _nhom_so_sanh(t, ts, thamso):
    """Lấy nhóm cấu hình chỉ khác nhau ĐÚNG ở tham số `ts`.

    Dò theo coordinate descent nghĩa là mỗi tham số được quét khi các tham số
    khác giữ nguyên. Nếu gộp bừa mọi dòng rồi lấy val_MAE nhỏ nhất cho từng
    giá trị, kết quả bị nhiễu: giá trị nào xuất hiện ở nhiều cấu hình hơn sẽ
    có cơ hội trúng val_MAE thấp hơn, không phải vì nó tốt hơn.

    Vì vậy chỉ so sánh TRONG nhóm có mọi tham số khác giống hệt nhau.
    """
    khac = [c for c in thamso if c != ts]
    if not khac:
        return t
    nhom = [g for _, g in t.groupby(khac) if g[ts].nunique() > 1]
    if not nhom:
        return None
    return max(nhom, key=lambda g: (g[ts].nunique(), -g["val_MAE"].min()))


def hinh_03(model, cfg):
    """Từ tuning_log.csv: mỗi tham số một nhóm cột, so sánh có kiểm soát."""
    p = ROOT_DIR / "results" / "dl" / model.lower() / "tables" / "tuning_log.csv"
    if not p.exists():
        print("  (bỏ qua hình 03: chưa có tuning_log.csv)")
        return None
    t = pd.read_csv(p, encoding="utf-8-sig")
    thamso = [c for c in ["lookback", "hidden", "lr", "layers", "dropout"]
              if c in t.columns and t[c].nunique() > 1]
    if not thamso:
        return None

    nhom = {ts: _nhom_so_sanh(t, ts, thamso) for ts in thamso}
    thamso = [ts for ts in thamso if nhom[ts] is not None]

    fig, axes = plt.subplots(1, len(thamso), figsize=(3.4 * len(thamso), 4.8),
                             squeeze=False)
    for ax, ts in zip(axes[0], thamso):
        g = nhom[ts].sort_values(ts)
        tot = g.loc[g.val_MAE.idxmin(), ts]
        mau = ["tab:green" if v == tot else "tab:blue" for v in g[ts]]
        b = ax.bar([str(v) for v in g[ts]], g.val_MAE, color=mau,
                   edgecolor="black", lw=0.8)
        ax.bar_label(b, fmt="%.2f", fontsize=8)
        lo, hi = g.val_MAE.min(), g.val_MAE.max()
        bien = max(hi - lo, 1e-9)
        ax.set_ylim(lo - bien * 0.6, hi + bien * 1.0)
        ax.set(xlabel=ts, ylabel="Validation MAE (kW)" if ts == thamso[0] else "",
               title=f"{ts}\nchênh lệch {100*(hi-lo)/lo:.2f}%")
    fig.suptitle(f"{model} — dò tham số trên chuỗi chính ({len(t)} cấu hình). "
                 f"Mỗi ô chỉ so các cấu hình khác nhau ĐÚNG ở tham số đó; "
                 f"cột xanh lá = được chọn.", y=1.0, fontsize=11)
    fig.savefig(FIG / "03_do_tham_so.png", dpi=DPI)
    plt.close(fig)

    hang = []
    for ts in thamso:
        g = nhom[ts]
        lo, hi = g.val_MAE.min(), g.val_MAE.max()
        hang.append({"tham_so": ts, "so_gia_tri_thu": g[ts].nunique(),
                     "gia_tri_tot_nhat": g.loc[g.val_MAE.idxmin(), ts],
                     "val_MAE_tot_nhat": lo, "val_MAE_te_nhat": hi,
                     "muc_anh_huong_%": 100 * (hi - lo) / lo})
    r = pd.DataFrame(hang).sort_values("muc_anh_huong_%", ascending=False).round(4)
    r.to_csv(TAB / "muc_anh_huong_tham_so.csv", index=False, encoding="utf-8-sig")
    return r


# ======================================================================
# HÌNH 04 - Hội tụ và độ ổn định giữa các seed
# ======================================================================

def hinh_04(model, cac_chuoi):
    p = ROOT_DIR / "results" / "experiments.csv"
    if not p.exists():
        return None
    e = pd.read_csv(p, encoding="utf-8-sig")
    e = e[(e.model == model) & (e.stage == "final")]
    if e.empty:
        return None
    # sổ cái chỉ-thêm có thể chứa nhiều cấu hình của cùng mô hình (bản đặt tay
    # và bản dò ra). Không lọc thì mọi con số dưới đây là trung bình của hai
    # cấu hình khác nhau, và so_seed thành 10 thay vì 5.
    e = chi_giu_cau_hinh_da_do(e)
    # nếu có cả hai cấu hình có/không đặc trưng lịch thì lấy cấu hình mới nhất
    if "calendar" in e.columns and e.calendar.nunique() > 1:
        e = e[e.calendar == e.calendar.iloc[-1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) best_epoch và tổng số epoch đã chạy
    x = np.arange(len(cac_chuoi))
    be = [e[e.customer == c]["best_epoch"] for c in cac_chuoi]
    er = [e[e.customer == c]["epochs_run"] for c in cac_chuoi]
    ax1.bar(x - 0.2, [v.mean() for v in er], 0.4, label="Tổng epoch đã chạy",
            color="lightsteelblue", yerr=[v.std() for v in er], capsize=3)
    b = ax1.bar(x + 0.2, [v.mean() for v in be], 0.4, label="Best epoch",
                color="tab:blue", yerr=[v.std() for v in be], capsize=3)
    ax1.bar_label(b, fmt="%.0f", fontsize=8)
    ax1.set(xticks=x, ylabel="Epoch",
            title="Hội tụ: best epoch so với tổng số epoch đã chạy\n"
                  "(chênh lệch = patience, nghĩa là early stopping đã chạy đủ)")
    ax1.set_xticklabels(cac_chuoi); ax1.legend(fontsize=9)

    # (b) độ ổn định giữa các seed
    pct = []
    for c in cac_chuoi:
        s = e[e.customer == c]["test_MAE"]
        ax2.scatter([c] * len(s), s, s=55, zorder=3, alpha=0.85)
        ax2.scatter([c], [s.mean()], marker="_", s=600, c="crimson", zorder=4)
        pct.append(100 * s.std() / s.mean())
    for i, v in enumerate(pct):
        ax2.annotate(f"std = {v:.1f}%", (i, e[e.customer == cac_chuoi[i]]["test_MAE"].max()),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color="crimson")
    ax2.set(ylabel="Test MAE (kW)", yscale="log",
            title="Độ ổn định giữa các seed\n(mỗi chấm = 1 seed, gạch đỏ = trung bình)")
    ax2.legend(["từng seed", "trung bình"], fontsize=8)

    fig.savefig(FIG / "04_hoi_tu_va_on_dinh.png", dpi=DPI)
    plt.close(fig)

    r = (e.groupby("customer")
           .agg(best_epoch_TB=("best_epoch", "mean"), best_epoch_std=("best_epoch", "std"),
                epochs_run_TB=("epochs_run", "mean"),
                test_MAE_TB=("test_MAE", "mean"), test_MAE_std=("test_MAE", "std"),
                test_MASE_TB=("test_MASE", "mean"), so_seed=("seed", "count"))
           .reindex(cac_chuoi).reset_index().round(4))
    r["std_tren_TB_%"] = (100 * r.test_MAE_std / r.test_MAE_TB).round(2)
    r.to_csv(TAB / "hoi_tu_va_on_dinh.csv", index=False, encoding="utf-8-sig")
    return r


# ======================================================================
# HÌNH 05 - Trước và sau khi dò tham số
# ======================================================================

def hinh_05(model, cac_chuoi):
    """So sánh lần chạy mốc (stage=baseline) với lần chạy chính thức."""
    p = ROOT_DIR / "results" / "experiments.csv"
    if not p.exists():
        return None
    e = pd.read_csv(p, encoding="utf-8-sig")
    moc = e[(e.model == model) & (e.stage == "baseline")]
    cuoi = e[(e.model == model) & (e.stage == "final")]
    if moc.empty or cuoi.empty:
        print("  (bỏ qua hình 05: chưa có cả stage baseline lẫn final)")
        return None
    cuoi = chi_giu_cau_hinh_da_do(cuoi)
    if "calendar" in cuoi.columns and cuoi.calendar.nunique() > 1:
        cuoi = cuoi[cuoi.calendar == cuoi.calendar.iloc[-1]]

    a = [moc[moc.customer == c]["test_MAE"].mean() for c in cac_chuoi]
    b = [cuoi[cuoi.customer == c]["test_MAE"].mean() for c in cac_chuoi]
    sd = [cuoi[cuoi.customer == c]["test_MAE"].std() for c in cac_chuoi]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cac_chuoi))
    ax.bar(x - 0.2, a, 0.4, label="Chạy mốc (cấu hình mặc định, 1 seed)",
           color="lightsteelblue", edgecolor="black", lw=0.8)
    bb = ax.bar(x + 0.2, b, 0.4, label="Sau khi dò tham số (5 seed)",
                color="tab:blue", edgecolor="black", lw=0.8, yerr=sd, capsize=3)
    for i, (u, v) in enumerate(zip(a, b)):
        ax.annotate(f"{100*(v/u-1):+.1f}%", (i + 0.2, v),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=9, weight="bold",
                    color="green" if v < u else "crimson")
    ax.set(xticks=x, ylabel="Test MAE (kW)", yscale="log",
           title=f"{model} — tác dụng của việc dò tham số")
    ax.set_xticklabels(cac_chuoi); ax.legend(fontsize=9)
    fig.savefig(FIG / "05_truoc_sau_do_tham_so.png", dpi=DPI)
    plt.close(fig)

    r = pd.DataFrame({"chuoi": cac_chuoi, "MAE_chay_moc": a,
                      "MAE_sau_do": b, "thay_doi_%": [100*(v/u-1) for u, v in zip(a, b)]})
    r.round(4).to_csv(TAB / "truoc_sau_do_tham_so.csv",
                      index=False, encoding="utf-8-sig")
    return r


# ======================================================================
# CHẠY CHÍNH
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="Biểu đồ phần huấn luyện một mô hình")
    ap.add_argument("--model", default="LSTM", choices=["LSTM", "MLP", "CNNLSTM"])
    ap.add_argument("--smooth", type=int, default=1,
                    help="làm mượt learning curve bằng trung bình trượt N epoch")
    a = ap.parse_args()

    global FIG, TAB
    goc = ROOT_DIR / "results" / "analysis_train" / a.model.lower()
    FIG, TAB = goc / "figures", goc / "tables"
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    cac_chuoi = cfg["customers"]["hourly_file_columns"]

    print("=" * 70)
    print(f"BIỂU ĐỒ PHẦN HUẤN LUYỆN — {a.model}")
    print("=" * 70)
    print(f"Chuỗi  : {cac_chuoi}")
    print(f"Đầu ra : {goc}\n")

    h = doc_history(a.model, cac_chuoi)
    if h is None:
        print(f"!! Không tìm thấy history_*.csv trong "
              f"results/dl/{a.model.lower()}/tables/")
        return
    print(f"[1/5] Learning curve từng chuỗi ({h.seed.nunique()} seed) ...")
    hinh_01(a.model, h, cac_chuoi, a.smooth)

    print("[2/5] Learning curve 4 chuỗi trên một hình ...")
    hinh_02(a.model, h, cac_chuoi, a.smooth)

    print("[3/5] Quá trình dò tham số ...")
    r3 = hinh_03(a.model, cfg)
    if r3 is not None:
        print("\n  Mức ảnh hưởng của từng tham số (giảm dần):")
        for _, r in r3.iterrows():
            print(f"    {r.tham_so:10} tốt nhất = {r.gia_tri_tot_nhat:<8} "
                  f"ảnh hưởng {r['muc_anh_huong_%']:5.2f}%")

    print("\n[4/5] Hội tụ và độ ổn định giữa các seed ...")
    r4 = hinh_04(a.model, cac_chuoi)
    if r4 is not None:
        for _, r in r4.iterrows():
            print(f"    {r.customer:8} best epoch {r.best_epoch_TB:5.1f} "
                  f"/ chạy {r.epochs_run_TB:5.1f} | test MAE {r.test_MAE_TB:8.3f} "
                  f"(std {r['std_tren_TB_%']:.1f}%)")

    print("\n[5/5] Trước và sau khi dò tham số ...")
    r5 = hinh_05(a.model, cac_chuoi)
    if r5 is not None:
        for _, r in r5.iterrows():
            print(f"    {r.chuoi:8} {r.MAE_chay_moc:8.3f} -> {r.MAE_sau_do:8.3f} "
                  f"({r['thay_doi_%']:+.1f}%)")

    print("\n" + "=" * 70)
    print(f"XONG.  Biểu đồ: {FIG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
