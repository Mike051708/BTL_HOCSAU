"""Phân tích lỗi và sinh biểu đồ cho báo cáo BTL Học sâu - Chủ đề 7.

Bám sát các yêu cầu của đề bài:

    "So sánh với naive baseline."                -> Hình 01, Bảng bang_tong_hop
    "Trực quan hóa dự báo và giá trị thật."      -> Hình 08
    "Phân tích thời điểm mô hình sai nhiều
     (đỉnh tải, cuối tuần...)."                  -> Hình 03, 04, 06, 07
    "MAE, RMSE; có thể bổ sung MAPE khi phù hợp" -> mọi bảng (kèm sMAPE và MASE)

CHẠY:
    python src/error_analysis.py                 # tất cả các chuỗi
    python src/error_analysis.py --customer MT_162
    python src/error_analysis.py --seed 42       # chỉ dùng 1 seed thay vì gộp 5

ĐẦU VÀO (đã có sẵn sau khi chạy src/train_dl.py):
    results/dl/<model>/predictions/test_<chuoi>_s<seed>.csv
    results/<chuoi>/baseline_predictions_test_<chuoi>.csv
    results/experiments.csv
    data/processed/train.csv                     # để tính d_train cho MASE

ĐẦU RA:
    results/analysis/figures/*.png
    results/analysis/tables/*.csv

Không cần GPU. Chạy khoảng 1-2 phút.
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
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import ROOT_DIR, load_config

OUT     = ROOT_DIR / "results" / "analysis"
FIG     = OUT / "figures"
TAB     = OUT / "tables"
DPI     = 130

# Nhãn tiếng Việt cho bốn baseline của src/baseline.py
TEN_BASELINE = {
    "y_last_value": "Last Value",
    "y_ma3":        "Moving Avg 3h",
    "y_snaive24":   "Seasonal Naive 24h",
    "y_snaive168":  "Seasonal Naive 168h",
}
THU = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"]

plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "font.size": 10,
})


# ======================================================================
# CHỈ SỐ
# ======================================================================

def mae(a, p):  return float(np.mean(np.abs(a - p)))
def rmse(a, p): return float(np.sqrt(np.mean((a - p) ** 2)))


def smape(a, p):
    """sMAPE (%): đối xứng, không nổ vô cực khi y_true gần 0."""
    d = (np.abs(a) + np.abs(p)) / 2.0
    m = d > 1e-9
    return float(np.mean(np.abs(a[m] - p[m]) / d[m]) * 100)


def mape_loc(a, p, nguong):
    """MAPE (%) nhưng BỎ QUA các điểm y_true < nguong.

    Đề bài ghi 'có thể bổ sung MAPE khi phù hợp'. MAPE nổ vô cực khi y_true
    gần 0, mà dữ liệu điện có những giờ tải rất thấp (MT_162 có 4 giờ bằng 0).
    Vì vậy phải nêu rõ ngưỡng lọc và tỷ lệ điểm bị loại; sMAPE được báo cáo
    song song vì không cần ngưỡng.
    """
    m = np.abs(a) >= nguong
    if m.sum() == 0:
        return np.nan, 0.0
    val = float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100)
    return val, float(100 * (1 - m.mean()))


def d_train_cua(customer, cfg):
    """Mẫu số MASE: sai phân bậc 1 trung bình trên tập TRAIN."""
    tr = pd.read_csv(ROOT_DIR / "data" / "processed" / "train.csv",
                     encoding="utf-8-sig")
    return float(np.mean(np.abs(np.diff(tr[customer].to_numpy(dtype=float)))))


# ======================================================================
# ĐỌC DỮ LIỆU
# ======================================================================

def doc_du_bao_dl(customer, seed=None):
    """Đọc dự báo test của mọi mô hình học sâu -> {ten_mo_hinh: DataFrame}.

    Gộp mọi seed (hoặc chỉ một seed nếu truyền --seed). Gộp nhiều seed cho
    ước lượng sai số theo nhóm ổn định hơn, vì mỗi nhóm có 5 lần quan sát
    độc lập thay vì 1.
    """
    out = {}
    goc = ROOT_DIR / "results" / "dl"
    if not goc.exists():
        return out
    for thu_muc in sorted(goc.iterdir()):
        pat = f"test_{customer}_s{seed}.csv" if seed else f"test_{customer}_s*.csv"
        tep = sorted((thu_muc / "predictions").glob(pat))
        if not tep:
            continue
        df = pd.concat([pd.read_csv(t, encoding="utf-8-sig") for t in tep],
                       ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        out[df["model"].iloc[0]] = df
    return out


def doc_du_bao_baseline(customer):
    """Đọc dự báo test của 4 baseline -> DataFrame dạng dài."""
    p = ROOT_DIR / "results" / customer / f"baseline_predictions_test_{customer}.csv"
    if not p.exists():
        return None
    w = pd.read_csv(p, encoding="utf-8-sig")
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    dai = []
    for cot, ten in TEN_BASELINE.items():
        if cot not in w.columns:
            continue
        dai.append(pd.DataFrame({
            "timestamp": w["timestamp"], "customer": customer, "model": ten,
            "seed": -1, "y_true": w["y_true"], "y_pred": w[cot],
            "error": w["y_true"] - w[cot],
        }))
    return pd.concat(dai, ignore_index=True) if dai else None


def gop_tat_ca(customer, seed=None):
    """Ghép baseline + các mô hình học sâu thành một bảng dài duy nhất."""
    phan = []
    b = doc_du_bao_baseline(customer)
    if b is not None:
        phan.append(b)
    for _, df in doc_du_bao_dl(customer, seed).items():
        phan.append(df[["timestamp", "customer", "model", "seed",
                        "y_true", "y_pred", "error"]])
    if not phan:
        return None
    d = pd.concat(phan, ignore_index=True)
    d["abs_error"] = d["error"].abs()
    d["gio"] = d["timestamp"].dt.hour
    d["thu"] = d["timestamp"].dt.dayofweek
    return d


def danh_dau_ngay_le(d):
    """Thêm cột is_holiday theo lịch nghỉ Bồ Đào Nha (nguồn dữ liệu là BĐN)."""
    try:
        import holidays
        nam = range(int(d["timestamp"].dt.year.min()),
                    int(d["timestamp"].dt.year.max()) + 1)
        pt = holidays.PT(years=nam)
        d["is_holiday"] = [ts.date() in pt for ts in d["timestamp"]]
        return d, True
    except ImportError:
        print("  [Cảnh báo] Chưa cài `holidays` -> bỏ qua phần ngày lễ.")
        print("             Cài bằng: pip install holidays")
        d["is_holiday"] = False
        return d, False


# ======================================================================
# BẢNG TỔNG HỢP  (đề bài: "So sánh với naive baseline", "MAE, RMSE, MAPE")
# ======================================================================

def bang_tong_hop(cac_chuoi, cfg, seed=None):
    hang = []
    for c in cac_chuoi:
        d = gop_tat_ca(c, seed)
        if d is None:
            continue
        dtr = d_train_cua(c, cfg)
        nguong = 0.05 * d["y_true"].mean()      # 5% mức tải trung bình
        for ten, g in d.groupby("model"):
            # Với mô hình nhiều seed: tính chỉ số cho TỪNG seed rồi lấy TB ± std
            ds = []
            for _, gs in g.groupby("seed"):
                a, p = gs["y_true"].to_numpy(), gs["y_pred"].to_numpy()
                mp, bo = mape_loc(a, p, nguong)
                ds.append({"MAE": mae(a, p), "RMSE": rmse(a, p),
                           "MASE": mae(a, p) / dtr, "sMAPE_%": smape(a, p),
                           "MAPE_%": mp, "ty_le_bo_%": bo})
            t = pd.DataFrame(ds)
            hang.append({
                "chuoi": c, "mo_hinh": ten, "so_seed": len(t),
                "MAE": t.MAE.mean(), "MAE_std": t.MAE.std(ddof=0),
                "RMSE": t.RMSE.mean(), "RMSE_std": t.RMSE.std(ddof=0),
                "MASE": t.MASE.mean(), "MASE_std": t.MASE.std(ddof=0),
                "sMAPE_%": t["sMAPE_%"].mean(),
                "MAPE_%": t["MAPE_%"].mean(),
                "MAPE_bo_qua_%": t["ty_le_bo_%"].mean(),
                "N_moi_seed": len(g) // max(len(t), 1),
            })
    r = pd.DataFrame(hang).round(4)
    r.to_csv(TAB / "bang_tong_hop.csv", index=False, encoding="utf-8-sig")
    return r


# ======================================================================
# HÌNH 01 - So sánh mô hình  (đề bài: "So sánh với naive baseline")
# ======================================================================

def hinh_01(tong_hop, cac_chuoi):
    thu_tu = list(TEN_BASELINE.values()) + ["MLP", "LSTM", "CNNLSTM"]
    mo_hinh = [m for m in thu_tu if m in tong_hop.mo_hinh.unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(cac_chuoi))
    w = 0.8 / len(mo_hinh)

    for i, m in enumerate(mo_hinh):
        sub = tong_hop[tong_hop.mo_hinh == m].set_index("chuoi").reindex(cac_chuoi)
        la_dl = m in ("MLP", "LSTM", "CNNLSTM")
        ax1.bar(x + i * w - 0.4 + w / 2, sub["MASE"], w, label=m,
                yerr=sub["MASE_std"], capsize=2,
                edgecolor="black" if la_dl else "none",
                linewidth=1.1 if la_dl else 0)

    ax1.axhline(1.0, ls="--", c="crimson", lw=1.2)
    ax1.text(len(cac_chuoi) - 0.45, 1.02, "MASE = 1: ngang Naive",
             color="crimson", fontsize=8, ha="right")
    ax1.set(xticks=x, ylabel="Test MASE (thấp hơn = tốt hơn)",
            title="So sánh mô hình theo MASE\n(chuẩn hoá, so được giữa các chuỗi)")
    ax1.set_xticklabels(cac_chuoi)
    ax1.legend(fontsize=8, ncol=2)

    # % cải thiện của mô hình học sâu so với baseline TỐT NHẤT từng chuỗi
    dl = [m for m in mo_hinh if m in ("MLP", "LSTM", "CNNLSTM")]
    for i, m in enumerate(dl):
        cai_thien = []
        for c in cac_chuoi:
            s = tong_hop[tong_hop.chuoi == c]
            tot_nhat = s[s.mo_hinh.isin(TEN_BASELINE.values())]["MAE"].min()
            cua_m = s[s.mo_hinh == m]["MAE"]
            cai_thien.append(100 * (1 - cua_m.iloc[0] / tot_nhat) if len(cua_m) else 0)
        b = ax2.bar(x + i * (0.8 / len(dl)) - 0.4 + 0.4 / len(dl),
                    cai_thien, 0.8 / len(dl), label=m, edgecolor="black", lw=1.1)
        ax2.bar_label(b, fmt="%.1f%%", fontsize=8)

    ax2.axhline(0, c="black", lw=0.8)
    ax2.set(xticks=x, ylabel="% giảm MAE so với baseline tốt nhất",
            title="Mức cải thiện của mô hình học sâu\nso với baseline tốt nhất từng chuỗi")
    ax2.set_xticklabels(cac_chuoi)
    ax2.legend(fontsize=9)

    fig.savefig(FIG / "01_so_sanh_mo_hinh.png", dpi=DPI)
    plt.close(fig)


# ======================================================================
# HÌNH 02 - Ablation đặc trưng lịch
# ======================================================================

def hinh_02(cac_chuoi):
    p = ROOT_DIR / "results" / "experiments.csv"
    if not p.exists():
        return None
    e = pd.read_csv(p, encoding="utf-8-sig")
    e = e[e.stage == "final"]
    if e.empty:
        return None
    e["cau_hinh"] = e.model + np.where(e.calendar == 1, " + lịch", " (không lịch)")

    g = (e.groupby(["customer", "cau_hinh"])
           .agg(MAE=("test_MAE", "mean"), MAE_std=("test_MAE", "std"),
                MASE=("test_MASE", "mean"), n=("seed", "count"))
           .reset_index())
    g.round(4).to_csv(TAB / "ablation_dac_trung_lich.csv",
                      index=False, encoding="utf-8-sig")

    cau_hinh = sorted(g.cau_hinh.unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(cac_chuoi)); w = 0.8 / len(cau_hinh)

    for i, ch in enumerate(cau_hinh):
        s = g[g.cau_hinh == ch].set_index("customer").reindex(cac_chuoi)
        ax1.bar(x + i * w - 0.4 + w / 2, s["MASE"], w, label=ch,
                yerr=s["MAE_std"] / s["MAE"] * s["MASE"], capsize=2)
    ax1.set(xticks=x, ylabel="Test MASE",
            title="Ma trận 2×2: mô hình × đặc trưng lịch")
    ax1.set_xticklabels(cac_chuoi); ax1.legend(fontsize=8)

    # Đặc trưng lịch làm MAE thay đổi bao nhiêu %? (âm = tốt lên)
    for i, mh in enumerate(sorted(e.model.unique())):
        pct = []
        for c in cac_chuoi:
            s = g[g.customer == c]
            co = s[s.cau_hinh == f"{mh} + lịch"]["MAE"]
            kh = s[s.cau_hinh == f"{mh} (không lịch)"]["MAE"]
            pct.append(100 * (co.iloc[0] / kh.iloc[0] - 1)
                       if len(co) and len(kh) else np.nan)
        b = ax2.bar(x + i * 0.4 - 0.2, pct, 0.4, label=mh,
                    edgecolor="black", lw=1.0)
        ax2.bar_label(b, fmt="%+.1f%%", fontsize=8)

    ax2.axhline(0, c="black", lw=1)
    ax2.set(xticks=x, ylabel="% thay đổi MAE khi THÊM đặc trưng lịch",
            title="Đóng góp của đặc trưng lịch\n(âm = tốt lên)")
    ax2.set_xticklabels(cac_chuoi); ax2.legend(fontsize=9)

    fig.savefig(FIG / "02_ablation_dac_trung_lich.png", dpi=DPI)
    plt.close(fig)
    return g


# ======================================================================
# HÌNH 03, 04 - Sai số theo giờ và theo thứ
#   (đề bài: "thời điểm mô hình sai nhiều - đỉnh tải, cuối tuần")
# ======================================================================

def hinh_03_04(customer, d, mo_hinh_ve):
    ket = {}
    for nhom, ten_cot, nhan, so_hinh, tieu_de in [
        ("gio", "gio", [str(h) for h in range(24)], "03",
         "Sai số theo giờ trong ngày"),
        ("thu", "thu", THU, "04",
         "Sai số theo thứ trong tuần"),
    ]:
        bang = (d[d.model.isin(mo_hinh_ve)]
                .groupby(["model", ten_cot])["abs_error"]
                .agg(MAE="mean", N="size").reset_index())
        ket[nhom] = bang

        fig, ax = plt.subplots(figsize=(13 if nhom == "gio" else 9, 4.8))
        x = np.arange(len(nhan)); w = 0.8 / len(mo_hinh_ve)
        for i, m in enumerate(mo_hinh_ve):
            s = bang[bang.model == m].set_index(ten_cot).reindex(range(len(nhan)))
            ax.bar(x + i * w - 0.4 + w / 2, s["MAE"], w, label=m)

        # Đánh dấu nhóm có sai số lớn nhất của mô hình đầu tiên
        s0 = bang[bang.model == mo_hinh_ve[0]].set_index(ten_cot)
        if len(s0):
            cao = bang["MAE"].max()
            ax.set_ylim(0, cao * 1.28)          # chừa chỗ cho chú thích
            xau = int(s0["MAE"].idxmax())
            ax.annotate(f"{mo_hinh_ve[0]} sai nhiều nhất: {nhan[xau]}"
                        f"  (MAE={s0['MAE'].max():.1f})",
                        xy=(xau, s0["MAE"].max()),
                        xytext=(xau, cao * 1.14),
                        ha="center", fontsize=9, color="crimson", weight="bold",
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))

        ax.set(xticks=x, ylabel="MAE (kW)",
               xlabel="Giờ trong ngày" if nhom == "gio" else "Thứ trong tuần",
               title=f"{tieu_de} — {customer}")
        ax.set_xticklabels(nhan)
        ax.legend(fontsize=9)
        fig.savefig(FIG / f"{so_hinh}_sai_so_theo_{nhom}_{customer}.png", dpi=DPI)
        plt.close(fig)

        bang.round(4).to_csv(TAB / f"sai_so_theo_{nhom}_{customer}.csv",
                             index=False, encoding="utf-8-sig")
    return ket


# ======================================================================
# HÌNH 05 - Phân phối sai số (kiểm tra thiên lệch hệ thống)
# ======================================================================

def hinh_05(customer, d, mo_hinh_ve):
    fig, axes = plt.subplots(1, len(mo_hinh_ve),
                             figsize=(5.2 * len(mo_hinh_ve), 4.3), squeeze=False)
    hang = []
    for ax, m in zip(axes[0], mo_hinh_ve):
        e = d[d.model == m]["error"].to_numpy()
        ax.hist(e, bins=70, edgecolor="none")
        ax.axvline(0, c="black", lw=1.2)
        ax.axvline(e.mean(), c="crimson", ls="--", lw=1.4,
                   label=f"trung bình = {e.mean():+.2f}")
        ax.set(xlabel="Sai số = thực tế − dự báo (kW)", ylabel="Số điểm",
               title=f"{m} — {customer}")
        ax.legend(fontsize=8)
        hang.append({
            "chuoi": customer, "mo_hinh": m,
            "trung_binh": e.mean(), "do_lech_chuan": e.std(),
            "trung_vi": np.median(e),
            "ty_le_du_bao_THAP_hon_%": 100 * (e > 0).mean(),
            "do_lech": float(pd.Series(e).skew()),
        })
    fig.savefig(FIG / f"05_phan_phoi_sai_so_{customer}.png", dpi=DPI)
    plt.close(fig)
    r = pd.DataFrame(hang).round(4)
    r.to_csv(TAB / f"thong_ke_sai_so_{customer}.csv",
             index=False, encoding="utf-8-sig")
    return r


# ======================================================================
# HÌNH 06 - Top-10 thời điểm sai nhiều nhất
# ======================================================================

def hinh_06(customer, d, mo_hinh_chinh):
    g = d[d.model == mo_hinh_chinh]
    # Trung bình |sai số| qua các seed cho từng mốc thời gian -> ổn định hơn 1 seed
    tb = (g.groupby("timestamp")
            .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"),
                 abs_error=("abs_error", "mean")).reset_index())
    top = tb.nlargest(10, "abs_error").copy()
    top["thu"] = top["timestamp"].dt.dayofweek.map(lambda i: THU[i])
    so = top.select_dtypes("number").columns          # chỉ làm tròn cột số
    top[so] = top[so].round(3)
    top.to_csv(TAB / f"top10_loi_lon_nhat_{customer}.csv",
               index=False, encoding="utf-8-sig")

    # Vẽ 4 cửa sổ ±36 giờ quanh 4 thời điểm sai nhất, không trùng ngày
    moc, da_co = [], set()
    for _, r in top.iterrows():
        ng = r["timestamp"].date()
        if ng not in da_co:
            moc.append(r["timestamp"]); da_co.add(ng)
        if len(moc) == 4:
            break

    fig, axes = plt.subplots(len(moc), 1, figsize=(13, 3.1 * len(moc)),
                             squeeze=False)
    for ax, m in zip(axes[:, 0], moc):
        w = tb[(tb.timestamp >= m - pd.Timedelta(hours=36)) &
               (tb.timestamp <= m + pd.Timedelta(hours=36))]
        ax.plot(w.timestamp, w.y_true, lw=1.6, label="Thực tế")
        ax.plot(w.timestamp, w.y_pred, lw=1.6, ls="--",
                label=f"Dự báo {mo_hinh_chinh}")
        ax.axvline(m, c="crimson", lw=1.2, alpha=0.8)
        sai = float(tb.loc[tb.timestamp == m, "abs_error"].iloc[0])
        ax.set(ylabel="kW",
               title=f"{m:%d/%m/%Y %H:%M} ({THU[m.dayofweek]}) — sai {sai:.1f} kW")
        ax.legend(fontsize=8)
    fig.savefig(FIG / f"06_top10_loi_lon_nhat_{customer}.png", dpi=DPI)
    plt.close(fig)
    return top


# ======================================================================
# HÌNH 07 - Ngày lễ so với ngày thường
# ======================================================================

def hinh_07(cac_chuoi, mo_hinh_ve, seed=None):
    hang = []
    for c in cac_chuoi:
        d = gop_tat_ca(c, seed)
        if d is None:
            continue
        d, co_le = danh_dau_ngay_le(d)
        if not co_le or not d["is_holiday"].any():
            continue
        for m in mo_hinh_ve:
            s = d[d.model == m]
            if s.empty:
                continue
            le = s[s.is_holiday]["abs_error"]
            thuong = s[~s.is_holiday]["abs_error"]
            if len(le) == 0:
                continue
            hang.append({
                "chuoi": c, "mo_hinh": m,
                "MAE_ngay_thuong": thuong.mean(), "N_thuong": len(thuong),
                "MAE_ngay_le": le.mean(), "N_le": len(le),
                "ty_le_tang_%": 100 * (le.mean() / thuong.mean() - 1),
            })
    if not hang:
        return None

    r = pd.DataFrame(hang).round(4)
    r.to_csv(TAB / "ngay_le_vs_ngay_thuong.csv", index=False, encoding="utf-8-sig")

    mh = sorted(r.mo_hinh.unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cac_chuoi)); w = 0.8 / (len(mh) * 2)
    for i, m in enumerate(mh):
        s = r[r.mo_hinh == m].set_index("chuoi").reindex(cac_chuoi)
        ax.bar(x + (2 * i) * w - 0.4 + w / 2, s["MAE_ngay_thuong"], w,
               label=f"{m} — ngày thường")
        b = ax.bar(x + (2 * i + 1) * w - 0.4 + w / 2, s["MAE_ngay_le"], w,
                   label=f"{m} — ngày lễ", hatch="//", edgecolor="black", lw=0.8)
        ax.bar_label(b, labels=[f"{v:+.0f}%" for v in s["ty_le_tang_%"]],
                     fontsize=7)
    ax.set(xticks=x, ylabel="MAE (kW)",
           title="Sai số ngày lễ so với ngày thường (lịch nghỉ Bồ Đào Nha)")
    ax.set_xticklabels(cac_chuoi); ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIG / "07_ngay_le_vs_ngay_thuong.png", dpi=DPI)
    plt.close(fig)
    return r


# ======================================================================
# HÌNH 08 - Dự báo so với giá trị thật
#   (đề bài: "Trực quan hóa dự báo và giá trị thật")
# ======================================================================

def hinh_08(customer, d, mo_hinh_ve, so_ngay=14):
    tb = (d.groupby(["model", "timestamp"])
            .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
            .reset_index())
    t0 = tb.timestamp.min()
    w = tb[tb.timestamp < t0 + pd.Timedelta(days=so_ngay)]

    fig, ax = plt.subplots(figsize=(14, 5))
    thuc = w[w.model == mo_hinh_ve[0]]
    ax.plot(thuc.timestamp, thuc.y_true, lw=2.0, c="black", label="Thực tế")
    for m in mo_hinh_ve:
        s = w[w.model == m]
        ax.plot(s.timestamp, s.y_pred, lw=1.3, alpha=0.9, label=f"Dự báo {m}")
    # tô nền cuối tuần
    for ngay in pd.date_range(w.timestamp.min().normalize(),
                              w.timestamp.max().normalize(), freq="D"):
        if ngay.dayofweek >= 5:
            ax.axvspan(ngay, ngay + pd.Timedelta(days=1), color="grey", alpha=0.10)
    ax.set(xlabel="Thời gian", ylabel="kW",
           title=f"Dự báo so với thực tế — {customer}, {so_ngay} ngày đầu tập test"
                 f"\n(vùng xám = cuối tuần)")
    ax.legend(fontsize=9, ncol=3)
    fig.savefig(FIG / f"08_du_bao_vs_thuc_te_{customer}.png", dpi=DPI)
    plt.close(fig)


# ======================================================================
# CHẠY CHÍNH
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="Phân tích lỗi và sinh biểu đồ")
    ap.add_argument("--customer", default=None, help="chỉ phân tích một chuỗi")
    ap.add_argument("--seed", type=int, default=None,
                    help="chỉ dùng một seed (mặc định: gộp mọi seed)")
    ap.add_argument("--days", type=int, default=14, help="số ngày vẽ ở hình 08")
    a = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    cac_chuoi = ([a.customer] if a.customer
                 else cfg["customers"]["hourly_file_columns"])
    chinh = cfg["customers"]["main"]

    print("=" * 70)
    print("PHÂN TÍCH LỖI — BTL Học sâu, Chủ đề 7")
    print("=" * 70)
    print(f"Chuỗi      : {cac_chuoi}  (chuỗi chính: {chinh})")
    print(f"Seed       : {'tất cả' if a.seed is None else a.seed}")
    print(f"Đầu ra     : {OUT}\n")

    print("[1/4] Bảng tổng hợp MAE / RMSE / MASE / sMAPE / MAPE ...")
    th = bang_tong_hop(cac_chuoi, cfg, a.seed)
    if th.empty:
        print("  !! Không đọc được dự báo nào. Kiểm tra results/dl/*/predictions/")
        return
    for c in cac_chuoi:
        s = th[th.chuoi == c].sort_values("MAE")
        print(f"\n  {c}:")
        for _, r in s.iterrows():
            print(f"    {r.mo_hinh:22} MAE {r.MAE:9.3f} ± {r.MAE_std:6.3f}"
                  f" | MASE {r.MASE:.4f} | sMAPE {r['sMAPE_%']:5.2f}%")

    print("\n[2/4] Hình 01 (so sánh mô hình) và 02 (ablation đặc trưng lịch) ...")
    hinh_01(th, cac_chuoi)
    hinh_02(cac_chuoi)

    # Ưu tiên vẽ: 2 mô hình học sâu + baseline tốt nhất
    co = list(th.mo_hinh.unique())
    dl = [m for m in ("MLP", "LSTM", "CNNLSTM") if m in co]
    bl_tot = (th[th.mo_hinh.isin(TEN_BASELINE.values())]
              .groupby("mo_hinh")["MASE"].mean().idxmin())
    ve = dl + [bl_tot]
    mo_hinh_chinh = dl[0] if dl else bl_tot
    print(f"  Mô hình đưa vào biểu đồ: {ve}")

    print("\n[3/4] Hình 03-06, 08 cho từng chuỗi ...")
    for c in cac_chuoi:
        d = gop_tat_ca(c, a.seed)
        if d is None:
            continue
        hinh_03_04(c, d, ve)
        hinh_05(c, d, dl or [bl_tot])
        top = hinh_06(c, d, mo_hinh_chinh)
        hinh_08(c, d, ve, a.days)
        t = top.iloc[0]
        print(f"  {c}: lỗi lớn nhất của {mo_hinh_chinh} tại "
              f"{t['timestamp']:%d/%m/%Y %H:%M} — sai {t['abs_error']:.1f} kW "
              f"(thực tế {t['y_true']:.1f}, dự báo {t['y_pred']:.1f})")

    print("\n[4/4] Hình 07 (ngày lễ so với ngày thường) ...")
    le = hinh_07(cac_chuoi, ve, a.seed)
    if le is not None:
        for _, r in le[le.mo_hinh == mo_hinh_chinh].iterrows():
            print(f"  {r['chuoi']}: ngày thường {r['MAE_ngay_thuong']:.2f} -> "
                  f"ngày lễ {r['MAE_ngay_le']:.2f} kW "
                  f"({r['ty_le_tang_%']:+.1f}%, {int(r['N_le'])} giờ lễ)")

    print("\n" + "=" * 70)
    print("XONG.")
    print(f"  Biểu đồ : {FIG}")
    print(f"  Bảng    : {TAB}")
    print("=" * 70)


if __name__ == "__main__":
    main()
