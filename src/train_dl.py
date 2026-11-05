"""Huấn luyện mô hình học sâu (LSTM / MLP / CNN-LSTM) cho dự báo điện năng t+1.

CHẠY TRÊN VSCODE (có GPU):

    # 0. Kiem tra moi truong truoc khi lam gi khac
    python src/train_dl.py --stage check

    # 1. Chay moc: 4 chuoi, 1 seed, cau hinh mac dinh
    python src/train_dl.py --stage baseline

    # 2. Do tham so - CHI tren chuoi chinh MT_162 (~12 lan chay)
    python src/train_dl.py --stage tune

    # 3. Chay chinh thuc: 4 chuoi x 5 seed voi cau hinh tot nhat
    python src/train_dl.py --stage final --lookback 48 --hidden 64 --layers 2

    # Chay le mot cau hinh bat ky
    python src/train_dl.py --stage single --customer MT_162 --lookback 24 --calendar

NGUYEN TAC BAT BUOC:
    - Moi quyet dinh chon tham so chi duoc nhin VALIDATION.
    - TEST chi duoc dung de bao cao sau khi da chot cau hinh.
      Script van tinh metric test moi lan chay de tien luu ho so, nhung
      KHONG duoc dung so test de chon cau hinh. Day la ro ri du lieu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # khong can man hinh, chi xuat file PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

try:
    from src.config import ROOT_DIR, load_config
    from src.data import load_hourly_data, split_by_time
    from src.metrics import evaluate_forecast, calculate_d_train
    from src.dl_utils import (set_seed_all, get_device, make_windows, make_eval_windows,
                              calendar_feats, fit_model, predict_scaled, log_experiment)
    from src.models_dl import build_model, count_params
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import ROOT_DIR, load_config
    from src.data import load_hourly_data, split_by_time
    from src.metrics import evaluate_forecast, calculate_d_train
    from src.dl_utils import (set_seed_all, get_device, make_windows, make_eval_windows,
                              calendar_feats, fit_model, predict_scaled, log_experiment)
    from src.models_dl import build_model, count_params


# ======================================================================
# MOT LAN CHAY = MOT CHUOI + MOT CAU HINH + MOT SEED
# ======================================================================

def run_once(customer, splits, cfg, args, seed, *, save_artifacts=True, tag=""):
    """Huấn luyện và đánh giá một lần. Trả về dict metric."""
    train_df, val_df, test_df = splits
    device = args._device
    n_cal = 6 if args.calendar else 0

    set_seed_all(seed)

    # --- Chuan hoa: fit CHI tren train, rieng tung chuoi ---
    tr_raw = train_df[customer].to_numpy(dtype=np.float64)
    va_raw = val_df[customer].to_numpy(dtype=np.float64)
    te_raw = test_df[customer].to_numpy(dtype=np.float64)

    scaler = StandardScaler().fit(tr_raw.reshape(-1, 1))
    tr_s = scaler.transform(tr_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)
    va_s = scaler.transform(va_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)
    te_s = scaler.transform(te_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

    # --- Cua so truot ---
    L = args.lookback
    Xtr, ytr = make_windows(tr_s, L)
    Xva, yva = make_eval_windows(tr_s, va_s, L)
    Xte, yte = make_eval_windows(va_s, te_s, L)

    # nhan cua train bat dau tu gio thu L cua tap train
    cal_tr = (calendar_feats(train_df.index[L:], args.holiday) if n_cal
              else np.zeros((len(Xtr), 0), dtype=np.float32))
    cal_va = (calendar_feats(val_df.index, args.holiday) if n_cal
              else np.zeros((len(Xva), 0), dtype=np.float32))
    cal_te = (calendar_feats(test_df.index, args.holiday) if n_cal
              else np.zeros((len(Xte), 0), dtype=np.float32))

    assert len(Xva) == len(val_df) == 3672, f"Validation co {len(Xva)} nhan, can 3672"
    assert len(Xte) == len(test_df) == 3672, f"Test co {len(Xte)} nhan, can 3672"

    # --- Mo hinh ---
    model = build_model(args.model, lookback=L, n_cal=n_cal, hidden=args.hidden,
                        layers=args.layers, dropout=args.dropout).to(device)
    n_par = count_params(model)
    print(f"  [{customer}] {args.model} L={L} hidden={args.hidden} layers={args.layers} "
          f"drop={args.dropout} lr={args.lr} cal={bool(n_cal)} seed={seed} "
          f"| {n_par:,} tham so | {len(Xtr):,} mau train")

    model, hist, best_epoch = fit_model(
        model, Xtr, ytr, cal_tr, Xva, yva, cal_va,
        device=device, max_epochs=args.max_epochs, patience=args.patience,
        min_delta=args.min_delta, lr=args.lr, batch_size=args.batch_size,
        verbose_every=args.verbose_every,
    )

    # --- Dua ve thang goc TRUOC khi tinh metric ---
    inv = lambda a: scaler.inverse_transform(a.reshape(-1, 1)).reshape(-1)
    va_pred = inv(predict_scaled(model, Xva, cal_va, device))
    te_pred = inv(predict_scaled(model, Xte, cal_te, device))

    d_train = calculate_d_train(tr_raw)
    m_va = evaluate_forecast(va_raw, va_pred, d_train)
    m_te = evaluate_forecast(te_raw, te_pred, d_train)

    print(f"    VAL  MAE {m_va['MAE']:8.3f} | RMSE {m_va['RMSE']:8.3f} | MASE {m_va['MASE']:.4f}")
    print(f"    TEST MAE {m_te['MAE']:8.3f} | RMSE {m_te['RMSE']:8.3f} | MASE {m_te['MASE']:.4f}")

    row = {
        "exp_id": f"{args.model}_{customer}_L{L}_h{args.hidden}_l{args.layers}"
                  f"_d{args.dropout}_lr{args.lr}_cal{int(bool(n_cal))}_s{seed}{tag}",
        "stage": args.stage, "model": args.model, "customer": customer,
        "lookback": L, "hidden": args.hidden, "layers": args.layers,
        "dropout": args.dropout, "lr": args.lr, "batch_size": args.batch_size,
        "calendar": int(bool(n_cal)), "holiday": int(args.holiday and bool(n_cal)),
        "seed": seed, "n_params": n_par, "best_epoch": best_epoch,
        "epochs_run": len(hist),
        "val_MAE": m_va["MAE"], "val_RMSE": m_va["RMSE"], "val_MASE": m_va["MASE"],
        "test_MAE": m_te["MAE"], "test_RMSE": m_te["RMSE"], "test_MASE": m_te["MASE"],
    }
    log_experiment(row)

    if save_artifacts:
        _save_artifacts(args, customer, seed, model, scaler, hist, best_epoch, n_par,
                        val_df.index, va_raw, va_pred, test_df.index, te_raw, te_pred)
    return row


def _save_artifacts(args, customer, seed, model, scaler, hist, best_epoch, n_par,
                    va_idx, va_true, va_pred, te_idx, te_true, te_pred):
    """Lưu checkpoint, history, dự báo, learning curve và biểu đồ test."""
    base = ROOT_DIR / "results" / "dl" / args.model.lower()
    for sub in ("tables", "predictions", "figures"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    mdir = ROOT_DIR / "models" / "dl"
    mdir.mkdir(parents=True, exist_ok=True)
    sfx = f"{customer}_s{seed}"

    torch.save({
        "model_name": args.model, "customer": customer, "seed": seed,
        "lookback": args.lookback, "hidden": args.hidden, "layers": args.layers,
        "dropout": args.dropout, "lr": args.lr, "calendar": bool(args.calendar),
        "holiday": bool(args.holiday), "best_epoch": best_epoch, "n_params": n_par,
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
    }, mdir / f"{args.model}_{sfx}.pt")

    hist.to_csv(base / "tables" / f"history_{sfx}.csv", index=False, encoding="utf-8-sig")

    for split, idx, yt, yp in (("validation", va_idx, va_true, va_pred),
                               ("test", te_idx, te_true, te_pred)):
        pd.DataFrame({"timestamp": idx, "customer": customer, "model": args.model,
                      "seed": seed, "y_true": yt, "y_pred": yp,
                      "error": yt - yp}).to_csv(
            base / "predictions" / f"{split}_{sfx}.csv", index=False, encoding="utf-8-sig")

    # Learning curve: them truc phu cho learning rate -> nhin ra cho LR giam
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist.epoch, hist.train_mse_scaled, label="Train MSE", lw=1.4)
    ax.plot(hist.epoch, hist.validation_mse_scaled, label="Validation MSE", lw=1.4)
    ax.axvline(best_epoch, ls="--", c="gray", lw=1, label=f"best epoch = {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="MSE (thang chuan hoa)",
           title=f"{args.model} - {customer} (seed {seed}, lookback {args.lookback})")
    ax2 = ax.twinx()
    ax2.plot(hist.epoch, hist.lr, c="tab:green", alpha=0.45, lw=1, label="learning rate")
    ax2.set_yscale("log"); ax2.set_ylabel("learning rate")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(base / "figures" / f"learning_curve_{sfx}.png", dpi=130)
    plt.close(fig)

    # 2 tuan dau tap test: du bao vs thuc te
    n = 24 * 14
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(te_idx[:n], te_true[:n], label="Thuc te", lw=1.3)
    ax.plot(te_idx[:n], te_pred[:n], label=f"Du bao {args.model}", lw=1.3, alpha=0.85)
    ax.set(xlabel="Thoi gian", ylabel="kW",
           title=f"{args.model} - {customer}: 2 tuan dau tap test")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(base / "figures" / f"test_forecast_{sfx}.png", dpi=130)
    plt.close(fig)


# ======================================================================
# CAC GIAI DOAN
# ======================================================================

def stage_check(args):
    """Kiểm tra môi trường trước khi chạy thật."""
    print("=" * 70)
    print("KIEM TRA MOI TRUONG")
    print("=" * 70)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA co san : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"CUDA build  : {torch.version.cuda}")
        print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("!! Khong thay GPU. Kiem tra lai: ban torch dang cai co the la ban CPU.")
        print("   Cai lai: pip uninstall torch -y")
        print("            pip install torch --index-url https://download.pytorch.org/whl/cu124")
    try:
        import holidays
        print(f"holidays    : {holidays.__version__}")
    except ImportError:
        print("holidays    : CHUA CAI (pip install holidays) -> is_holiday se bang 0")

    cfg = load_config()
    df = load_hourly_data(cfg, customers=cfg["customers"]["hourly_file_columns"])
    tr, va, te = split_by_time(df, cfg)
    print(f"\nDu lieu     : {len(df):,} gio | {df.index.min()} -> {df.index.max()}")
    print(f"Train/Val/Test: {len(tr):,} / {len(va):,} / {len(te):,}")
    print(f"Chuoi       : {list(df.columns)}")
    print("\n-> Moi thu san sang. Chay tiep: python src/train_dl.py --stage baseline")


def stage_baseline(args, splits, cfg):
    """Chạy mốc: 4 chuỗi, 1 seed, cấu hình mặc định."""
    print("\n" + "=" * 70)
    print(f"GIAI DOAN 1 - CHAY MOC ({args.model}, seed {args.seed})")
    print("=" * 70)
    rows = [run_once(c, splits, cfg, args, args.seed)
            for c in cfg["customers"]["hourly_file_columns"]]
    out = pd.DataFrame(rows)
    p = ROOT_DIR / "results" / "dl" / args.model.lower() / "tables" / "baseline_run.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\nDa luu: {p}")
    print(out[["customer", "best_epoch", "val_MAE", "val_MASE", "test_MAE", "test_MASE"]]
          .round(4).to_string(index=False))


def stage_tune(args, splits, cfg):
    """Dò tham số TỪNG CÁI MỘT, chỉ trên chuỗi chính. Chọn theo validation MAE."""
    main = cfg["customers"]["main"]
    print("\n" + "=" * 70)
    print(f"GIAI DOAN 2 - DO THAM SO tren chuoi chinh {main}")
    print("Dò từng tham số một (coordinate descent), KHONG do luoi day du.")
    print("Chon theo VALIDATION MAE. Khong nhin test.")
    print("=" * 70)

    # Thu tu uu tien: lookback quan trong nhat voi LSTM
    # (LSTM phai cuon qua lookback buoc -> chuoi dai lam mo tin hieu buoc dau)
    grid = [
        ("lookback", [24, 48, 168]),
        ("hidden",   [32, 64, 128]),
        ("lr",       [3e-4, 1e-3]),
        ("layers",   [1, 2]),
        ("dropout",  [0.1, 0.2]),
    ]
    cache, history = {}, []

    def evaluate(**over):
        for k, v in over.items():
            setattr(args, k, v)
        key = (args.lookback, args.hidden, args.lr, args.layers, args.dropout)
        if key in cache:
            return cache[key]
        r = run_once(main, splits, cfg, args, args.seed, save_artifacts=False, tag="_tune")
        cache[key] = r["val_MAE"]
        history.append({**{k: v for k, v in zip(
            ["lookback", "hidden", "lr", "layers", "dropout"], key)},
            "val_MAE": r["val_MAE"], "val_MASE": r["val_MASE"],
            "best_epoch": r["best_epoch"]})
        return r["val_MAE"]

    for name, values in grid:
        print(f"\n--- Do '{name}' trong {values} "
              f"(giu cac tham so khac o gia tri tot nhat hien tai) ---")
        scores = {v: evaluate(**{name: v}) for v in values}
        best = min(scores, key=scores.get)
        setattr(args, name, best)
        for v, s in scores.items():
            print(f"    {name}={v:<8} val_MAE = {s:8.4f}" + ("   <-- chon" if v == best else ""))

    final = {"model": args.model, "lookback": args.lookback, "hidden": args.hidden,
             "layers": args.layers, "dropout": args.dropout, "lr": args.lr,
             "calendar": bool(args.calendar), "holiday": bool(args.holiday),
             "tuned_on": main, "n_runs": len(cache)}
    base = ROOT_DIR / "results" / "dl" / args.model.lower() / "tables"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(base / "tuning_log.csv", index=False, encoding="utf-8-sig")
    (base / "best_config.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"CAU HINH TOT NHAT (sau {len(cache)} lan chay):")
    for k, v in final.items():
        print(f"  {k:12} = {v}")
    print(f"\nDa luu: {base/'best_config.json'} va {base/'tuning_log.csv'}")
    print("\nChay tiep:")
    print(f"  python src/train_dl.py --stage final --lookback {args.lookback} "
          f"--hidden {args.hidden} --layers {args.layers} --dropout {args.dropout} "
          f"--lr {args.lr}" + (" --calendar" if args.calendar else ""))


def stage_final(args, splits, cfg):
    """Chạy chính thức: 4 chuỗi × nhiều seed, báo cáo trung bình ± độ lệch chuẩn."""
    seeds = cfg["project"]["seeds"][:args.n_seeds]
    customers = cfg["customers"]["hourly_file_columns"]
    print("\n" + "=" * 70)
    print(f"GIAI DOAN 3 - CHAY CHINH THUC: {len(customers)} chuoi x {len(seeds)} seed "
          f"= {len(customers)*len(seeds)} lan chay")
    print(f"Cau hinh: L={args.lookback} hidden={args.hidden} layers={args.layers} "
          f"drop={args.dropout} lr={args.lr} calendar={bool(args.calendar)}")
    print("=" * 70)

    rows = []
    for i, c in enumerate(customers, 1):
        for j, s in enumerate(seeds, 1):
            print(f"\n[{(i-1)*len(seeds)+j}/{len(customers)*len(seeds)}] {c} seed {s}")
            rows.append(run_once(c, splits, cfg, args, s, tag="_final"))

    df = pd.DataFrame(rows)
    summary = (df.groupby(["model", "customer"])
                 .agg(val_MAE_mean=("val_MAE", "mean"), val_MAE_std=("val_MAE", "std"),
                      test_MAE_mean=("test_MAE", "mean"), test_MAE_std=("test_MAE", "std"),
                      test_RMSE_mean=("test_RMSE", "mean"), test_RMSE_std=("test_RMSE", "std"),
                      test_MASE_mean=("test_MASE", "mean"), test_MASE_std=("test_MASE", "std"),
                      best_epoch_mean=("best_epoch", "mean"), n_seeds=("seed", "count"))
                 .round(4).reset_index())

    base = ROOT_DIR / "results" / "dl" / args.model.lower() / "tables"
    base.mkdir(parents=True, exist_ok=True)
    df.to_csv(base / "final_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(base / "final_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("KET QUA CUOI CUNG (trung binh +- do lech chuan)")
    print("=" * 70)
    for _, r in summary.iterrows():
        print(f"{r['customer']:8} test MAE = {r['test_MAE_mean']:8.3f} +- {r['test_MAE_std']:6.3f}"
              f" | MASE = {r['test_MASE_mean']:.4f} +- {r['test_MASE_std']:.4f}")
    print(f"\nDa luu: {base/'final_summary.csv'}")


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Huan luyen mo hinh hoc sau du bao dien nang t+1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", default="baseline",
                   choices=["check", "baseline", "tune", "final", "single"])
    p.add_argument("--model", default="LSTM", choices=["LSTM", "MLP", "CNNLSTM"])
    p.add_argument("--customer", default=None, help="chi dung cho --stage single")
    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--calendar", action="store_true",
                   help="them dac trung lich (gio, thu, cuoi tuan, ngay le)")
    p.add_argument("--no-holiday", dest="holiday", action="store_false",
                   help="tat rieng cot is_holiday khi da bat --calendar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=5, help="so seed cho --stage final")
    p.add_argument("--verbose-every", type=int, default=10)
    p.set_defaults(holiday=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "check":
        stage_check(args)
        return

    cfg = load_config()
    args._device = get_device()
    df = load_hourly_data(cfg, customers=cfg["customers"]["hourly_file_columns"])
    splits = split_by_time(df, cfg)

    if args.stage == "baseline":
        stage_baseline(args, splits, cfg)
    elif args.stage == "tune":
        stage_tune(args, splits, cfg)
    elif args.stage == "final":
        stage_final(args, splits, cfg)
    elif args.stage == "single":
        c = args.customer or cfg["customers"]["main"]
        run_once(c, splits, cfg, args, args.seed)


if __name__ == "__main__":
    main()
