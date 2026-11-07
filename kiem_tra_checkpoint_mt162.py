"""Kiem tra 5 checkpoint MLP MT_162 (hidden=128) co cho ra dung so da bao cao.

CHAY: dat tep nay o THU MUC GOC cua repo BTL_HOCSAU, dat 5 tep .pt vao
models/dl/, roi chay:

    python kiem_tra_checkpoint_mt162.py

Ket qua dung phai la:  MAE = 20.6384   RMSE = 31.2840  (trung binh 5 seed)

Script chi NAP checkpoint va suy luan, khong huan luyen lai. Moi checkpoint
tu chua scaler nen khong can tep scaler rieng.
"""
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data import load_hourly_data, split_by_time
from src.dl_utils import make_eval_windows
from src.models_dl import build_model

CUS = "MT_162"
MONG_DOI_MAE = 20.6384
MONG_DOI_RMSE = 31.2840

cfg = load_config()
df = load_hourly_data(cfg, customers=cfg["customers"]["hourly_file_columns"])
_, val_df, test_df = split_by_time(df, cfg)

print("=" * 76)
print(f"KIEM TRA CHECKPOINT {CUS} (nap lai, khong train)")
print("=" * 76)

maes, rmses = [], []
for seed in range(42, 47):
    p = Path("models") / "dl" / f"MLP_{CUS}_s{seed}.pt"
    if not p.exists():
        raise SystemExit(f"Khong thay {p}. Dat 5 tep .pt vao models/dl/ truoc.")
    ck = torch.load(p, map_location="cpu", weights_only=False)

    if ck["hidden"] != 128:
        raise SystemExit(
            f"{p.name}: hidden = {ck['hidden']}, khong phai 128.\n"
            "Day la checkpoint cua CAU HINH CU. Lop dau phai co dang (256, 168)."
        )

    L = ck["lookback"]
    model = build_model(ck["model_name"], lookback=L, n_cal=0, hidden=ck["hidden"],
                        layers=ck["layers"], dropout=ck["dropout"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    mean, scale = float(ck["scaler_mean"][0]), float(ck["scaler_scale"][0])
    X, y = make_eval_windows((val_df[CUS].to_numpy() - mean) / scale,
                             (test_df[CUS].to_numpy() - mean) / scale, L)
    cal = np.zeros((len(X), 0), dtype=np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(X), torch.from_numpy(cal)).numpy().reshape(-1)

    pred = pred * scale + mean
    true = y.reshape(-1) * scale + mean
    mae = float(np.abs(true - pred).mean())
    rmse = float(np.sqrt(((true - pred) ** 2).mean()))
    maes.append(mae)
    rmses.append(rmse)
    print(f"  seed {seed}: MAE = {mae:8.4f}   RMSE = {rmse:8.4f}   "
          f"(hidden={ck['hidden']}, L={L}, calendar={ck['calendar']})")

mae_tb, rmse_tb = float(np.mean(maes)), float(np.mean(rmses))
print()
print(f"  Trung binh 5 seed :  MAE = {mae_tb:.4f}   RMSE = {rmse_tb:.4f}")
print(f"  So trong bao cao  :  MAE = {MONG_DOI_MAE:.4f}   RMSE = {MONG_DOI_RMSE:.4f}")
print()
ok = abs(mae_tb - MONG_DOI_MAE) < 5e-4 and abs(rmse_tb - MONG_DOI_RMSE) < 5e-4
print("  ->", "KHOP. Checkpoint dung, dung duoc cho demo."
      if ok else "LECH. Kiem tra lai du lieu va phien ban thu vien.")
print("=" * 76)
