"""Tiện ích dùng chung cho các mô hình học sâu (MLP, LSTM, CNN-LSTM).

Gom ở một chỗ những thứ mọi mô hình đều cần, để các mô hình khác nhau
được huấn luyện bằng ĐÚNG MỘT vòng lặp -> so sánh mới công bằng:

    - set_seed_all   : cố định seed cho random / numpy / torch / cuda
    - make_windows   : cửa sổ trượt
    - calendar_feats : đặc trưng lịch (giờ, thứ, cuối tuần, ngày lễ)
    - fit_model      : vòng lặp huấn luyện có ReduceLROnPlateau, min_delta,
                       gradient clipping và early stopping
    - log_experiment : ghi một dòng vào results/experiments.csv

Vì sao vòng lặp này khác bản đầu tiên của nhóm:
    1. ReduceLROnPlateau : learning rate cố định 1e-3 làm validation dao
       động +-6%, early stopping bắt nhầm nhiễu thành tín hiệu.
    2. min_delta         : cải thiện nhỏ hơn 1e-4 không được tính là tiến bộ.
    3. clip_grad_norm    : chống exploding gradient, bệnh kinh điển của RNN.
    4. patience lớn hơn  : 30 thay vì 10, vì nhiễu validation ~6%.
"""
from __future__ import annotations

import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.config import ROOT_DIR, load_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ROOT_DIR, load_config


# ======================================================================
# SEED VA THIET BI
# ======================================================================

def set_seed_all(seed: int) -> int:
    """Cố định seed cho random, numpy, torch và CUDA.

    src/config.py:set_seed chỉ đặt random và numpy (viết khi nhóm chưa chốt
    framework). Hàm này đặt thêm torch nên dùng cho mọi thí nghiệm học sâu.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_device(verbose: bool = True) -> torch.device:
    """Trả về cuda nếu có, ngược lại cpu."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        if dev.type == "cuda":
            print(f"[Thiet bi] GPU: {torch.cuda.get_device_name(0)} "
                  f"| CUDA {torch.version.cuda} | torch {torch.__version__}")
        else:
            print(f"[Thiet bi] CPU (torch {torch.__version__}). "
                  "LSTM se chay rat cham - kiem tra lai ban cai torch co CUDA.")
    return dev


# ======================================================================
# CUA SO TRUOT
# ======================================================================

def make_windows(values, lookback: int):
    """Cắt cửa sổ trượt từ một mảng 1 chiều.

    Trả về X shape (N, lookback, 1) và y shape (N, 1).
    Mẫu thứ i: X = values[i : i+lookback], y = values[i+lookback].
    Thứ tự thời gian được giữ nguyên, không xáo trộn.
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(v) <= lookback:
        raise ValueError(f"Chuoi dai {len(v)} khong du cho lookback={lookback}.")
    X, y = [], []
    for t in range(lookback, len(v)):
        X.append(v[t - lookback:t])
        y.append(v[t])
    X = np.asarray(X, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def make_eval_windows(context_values, eval_values, lookback: int):
    """Cửa sổ cho validation/test, ghép `lookback` giờ cuối của tập TRƯỚC làm ngữ cảnh.

    Nhờ vậy mọi giá trị lookback đều cho ra đúng len(eval_values) nhãn,
    và không có cửa sổ nào vắt qua ranh giới hai tập.
    Đây là cùng quy ước với src/data.py:get_lookback_context.
    """
    ctx = np.asarray(context_values, dtype=np.float32).reshape(-1)[-lookback:]
    ev = np.asarray(eval_values, dtype=np.float32).reshape(-1)
    if len(ctx) < lookback:
        raise ValueError("Khong du ngu canh cho lookback yeu cau.")
    return make_windows(np.concatenate([ctx, ev]), lookback)


# ======================================================================
# DAC TRUNG LICH
# ======================================================================

CAL_COLUMNS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]


def calendar_feats(ts_index: pd.DatetimeIndex, use_holiday: bool = True) -> np.ndarray:
    """Đặc trưng lịch của GIỜ CẦN DỰ BÁO (không phải của cửa sổ lịch sử).

    Dùng sin/cos thay cho số nguyên 0-23 để mô hình hiểu 23h và 0h liền kề.
    is_holiday dùng lịch nghỉ Bồ Đào Nha (nguồn dữ liệu là Bồ Đào Nha);
    nếu chưa cài gói `holidays` thì cột này bằng 0 và có cảnh báo.

    Trả về mảng (N, 6) theo thứ tự CAL_COLUMNS.
    """
    ts = pd.DatetimeIndex(ts_index)
    h = ts.hour.to_numpy()
    d = ts.dayofweek.to_numpy()

    hol = np.zeros(len(ts), dtype=np.float32)
    if use_holiday:
        try:
            import holidays
            pt = holidays.PT(years=range(int(ts.year.min()), int(ts.year.max()) + 1))
            hol = np.array([ts_i.date() in pt for ts_i in ts], dtype=np.float32)
        except ImportError:
            print("[Canh bao] Chua cai goi `holidays` -> cot is_holiday = 0. "
                  "Cai bang: pip install holidays")

    return np.stack([
        np.sin(2 * np.pi * h / 24.0),
        np.cos(2 * np.pi * h / 24.0),
        np.sin(2 * np.pi * d / 7.0),
        np.cos(2 * np.pi * d / 7.0),
        (d >= 5).astype(np.float32),
        hol,
    ], axis=1).astype(np.float32)


# ======================================================================
# VONG LAP HUAN LUYEN
# ======================================================================

def fit_model(
    model: nn.Module,
    X_train, y_train, cal_train,
    X_val, y_val, cal_val,
    *,
    device,
    max_epochs: int = 300,
    patience: int = 30,
    min_delta: float = 1e-4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    sched_factor: float = 0.5,
    sched_patience: int = 8,
    min_lr: float = 1e-5,
    clip_norm: float = 1.0,
    verbose_every: int = 10,
):
    """Huấn luyện một mô hình, trả về (model_tot_nhat, history_df, best_epoch).

    Trọng số trả về là trọng số tại epoch có validation loss thấp nhất.
    Loss tính trên thang ĐÃ CHUẨN HÓA; metric cuối cùng phải tính lại trên
    thang gốc sau inverse_transform.
    """
    ds = TensorDataset(torch.from_numpy(X_train),
                       torch.from_numpy(cal_train),
                       torch.from_numpy(y_train))
    # shuffle chỉ xáo thứ tự các cửa sổ BÊN TRONG tập train,
    # không làm xáo trộn ranh giới thời gian train/validation/test.
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    Xv = torch.from_numpy(X_val).to(device)
    Cv = torch.from_numpy(cal_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_factor,
        patience=sched_patience, min_lr=min_lr,
    )
    loss_fn = nn.MSELoss()

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for xb, cb, yb in loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, cb), yb)
            loss.backward()
            # chong exploding gradient - bat buoc voi RNN/LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv, Cv), yv).item()

        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_mse_scaled": float(np.mean(losses)),
            "validation_mse_scaled": val_loss,
            "lr": cur_lr,
        })
        scheduler.step(val_loss)

        # min_delta: cai thien nho hon nguong = nhieu, khong tinh la tien bo
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        if verbose_every and epoch % verbose_every == 0:
            print(f"    epoch {epoch:>3} | train {np.mean(losses):.5f} "
                  f"| val {val_loss:.5f} | lr {cur_lr:.2e} | cho {wait}/{patience}")

        if wait >= patience:
            print(f"    -> Early stopping o epoch {epoch} (best = epoch {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist = pd.DataFrame(history)
    print(f"    Xong {len(hist)} epoch trong {time.time()-t0:.0f}s "
          f"| best epoch {best_epoch} | val MSE {best_loss:.5f}")
    return model, hist, best_epoch


@torch.no_grad()
def predict_scaled(model, X, cal, device, batch_size: int = 1024) -> np.ndarray:
    """Dự báo theo lô, trả về mảng 1 chiều trên thang đã chuẩn hóa."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        cb = torch.from_numpy(cal[i:i + batch_size]).to(device)
        out.append(model(xb, cb).cpu().numpy().reshape(-1))
    return np.concatenate(out)


# ======================================================================
# GHI NHAT KY THI NGHIEM
# ======================================================================

def log_experiment(row: dict, path=None) -> Path:
    """Ghi (nối thêm) một dòng vào results/experiments.csv.

    Mỗi lần chạy một dòng. Bảng này trả lời câu hỏi của hội đồng
    "nhóm đã thử bao nhiêu cấu hình" và chứng minh cấu hình cuối cùng
    không phải được chọn tùy tiện.
    """
    path = Path(path) if path else ROOT_DIR / "results" / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, encoding="utf-8-sig")
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False, encoding="utf-8-sig")
    return path
