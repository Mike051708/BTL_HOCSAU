"""Demo: nạp checkpoint .pt và dự báo công suất (kW) của GIỜ KẾ TIẾP.

Checkpoint do `python src/train_dl.py --stage final|single` sinh ra ở models/dl/
(không nằm trong Git). Mỗi tệp đã chứa kiến trúc, siêu tham số, scaler_mean và
scaler_scale nên không cần dữ liệu huấn luyện để đưa kết quả về kW.

Đầu vào là các giờ gần nhất, theo thứ tự thời gian, đủ `lookback` giờ của
checkpoint (MLP: 168, LSTM: 24). Nếu dư thì lấy các giờ cuối.

Cách dùng (không cần --checkpoint thì dùng demo/checkpoints/MLP_MT_162_khong_lich.pt):
    python demo/predict.py                                  # trình diễn nhanh
    python demo/predict.py --csv demo/sample_input.csv      # dữ liệu mới dạng CSV (timestamp, kW)
    python demo/predict.py --values 280 290 ...             # danh sách kW (checkpoint không dùng lịch)
    python demo/predict.py --evaluate-test                  # chấm cả tập test, so với baseline
    python demo/predict.py --checkpoint demo/checkpoints/LSTM_MT_162_co_lich.pt --csv demo/sample_input.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dl_utils import calendar_feats          # noqa: E402
from src.models_dl import build_model            # noqa: E402


def load_checkpoint(path):
    """Nạp checkpoint và dựng lại mô hình ở chế độ đánh giá."""
    # Checkpoint chứa mảng numpy (scaler) nên cần weights_only=False.
    # Chỉ nạp tệp do chính nhóm huấn luyện ra.
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_cal = 6 if ck["calendar"] else 0
    model = build_model(ck["model_name"], lookback=ck["lookback"], n_cal=n_cal,
                        hidden=ck["hidden"], layers=ck["layers"], dropout=ck["dropout"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def read_input(args, ck):
    """Trả về (giá trị kW, timestamp hoặc None)."""
    if args.csv:
        df = pd.read_csv(args.csv, encoding="utf-8-sig")
        if df.shape[1] < 2:
            sys.exit("CSV cần hai cột: timestamp và kW.")
        ts = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]))
        if len(ts) > 1 and not (ts.to_series().diff().dropna() == pd.Timedelta("1h")).all():
            sys.exit("timestamp phải cách đều đúng 1 giờ và tăng dần.")
        return df.iloc[:, 1].to_numpy(dtype=float), ts
    if args.values:
        return np.asarray(args.values, dtype=float), None
    if args.demo_data:
        val = pd.read_csv(ROOT / "data" / "processed" / "validation.csv",
                          encoding="utf-8-sig", parse_dates=["timestamp"])
        col = args.customer or ck.get("customer", "MT_162")
        return val[col].to_numpy(dtype=float), pd.DatetimeIndex(val["timestamp"])
    sys.exit("Cần một trong: --csv, --values, --demo-data.")


def predict_next(model, ck, values, timestamps=None):
    """Dự báo giờ kế tiếp. Trả về (dự báo kW, mốc thời gian được dự báo hoặc None)."""
    L = ck["lookback"]
    if len(values) < L:
        raise ValueError(f"Cần ít nhất {L} giờ lịch sử, chỉ có {len(values)}.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Dữ liệu có giá trị thiếu, vô cực hoặc âm.")

    window = np.asarray(values[-L:], dtype=np.float64)
    scaled = ((window - ck["scaler_mean"][0]) / ck["scaler_scale"][0]).astype(np.float32)
    x = torch.from_numpy(scaled.reshape(1, L, 1))

    target_ts = None
    if ck["calendar"]:
        if timestamps is None:
            raise ValueError("Checkpoint dùng đặc trưng lịch: cần dữ liệu CSV có cột timestamp "
                             "(dòng lệnh: --csv).")
        target_ts = timestamps[-1] + pd.Timedelta(hours=1)
        cal = calendar_feats(pd.DatetimeIndex([target_ts]), use_holiday=ck["holiday"])
    else:
        cal = np.zeros((1, 0), dtype=np.float32)
        if timestamps is not None:
            target_ts = timestamps[-1] + pd.Timedelta(hours=1)

    with torch.no_grad():
        y_scaled = model(x, torch.from_numpy(cal)).item()
    return y_scaled * ck["scaler_scale"][0] + ck["scaler_mean"][0], target_ts


def evaluate_test(model, ck, customer=None):
    """Chấm checkpoint trên TOÀN BỘ tập test (3.672 giờ) và so với baseline Seasonal Naive.

    Dùng đúng quy ước của train_dl.py: ngữ cảnh lấy từ cuối tập validation,
    metric tính trên thang kW gốc. Chỉ để kiểm tra checkpoint, không dùng chọn mô hình.
    """
    from src.config import load_config
    from src.data import load_hourly_data, split_by_time
    from src.dl_utils import make_eval_windows, predict_scaled
    from src.metrics import calculate_d_train, evaluate_forecast

    cfg = load_config()
    col = customer or ck.get("customer", cfg["customers"]["main"])
    df = load_hourly_data(cfg, customers=[col])
    train, val, test = split_by_time(df, cfg)
    mu, sd = ck["scaler_mean"][0], ck["scaler_scale"][0]
    sc = lambda d: ((d[col].to_numpy(dtype=np.float64) - mu) / sd).astype(np.float32)

    X, _ = make_eval_windows(sc(val), sc(test), ck["lookback"])
    cal = (calendar_feats(test.index, use_holiday=ck["holiday"]) if ck["calendar"]
           else np.zeros((len(X), 0), dtype=np.float32))
    pred = predict_scaled(model, X, cal, torch.device("cpu")) * sd + mu

    y = test[col].to_numpy(dtype=float)
    d_train = calculate_d_train(train[col].to_numpy(dtype=float))
    m = evaluate_forecast(y, pred, d_train)

    full = df[col].to_numpy(dtype=float)           # baseline: cùng giờ hôm qua / tuần trước
    pos = df.index.get_indexer(test.index)
    base = {f"Seasonal Naive {k}h": evaluate_forecast(y, full[pos - k], d_train)
            for k in (24, 168)}
    return col, m, base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(Path(__file__).parent / "checkpoints"
                                                / "MLP_MT_162_khong_lich.pt"),
                    help="đường dẫn tệp .pt (mặc định: MLP MT_162 trong demo/checkpoints/)")
    ap.add_argument("--csv", help="CSV: cột 1 timestamp, cột 2 kW")
    ap.add_argument("--values", type=float, nargs="+", help="các giá trị kW liên tiếp")
    ap.add_argument("--demo-data", action="store_true",
                    help="dùng data/processed/validation.csv làm dữ liệu mới")
    ap.add_argument("--evaluate-test", action="store_true",
                    help="chấm checkpoint trên toàn bộ tập test và so với baseline")
    ap.add_argument("--customer", help="cột chuỗi khi dùng --demo-data (mặc định theo checkpoint)")
    args = ap.parse_args()

    model, ck = load_checkpoint(args.checkpoint)

    if args.evaluate_test:
        col, m, base = evaluate_test(model, ck, args.customer)
        print(f"Mô hình : {ck['model_name']} | lookback {ck['lookback']} | "
              f"đặc trưng lịch: {'có' if ck['calendar'] else 'không'} | chuỗi {col}")
        print(f"Tập test: {m['N_samples']} giờ (01/08/2014 → 01/01/2015)")
        print(f"{'':22}{'MAE (kW)':>10}{'RMSE (kW)':>11}{'MASE':>8}")
        for name, r in [(ck["model_name"], m), *base.items()]:
            print(f"{name:22}{r['MAE']:>10.3f}{r['RMSE']:>11.3f}{r['MASE']:>8.3f}")
        return

    if not (args.csv or args.values):
        args.demo_data = True        # không đưa dữ liệu nào thì chạy chế độ trình diễn
    values, ts = read_input(args, ck)
    try:
        y, target = predict_next(model, ck, values, ts)
    except ValueError as e:
        sys.exit(f"Lỗi: {e}")

    print(f"Mô hình : {ck['model_name']} | chuỗi huấn luyện {ck.get('customer', '?')} "
          f"| lookback {ck['lookback']} | đặc trưng lịch: {'có' if ck['calendar'] else 'không'}")
    print(f"Đầu vào : {len(values)} giờ, dùng {ck['lookback']} giờ cuối "
          f"(giờ cuối = {values[-1]:.2f} kW)")
    when = f"giờ nhãn {target:%Y-%m-%d %H:%M}" if target is not None else "giờ kế tiếp"
    print(f"Dự báo  : {y:.2f} kW cho {when}")


if __name__ == "__main__":
    main()
