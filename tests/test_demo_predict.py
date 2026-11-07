"""Kiểm tra demo/predict.py bằng checkpoint nhỏ tự dựng trong thư mục tạm."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="Cần PyTorch để kiểm tra demo/predict.py")
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT))

import predict  # noqa: E402
from src.models_dl import build_model  # noqa: E402


def _ckpt(tmp_path, name="MLP", lookback=24, calendar=False):
    torch.manual_seed(0)
    model = build_model(name, lookback=lookback, n_cal=6 if calendar else 0,
                        hidden=8, layers=1, dropout=0.0)
    path = tmp_path / "ck.pt"
    torch.save({"model_name": name, "customer": "MT_X", "lookback": lookback,
                "hidden": 8, "layers": 1, "dropout": 0.0, "calendar": calendar,
                "holiday": False, "model_state_dict": model.state_dict(),
                "scaler_mean": np.array([100.0]), "scaler_scale": np.array([20.0])}, path)
    return path


def test_du_bao_dung_thang_do_va_mot_so(tmp_path):
    model, ck = predict.load_checkpoint(_ckpt(tmp_path))
    y, _ = predict.predict_next(model, ck, np.full(24, 100.0))
    assert np.isfinite(y)
    # đầu vào bằng trung bình scaler -> vector 0 -> kết quả = đầu ra mạng * 20 + 100
    with torch.no_grad():
        raw = model(torch.zeros(1, 24, 1), torch.zeros(1, 0)).item()
    assert y == pytest.approx(raw * 20.0 + 100.0, rel=1e-5)


def test_chi_dung_lookback_gio_cuoi(tmp_path):
    model, ck = predict.load_checkpoint(_ckpt(tmp_path))
    ngan = np.arange(24, dtype=float) + 90
    dai = np.concatenate([np.full(50, 5000.0), ngan])    # phần đầu phải bị bỏ qua
    assert predict.predict_next(model, ck, dai)[0] == pytest.approx(
        predict.predict_next(model, ck, ngan)[0])


def test_thieu_gio_hoac_du_lieu_xau_bi_tu_choi(tmp_path):
    model, ck = predict.load_checkpoint(_ckpt(tmp_path))
    with pytest.raises(ValueError):
        predict.predict_next(model, ck, np.ones(5))
    with pytest.raises(ValueError):
        predict.predict_next(model, ck, np.array([1.0] * 23 + [np.nan]))
    with pytest.raises(ValueError):
        predict.predict_next(model, ck, np.array([1.0] * 23 + [-3.0]))


def test_checkpoint_co_lich_can_timestamp(tmp_path):
    model, ck = predict.load_checkpoint(_ckpt(tmp_path, "LSTM", calendar=True))
    with pytest.raises(ValueError):
        predict.predict_next(model, ck, np.full(24, 100.0))
    ts = pd.date_range("2014-08-01 01:00", periods=24, freq="h")
    y, target = predict.predict_next(model, ck, np.full(24, 100.0), ts)
    assert np.isfinite(y) and target == ts[-1] + pd.Timedelta(hours=1)
