import numpy as np
import pandas as pd
import pytest
from src.baseline import predict_one
from src.metrics import calculate_d_train, calculate_mae, calculate_rmse


def test_predict_one_toy_data():
    """Kiểm tra tính đúng đắn của 4 Baseline trên chuỗi giả lập 170 giờ[cite: 3]."""
    dates = pd.date_range("2021-01-01", periods=170, freq="h")
    # Tạo chuỗi tăng dần 1, 2, 3...
    values = np.arange(1, 171, dtype=float)
    series = pd.Series(values, index=dates)

    target_ts = dates[169]  # Mốc index 169 (giờ thứ 170)
    result = predict_one(series, target_ts)

    # pos = 169 -> y_true = 170.0
    assert result["y_true"] == 170.0
    # Last Value: pos - 1 -> 169.0
    assert result["y_last_value"] == 169.0
    # MA3: mean(167, 168, 169) -> 168.0
    assert result["y_ma3"] == 168.0
    # Seasonal Naive 24h: pos - 24 -> 146.0
    assert result["y_snaive24"] == 146.0
    # Seasonal Naive 168h: pos - 168 -> 2.0
    assert result["y_snaive168"] == 2.0


def test_metrics_calculation():
    """Kiểm tra công thức MAE, RMSE, d_train[cite: 3]."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    assert calculate_mae(y_true, y_pred) == pytest.approx(2.3333333)
    assert calculate_rmse(y_true, y_pred) == pytest.approx(2.3804761)

    y_train = np.array([10.0, 15.0, 20.0, 25.0])
    assert calculate_d_train(y_train) == pytest.approx(5.0)