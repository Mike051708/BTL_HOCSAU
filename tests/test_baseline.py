import numpy as np
import pandas as pd
import pytest

from src.baseline import predict_one
from src.data import (
    create_sliding_window,
    fit_and_save_scaler,
    transform_data,
)
from src.metrics import (
    calculate_d_train,
    calculate_mae,
    calculate_rmse,
)


# --------------------------------------------------
# 1. Kiểm tra 4 baseline
# --------------------------------------------------

def test_predict_one_toy_data():
    """Kiểm tra tính đúng đắn của 4 baseline trên chuỗi giả lập 170 giờ."""
    dates = pd.date_range("2021-01-01", periods=170, freq="h")
    values = np.arange(1, 171, dtype=float)
    series = pd.Series(values, index=dates)

    target_ts = dates[169]
    result = predict_one(series, target_ts)

    assert result["y_true"] == 170.0
    assert result["y_last_value"] == 169.0
    assert result["y_ma3"] == 168.0
    assert result["y_snaive24"] == 146.0
    assert result["y_snaive168"] == 2.0


# --------------------------------------------------
# 2. Kiểm tra metric
# --------------------------------------------------

def test_metrics_calculation():
    """Kiểm tra MAE, RMSE và d_train."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    assert calculate_mae(y_true, y_pred) == pytest.approx(2.3333333)
    assert calculate_rmse(y_true, y_pred) == pytest.approx(2.3804761)

    y_train = np.array([10.0, 15.0, 20.0, 25.0])
    assert calculate_d_train(y_train) == pytest.approx(5.0)


# --------------------------------------------------
# 3. Kiểm tra sliding window
# --------------------------------------------------

@pytest.mark.parametrize(
    "window_size, expected_buffer",
    [
        (24, 144),
        (48, 120),
        (168, 0),
    ],
)
def test_sliding_window_buffer(window_size, expected_buffer):
    """Buffer phải được tính bằng n_context - window_size."""
    n_context = 168
    data = np.arange(200, dtype=float)
    timestamps = pd.date_range(
        "2021-01-01",
        periods=len(data),
        freq="h",
    )

    X, y, target_timestamps = create_sliding_window(
        data_array=data,
        timestamps=timestamps,
        window_size=window_size,
        n_context=n_context,
    )

    # Tổng số cửa sổ ban đầu là len(data) - window_size.
    expected_samples = len(data) - window_size - expected_buffer

    assert len(X) == expected_samples
    assert len(y) == expected_samples
    assert len(target_timestamps) == expected_samples

    # Mốc đầu tiên cần đánh giá phải là timestamp index 168.
    assert target_timestamps[0] == timestamps[168]


def test_sliding_window_target_alignment():
    """X, y và timestamp phải khớp cùng một mốc dự báo."""
    data = np.arange(200, dtype=float)
    timestamps = pd.date_range(
        "2021-01-01",
        periods=len(data),
        freq="h",
    )

    X, y, target_timestamps = create_sliding_window(
        data_array=data,
        timestamps=timestamps,
        window_size=24,
        n_context=168,
    )

    # Với window 24 và buffer 144, X đầu tiên lấy dữ liệu index 144..167.
    np.testing.assert_array_equal(
        np.asarray(X[0]).reshape(-1),
        data[144:168],
    )

    # Nhãn đầu tiên tương ứng index 168.
    assert float(np.asarray(y[0]).reshape(-1)[0]) == 168.0
    assert target_timestamps[0] == timestamps[168]


# --------------------------------------------------
# 4. Kiểm tra scaler chỉ fit trên train
# --------------------------------------------------

def test_scaler_fit_only_on_train(tmp_path):
    """Scaler phải được fit từ train của một chuỗi và lưu riêng theo chuỗi."""
    train_df = pd.DataFrame({
        "MT_158": [10.0, 20.0, 30.0],
    })

    scaler = fit_and_save_scaler(
        train_df=train_df,
        customer="MT_158",
        models_dir=tmp_path,
    )

    # Trung bình scaler phải bằng trung bình train, không phải dữ liệu eval.
    assert scaler.mean_[0] == pytest.approx(20.0)

    # Scaler riêng theo tên chuỗi cần được lưu ra file.
    assert (tmp_path / "scaler_MT_158.pkl").exists()


def test_transform_data_uses_fitted_scaler():
    """Dữ liệu được transform bằng scaler đã fit trên train."""
    train_df = pd.DataFrame({
        "MT_158": [10.0, 20.0, 30.0],
    })

    scaler = fit_and_save_scaler(
        train_df=train_df,
        customer="MT_158",
        models_dir="models",
    )

    transformed = transform_data(train_df, scaler)

    assert transformed.shape == train_df.shape
    assert np.asarray(transformed).mean() == pytest.approx(0.0, abs=1e-7)
    
    
    