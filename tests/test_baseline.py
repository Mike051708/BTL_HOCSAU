import numpy as np
import pandas as pd
import pytest

from src.baseline import predict_one
from src.data import (
    create_sliding_window_for_eval,
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

    X, y, target_timestamps = create_sliding_window_for_eval(
        full_data_array=data,
        full_timestamps=timestamps,
        eval_start_timestamp=timestamps[n_context],
        window_size=window_size,
    )

    # Mọi window_size đều phải cho ra cùng một số nhãn: len(data) - n_context.
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

    X, y, target_timestamps = create_sliding_window_for_eval(
        full_data_array=data,
        full_timestamps=timestamps,
        eval_start_timestamp=timestamps[168],
        window_size=24,
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
    # --------------------------------------------------
# 5. Kiểm tra chia tập theo thời gian
# --------------------------------------------------

def test_split_by_time_no_overlap():
    """Train, validation và test không được chồng lấn thời gian."""
    from src.data import split_by_time

    dates = pd.date_range(
        "2021-01-01",
        periods=10,
        freq="h",
    )

    df = pd.DataFrame(
        {"MT_158": np.arange(10, dtype=float)},
        index=dates,
    )

    cfg = {
        "split": {
            "train_end": "2021-01-01 03:00:00",
            "val_end": "2021-01-01 06:00:00",
            "expected_rows": {
                "train": 4,
                "validation": 3,
                "test": 3,
            },
        }
    }

    train, validation, test = split_by_time(df, cfg)

    assert len(train) == 4
    assert len(validation) == 3
    assert len(test) == 3

    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()

    assert train.index.max() == dates[3]
    assert validation.index.min() == dates[4]
    assert test.index.min() == dates[7]
def test_scaler_does_not_fit_on_validation_or_test(tmp_path):
    """Scaler chỉ fit trên train, không fit validation/test."""
    train_df = pd.DataFrame({
        "MT_158": [10.0, 20.0, 30.0],
    })

    validation_df = pd.DataFrame({
        "MT_158": [100.0, 200.0],
    })

    test_df = pd.DataFrame({
        "MT_158": [1000.0, 2000.0],
    })

    scaler = fit_and_save_scaler(
        train_df=train_df,
        customer="MT_158",
        models_dir=tmp_path,
    )

    # Mean chỉ được tính từ train: (10 + 20 + 30) / 3 = 20
    assert scaler.mean_[0] == pytest.approx(20.0)

    # Validation và test không làm thay đổi scaler đã fit.
    assert scaler.mean_[0] != pytest.approx(validation_df["MT_158"].mean())
    assert scaler.mean_[0] != pytest.approx(test_df["MT_158"].mean())
    
    
    