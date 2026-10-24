import numpy as np
import pandas as pd


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính sai số tuyệt đối trung bình (MAE) trên thang đo gốc."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính căn bậc hai sai số bình phương trung bình (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_d_train(y_train: np.ndarray) -> float:
    """Tính hằng số mẫu số d_train cho MASE từ tập Train gốc[cite: 3].

    d_train = (1 / (T - 1)) * sum(|y_t - y_{t-1}|)
    """
    if len(y_train) < 2:
        raise ValueError("Tập Train cần ít nhất 2 mẫu để tính d_train.")
    d_train = float(np.mean(np.abs(np.diff(y_train))))
    if d_train == 0:
        print("Cảnh báo: d_train = 0. MASE không xác định.")
    return d_train


def calculate_mase(
    y_true: np.ndarray, y_pred: np.ndarray, d_train: float
) -> float:
    """Tính sai số tuyệt đối được chuẩn hóa (MASE)[cite: 3]."""
    if d_train == 0 or np.isnan(d_train):
        return np.nan
    mae = calculate_mae(y_true, y_pred)
    return float(mae / d_train)


def evaluate_forecast(
    y_true: np.ndarray, y_pred: np.ndarray, d_train: float = None
) -> dict:
    """Đánh giá toàn bộ chỉ số MAE, RMSE, MASE[cite: 3]."""
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mase = calculate_mase(y_true, y_pred, d_train) if d_train is not None else np.nan
    return {"MAE": mae, "RMSE": rmse, "MASE": mase, "N_samples": len(y_true)}


def evaluate_by_group(
    df_results: pd.DataFrame,
    pred_col: str,
    group_by: str = "hour",
) -> pd.DataFrame:
    """Đánh giá gom nhóm theo giờ/thứ/tháng để hỗ trợ thành viên phân tích lỗi[cite: 3].

    df_results: DataFrame chứa các cột 'timestamp', 'y_true' và cột dự báo.
    """
    df_eval = df_results.copy()
    if not isinstance(df_eval["timestamp"].dtype, pd.DatetimeIndex):
        df_eval["timestamp"] = pd.to_datetime(df_eval["timestamp"])

    if group_by == "hour":
        groups = df_eval["timestamp"].dt.hour
    elif group_by == "dayofweek":
        groups = df_eval["timestamp"].dt.dayofweek
    elif group_by == "month":
        groups = df_eval["timestamp"].dt.month
    elif group_by == "is_weekend":
        groups = df_eval["timestamp"].dt.dayofweek.isin([5, 6])
    else:
        raise ValueError(f"group_by '{group_by}' không hợp lệ.")

    results = []
    for grp_name, grp_data in df_eval.groupby(groups):
        y_true = grp_data["y_true"].to_numpy()
        y_pred = grp_data[pred_col].to_numpy()
        mae = calculate_mae(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)
        results.append(
            {
                group_by: grp_name,
                "MAE": mae,
                "RMSE": rmse,
                "N_samples": len(grp_data),
            }
        )
    return pd.DataFrame(results)