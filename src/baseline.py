"""Các phương pháp Baseline cho bài toán dự báo điện năng theo giờ.

Bốn phương pháp:
    1. Last Value
    2. Moving Average 3 giờ
    3. Seasonal Naive 24 giờ
    4. Seasonal Naive 168 giờ

Không chia lại dữ liệu, không sửa dữ liệu nguồn.
Dữ liệu đầu vào phải là chuỗi theo giờ đã được sắp xếp.
"""

import numpy as np
import pandas as pd


BASELINE_COLUMNS = [
    "y_last_value",
    "y_ma3",
    "y_snaive24",
    "y_snaive168",
]


def _validate_series(series):
    """Kiểm tra chuỗi điện năng đầu vào."""
    if not isinstance(series, pd.Series):
        raise TypeError("series phải là pandas.Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Index của series phải là DatetimeIndex.")

    if series.empty:
        raise ValueError("Chuỗi dữ liệu đang rỗng.")

    if series.index.has_duplicates:
        raise ValueError("Chuỗi có timestamp trùng lặp.")

    if not series.index.is_monotonic_increasing:
        raise ValueError("Chuỗi phải được sắp xếp theo thời gian tăng dần.")

    if series.isna().any():
        raise ValueError(
            "Chuỗi có giá trị thiếu. Hãy xử lý theo quy ước dữ liệu của nhóm."
        )

    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError("Chuỗi có giá trị vô cực hoặc không hợp lệ.")

    gaps = series.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Chuỗi không liên tục mỗi giờ. Hãy kiểm tra dữ liệu/DST trước khi chạy."
        )


def predict_one(series, timestamp):
    """Dự báo tại một timestamp bằng bốn phương pháp Baseline.

    series: chuỗi đầy đủ các giá trị thực tế đã quan sát.
    timestamp: thời điểm cần dự báo.

    Giá trị ở timestamp mục tiêu không được dùng làm đầu vào dự báo.
    """
    _validate_series(series)

    timestamp = pd.Timestamp(timestamp)
    pos = series.index.get_indexer([timestamp])[0]

    if pos < 0:
        raise ValueError(f"Không tìm thấy timestamp mục tiêu: {timestamp}")

    if pos < 168:
        raise ValueError(
            f"Không đủ 168 giờ lịch sử trước timestamp {timestamp}."
        )

    values = series.to_numpy(dtype=float)

    return {
        "timestamp": timestamp,
        "y_true": float(values[pos]),
        "y_last_value": float(values[pos - 1]),
        "y_ma3": float(np.mean(values[pos - 3:pos])),
        "y_snaive24": float(values[pos - 24]),
        "y_snaive168": float(values[pos - 168]),
    }


def predict_baselines(series, target_timestamps):
    """Tạo dự báo cho danh sách timestamp mục tiêu.

    Parameters
    ----------
    series : pandas.Series
        Chuỗi đầy đủ theo giờ, index là timestamp.
    target_timestamps : iterable
        Các timestamp mục tiêu thuộc Validation hoặc Test.

    Returns
    -------
    pandas.DataFrame
        Các cột: timestamp, y_true và bốn cột dự báo Baseline.
    """
    # Kiểm tra chuỗi một lần duy nhất
    _validate_series(series)

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(target_timestamps)
    )

    if timestamps.has_duplicates:
        raise ValueError(
            "Danh sách timestamp mục tiêu có phần tử trùng."
        )

    if len(timestamps) == 0:
        raise ValueError(
            "Danh sách timestamp mục tiêu đang rỗng."
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Timestamp mục tiêu phải được sắp xếp tăng dần."
        )

    # Lấy vị trí timestamp mục tiêu trong chuỗi đầy đủ
    positions = series.index.get_indexer(timestamps)

    if (positions < 0).any():
        raise ValueError(
            "Có timestamp mục tiêu không tồn tại trong series."
        )

    # Cần đủ 168 giờ lịch sử cho Seasonal Naive 168
    if (positions < 168).any():
        raise ValueError(
            "Có timestamp không đủ 168 giờ lịch sử."
        )

    values = series.to_numpy(dtype=float)

    # Dự báo trực tiếp bằng NumPy, không gọi predict_one lặp lại
    result = pd.DataFrame({
        "timestamp": timestamps,
        "y_true": values[positions],
        "y_last_value": values[positions - 1],
        "y_ma3": (
            values[positions - 1]
            + values[positions - 2]
            + values[positions - 3]
        ) / 3,
        "y_snaive24": values[positions - 24],
        "y_snaive168": values[positions - 168],
    })

    return result[
        ["timestamp", "y_true", *BASELINE_COLUMNS]
    ]

def get_common_eval_timestamps(series, target_timestamps):
    """Trả về timestamp có đủ lịch sử cho cả bốn Baseline.

    Chỉ lọc timestamp mục tiêu; không đưa lịch sử context
    vào tập nhãn dùng để tính metric.
    """
    _validate_series(series)

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(target_timestamps)
    )

    if timestamps.has_duplicates:
        raise ValueError(
            "Danh sách timestamp mục tiêu có phần tử trùng."
        )

    if len(timestamps) == 0:
        return timestamps

    positions = series.index.get_indexer(timestamps)

    # Timestamp phải tồn tại trong series và có đủ 168 giờ lịch sử
    valid = (positions >= 168)

    return timestamps[valid]
