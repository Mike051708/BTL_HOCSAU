
from pathlib import Path

import pandas as pd


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ DANH SÁCH KHÁCH HÀNG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT_DIR / "data" / "raw" / "LD2011_2014.txt"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
TABLES_DIR = ROOT_DIR / "results" / "tables"

CUSTOMERS = [
    "MT_156",
    "MT_158",
    "MT_159",
    "MT_168",
    "MT_169",
]


# ============================================================
# 2. HÀM LƯU BÁO CÁO CHẤT LƯỢNG DỮ LIỆU
# ============================================================

def save_quality_report(df, output_path):
    rows = []

    for customer in CUSTOMERS:
        series = df[customer]

        rows.append({
            "customer": customer,
            "records_15min": len(series),
            "missing_count": int(series.isna().sum()),
            "zero_count": int(series.eq(0).sum()),
            "negative_count": int(series.lt(0).sum()),
            "mean_15min": series.mean(),
            "std_15min": series.std(),
            "min_15min": series.min(),
            "max_15min": series.max(),
        })

    report = pd.DataFrame(rows)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")

    return report


# ============================================================
# 3. HÀM CHÍNH
# ============================================================

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    quality_path = TABLES_DIR / "selected_customers_quality.csv"
    hourly_report_path = TABLES_DIR / "hourly_preprocessing_report.csv"
    hourly_data_path = PROCESSED_DIR / "series_hourly.csv"

    # --------------------------------------------------------
    # Bước 1: Kiểm tra file dữ liệu gốc
    # --------------------------------------------------------

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu:\n{RAW_PATH}"
        )

    print("=" * 65)
    print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ĐIỆN NĂNG")
    print("=" * 65)

    print("\n[1/6] Đang kiểm tra tên cột...")

    # Chỉ đọc dòng tiêu đề để xác định tên cột.
    header = pd.read_csv(
        RAW_PATH,
        sep=";",
        nrows=0
    )

    columns = header.columns.astype(str).str.strip().tolist()

    if len(columns) < 2:
        raise ValueError(
            "Không đọc được cấu trúc cột của file dữ liệu."
        )

    time_col = columns[0]

    print("Cột thời gian:", repr(time_col))
    print("Số cột trong file:", len(columns))
    print("Một số cột đầu:", columns[:10])

    missing_columns = [
        customer
        for customer in CUSTOMERS
        if customer not in columns
    ]

    if missing_columns:
        raise ValueError(
            "Không tìm thấy các cột khách hàng: "
            + ", ".join(missing_columns)
            + "\nHãy kiểm tra lại tên cột trong file dữ liệu."
        )

    # --------------------------------------------------------
    # Bước 2: Chỉ đọc thời gian và 5 khách hàng cần dùng
    # --------------------------------------------------------

    print("\n[2/6] Đang đọc dữ liệu của 5 khách hàng...")

    # Dữ liệu gốc sử dụng dấu chấm phẩy để phân tách cột.
    # decimal=',' giúp đọc đúng số thập phân dùng dấu phẩy.
    df = pd.read_csv(
        RAW_PATH,
        sep=";",
        usecols=[time_col] + CUSTOMERS,
        decimal=",",
        low_memory=False
    )

    df.columns = df.columns.astype(str).str.strip()

    print("\nNăm dòng đầu sau khi đọc:")
    print(df.head().to_string())

    print("\nKiểu dữ liệu trước khi chuyển đổi:")
    print(df.dtypes.to_string())

    # --------------------------------------------------------
    # Bước 3: Chuẩn hóa thời gian và dữ liệu số
    # --------------------------------------------------------

    print("\n[3/6] Đang chuẩn hóa dữ liệu...")

    df[time_col] = pd.to_datetime(
        df[time_col],
        errors="coerce"
    )

    invalid_time_count = int(df[time_col].isna().sum())

    # Chỉ loại các dòng không có thời gian hợp lệ.
    df = df.dropna(subset=[time_col]).copy()
    df = df.sort_values(time_col)

    for customer in CUSTOMERS:
        # Chuyển dữ liệu sang kiểu số.
        # Giá trị không thể chuyển đổi sẽ thành NaN để kiểm tra,
        # không tự động thay bằng 0.
        df[customer] = pd.to_numeric(
            df[customer],
            errors="coerce"
        )

    df = df.set_index(time_col)
    df.index.name = "timestamp"

    if df.index.has_duplicates:
        duplicate_count = int(df.index.duplicated().sum())

        raise ValueError(
            f"Phát hiện {duplicate_count} timestamp bị trùng. "
            "Cần kiểm tra dữ liệu trước khi tiếp tục."
        )

    print("Số bản ghi:", len(df))
    print("Thời điểm đầu:", df.index.min())
    print("Thời điểm cuối:", df.index.max())
    print("Timestamp không hợp lệ đã loại:", invalid_time_count)

    # --------------------------------------------------------
    # Bước 4: Kiểm tra chất lượng dữ liệu
    # --------------------------------------------------------

    print("\n[4/6] Đang kiểm tra chất lượng dữ liệu...")

    quality_report = save_quality_report(
        df,
        quality_path
    )

    print("\nBáo cáo chất lượng:")
    print(quality_report.to_string(index=False))

    missing_total = int(df[CUSTOMERS].isna().sum().sum())
    negative_total = int(
        df[CUSTOMERS].lt(0).sum().sum()
    )

    if missing_total > 0:
        print(
            "\nĐÃ DỪNG: Phát hiện dữ liệu thiếu."
            "\nBáo cáo đã lưu tại:"
            f"\n{quality_path}"
        )

        raise ValueError(
            f"Có tổng cộng {missing_total} giá trị thiếu "
            "trong 5 chuỗi. Không tự điền dữ liệu khi chưa "
            "xác định nguyên nhân."
        )

    if negative_total > 0:
        print(
            "\nĐÃ DỪNG: Phát hiện giá trị âm."
            f"\nBáo cáo: {quality_path}"
        )

        raise ValueError(
            f"Phát hiện {negative_total} giá trị âm. "
            "Cần kiểm tra trước khi xử lý tiếp."
        )

    # --------------------------------------------------------
    # Bước 5: Tổng hợp dữ liệu 15 phút thành dữ liệu theo giờ
    # --------------------------------------------------------

    print("\n[5/6] Đang tổng hợp dữ liệu theo giờ...")

    # Timestamp trong bộ dữ liệu là thời điểm kết thúc
    # khoảng đo 15 phút.
    # Bốn mốc :15, :30, :45, :00 tạo thành một giờ.
    hourly = df[CUSTOMERS].resample(
        "1h",
        closed="right",
        label="right"
    ).mean()

    hourly_count = df[CUSTOMERS].resample(
        "1h",
        closed="right",
        label="right"
    ).count()

    # Đếm số giờ có ít nhất một khách hàng không đủ 4 bản ghi.
    incomplete_hours = int(
        (hourly_count != 4).any(axis=1).sum()
    )

    # Lưu dữ liệu giờ để dùng cho các mô hình sau này.
    hourly.index.name = "timestamp"

    hourly.reset_index().to_csv(
        hourly_data_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Bước 6: Lưu báo cáo dữ liệu theo giờ
    # --------------------------------------------------------

    print("\n[6/6] Đang tạo báo cáo dữ liệu theo giờ...")

    hourly_report_rows = []

    for customer in CUSTOMERS:
        series = hourly[customer]

        hourly_report_rows.append({
            "customer": customer,
            "records_hourly": len(series),
            "missing_hourly": int(series.isna().sum()),
            "zero_hourly": int(series.eq(0).sum()),
            "mean_hourly": series.mean(),
            "std_hourly": series.std(),
            "min_hourly": series.min(),
            "max_hourly": series.max(),
            "hours_with_not_4_records": incomplete_hours,
        })

    hourly_report = pd.DataFrame(hourly_report_rows)

    hourly_report.to_csv(
        hourly_report_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # In kết quả
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 65)

    print("\nBáo cáo dữ liệu theo giờ:")
    print(hourly_report.to_string(index=False))

    print("\nCác file đã tạo:")

    print(f"1. Chất lượng dữ liệu: {quality_path}")
    print(f"2. Dữ liệu theo giờ: {hourly_data_path}")
    print(f"3. Báo cáo theo giờ: {hourly_report_path}")

    print("\nSố bản ghi theo giờ:", len(hourly))
    print("Số giờ không đủ 4 bản ghi:", incomplete_hours)

    print(
        "\nLưu ý: Các giờ không đủ 4 bản ghi cần được kiểm tra "
        "trước khi quyết định có sử dụng hay không."
    )


if __name__ == "__main__":
    main()
