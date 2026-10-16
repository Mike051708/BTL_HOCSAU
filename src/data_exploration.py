
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Thư mục gốc của dự án
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "raw" / "LD2011_2014.txt"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("KHÁM PHÁ DỮ LIỆU ĐIỆN NĂNG")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {DATA_FILE}"
        )

    print("\n1. Đang đọc dữ liệu...")
    df = pd.read_csv(
        DATA_FILE,
        sep=";",
        index_col=0,
        parse_dates=True,
        decimal=",",
        low_memory=False,
    )

    df.index.name = "timestamp"
    df.columns = df.columns.str.strip().str.strip('"')
    df = df.apply(pd.to_numeric, errors="coerce")

    print("\n2. Kích thước dữ liệu")
    print(f"Số dòng: {df.shape[0]:,}")
    print(f"Số khách hàng/cột: {df.shape[1]:,}")
    print(f"Thời điểm đầu: {df.index.min()}")
    print(f"Thời điểm cuối: {df.index.max()}")

    print("\n3. Kiểm tra dữ liệu thiếu")
    missing = df.isna().sum()
    print(f"Tổng số ô thiếu: {int(missing.sum()):,}")

    summary = pd.DataFrame({
        "missing_count": missing,
        "zero_count": (df == 0).sum(),
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
    })

    summary.to_csv(
        TABLE_DIR / "data_summary.csv",
        encoding="utf-8-sig",
    )

    print("\n4. Thống kê 5 khách hàng đầu tiên")
    print(summary.head().to_string())

    print("\n5. Vẽ biểu đồ dữ liệu")

    # Chọn chuỗi chính (MT_158) để vẽ; cột đầu tiên (MT_001) bằng 0 suốt đầu năm 2011
    # nên không phù hợp để khảo sát.
    MAIN_CUSTOMER = "MT_158"
    if MAIN_CUSTOMER not in df.columns:
        raise ValueError(f"Không tìm thấy cột {MAIN_CUSTOMER}")
    series = df[MAIN_CUSTOMER].dropna()

    plt.figure(figsize=(14, 5))
    plt.plot(series.index, series.values, linewidth=0.7)
    plt.title(f"Chuỗi tiêu thụ điện ban đầu - {series.name}")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị trong dữ liệu gốc")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "01_customer_series_raw.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # Biểu đồ một tuần đầu tiên để quan sát chi tiết
    first_week = series.loc[
        series.index < series.index.min() + pd.Timedelta(days=7)
    ]

    plt.figure(figsize=(14, 5))
    plt.plot(first_week.index, first_week.values, marker=".", linewidth=0.8)
    plt.title(f"Chuỗi điện trong tuần đầu - {series.name}")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị trong dữ liệu gốc")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "02_customer_first_week.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\nĐã lưu bảng thống kê tại:")
    print(TABLE_DIR / "data_summary.csv")

    print("\nĐã lưu biểu đồ tại:")
    print(FIGURE_DIR / "01_customer_series_raw.png")
    print(FIGURE_DIR / "02_customer_first_week.png")

    print("\nHoàn thành khám phá dữ liệu ban đầu!")


if __name__ == "__main__":
    main()
