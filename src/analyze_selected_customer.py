
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "LD2011_2014.txt"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CUSTOMER = "MT_158"


def main():
    print("Đang đọc dữ liệu...")

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        decimal=",",
        low_memory=False
    )

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[CUSTOMER] = pd.to_numeric(df[CUSTOMER], errors="coerce")

    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)

    series = df[CUSTOMER].dropna()

    # 1. Kiểm tra khoảng thời gian giữa các bản ghi
    time_diff = series.index.to_series().diff()

    print("\n===== KIỂM TRA THỜI GIAN =====")
    print("Thời điểm đầu:", series.index.min())
    print("Thời điểm cuối:", series.index.max())
    print("Số bản ghi:", len(series))
    print("Số timestamp trùng:", series.index.duplicated().sum())
    print("\nCác khoảng cách thời gian thường gặp:")
    print(time_diff.value_counts().head(10))

    # 2. Vẽ tuần đầu tiên có dữ liệu
    first_week = series.iloc[:7 * 96]

    plt.figure(figsize=(14, 5))
    plt.plot(first_week.index, first_week.values, linewidth=0.8)
    plt.title(f"{CUSTOMER} - 7 ngày đầu trong dữ liệu")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị điện năng gốc")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{CUSTOMER}_first_week_detail.png",
        dpi=150
    )
    plt.close()

    # 3. Trung bình theo giờ trong ngày
    hourly_profile = series.groupby(series.index.hour).mean()
    hourly_profile.index.name = "hour"
    hourly_profile.name = "mean_value"
    hourly_profile.to_csv(
        TABLE_DIR / f"{CUSTOMER}_hourly_profile.csv"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_profile.index, hourly_profile.values, marker="o")
    plt.title(f"{CUSTOMER} - Trung bình theo giờ trong ngày")
    plt.xlabel("Giờ trong ngày")
    plt.ylabel("Giá trị điện năng trung bình")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{CUSTOMER}_hourly_profile.png",
        dpi=150
    )
    plt.close()

    # 4. Trung bình theo ngày trong tuần
    daily_profile = series.groupby(series.index.dayofweek).mean()
    daily_profile.index = [
        "Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5",
        "Thứ 6", "Thứ 7", "Chủ nhật"
    ]
    daily_profile.index.name = "day_of_week"
    daily_profile.name = "mean_value"
    daily_profile.to_csv(
        TABLE_DIR / f"{CUSTOMER}_weekly_profile.csv"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(daily_profile.index, daily_profile.values, marker="o")
    plt.title(f"{CUSTOMER} - Trung bình theo ngày trong tuần")
    plt.xlabel("Ngày trong tuần")
    plt.ylabel("Giá trị điện năng trung bình")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"{CUSTOMER}_weekly_profile.png",
        dpi=150
    )
    plt.close()

    print("\nĐã lưu các biểu đồ và bảng phân tích.")
    print("Hãy kiểm tra thư mục results/figures và results/tables.")


if __name__ == "__main__":
    main()
