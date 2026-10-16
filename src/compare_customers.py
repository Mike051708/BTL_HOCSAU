
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "raw" / "LD2011_2014.txt"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"
TABLE_DIR = PROJECT_DIR / "results" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CUSTOMERS = ["MT_156", "MT_158", "MT_159"]


# =========================
# 2. ĐỌC DỮ LIỆU
# =========================
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {DATA_PATH}"
        )

    print("Đang đọc dữ liệu, vui lòng chờ...")

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        decimal=",",
        low_memory=False
    )

    # Chuyển cột thời gian sang datetime
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(
        df[time_col],
        errors="coerce"
    )

    # Chuyển các cột khách hàng thành số
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Loại các dòng không có thời gian hợp lệ
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    print(f"Số dòng dữ liệu: {len(df):,}")
    print(f"Số khách hàng: {len(df.columns) - 1}")
    print(f"Cột thời gian: {time_col}")

    return df, time_col


# =========================
# 3. KIỂM TRA KHÁCH HÀNG
# =========================
def summarize_customers(df):
    available = [c for c in CUSTOMERS if c in df.columns]
    missing = [c for c in CUSTOMERS if c not in df.columns]

    if missing:
        print(f"Cảnh báo: Không tìm thấy các cột {missing}")

    if not available:
        raise ValueError(
            "Không tìm thấy khách hàng cần so sánh. "
            "Hãy kiểm tra tên cột trong file dữ liệu."
        )

    records = []

    for customer in available:
        s = df[customer]
        valid = s.dropna()
        nonzero = valid[valid != 0]

        records.append({
            "customer": customer,
            "total_records": len(s),
            "missing_count": int(s.isna().sum()),
            "missing_percent": round(s.isna().mean() * 100, 3),
            "zero_count": int((s == 0).sum()),
            "zero_percent": round((s == 0).mean() * 100, 3),
            "nonzero_count": int((s != 0).sum()),
            "mean": valid.mean(),
            "std": valid.std(),
            "min": valid.min(),
            "max": valid.max(),
            "median": valid.median(),
            "nonzero_mean": nonzero.mean()
        })

    summary = pd.DataFrame(records)

    output_path = TABLE_DIR / "selected_customers_summary.csv"
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== THỐNG KÊ CÁC KHÁCH HÀNG =====")
    print(summary.round(4).to_string(index=False))
    print(f"\nĐã lưu bảng: {output_path}")

    return available, summary


# =========================
# 4. VẼ BIỂU ĐỒ TUẦN ĐẦU
# =========================
def plot_first_week(df, time_col, customers):
    # 7 ngày, mỗi ngày 96 mốc 15 phút
    first_week = df.iloc[:7 * 96].copy()

    if first_week.empty:
        print("Không đủ dữ liệu để vẽ biểu đồ tuần đầu.")
        return

    plt.figure(figsize=(14, 6))

    for customer in customers:
        plt.plot(
            first_week[time_col],
            first_week[customer],
            label=customer,
            linewidth=1
        )

    plt.title("So sánh điện năng của 3 khách hàng trong tuần đầu")
    plt.xlabel("Thời gian")
    plt.ylabel("Giá trị điện năng trong dữ liệu gốc")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()

    output_path = FIGURE_DIR / "selected_customers_first_week.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Đã lưu biểu đồ tuần đầu: {output_path}")


# =========================
# 5. VẼ CHUỖI DỮ LIỆU ĐẦY ĐỦ
# =========================
def plot_full_series(df, time_col, customers):
    for customer in customers:
        plt.figure(figsize=(14, 5))

        plt.plot(
            df[time_col],
            df[customer],
            linewidth=0.6
        )

        plt.title(f"Chuỗi điện năng đầy đủ - {customer}")
        plt.xlabel("Thời gian")
        plt.ylabel("Giá trị điện năng trong dữ liệu gốc")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = FIGURE_DIR / f"{customer}_full_series.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Đã lưu biểu đồ: {output_path}")


# =========================
# 6. CHẠY CHƯƠNG TRÌNH
# =========================
def main():
    df, time_col = load_data()

    customers, summary = summarize_customers(df)

    plot_first_week(df, time_col, customers)
    plot_full_series(df, time_col, customers)

    print("\nĐÃ HOÀN THÀNH SO SÁNH DỮ LIỆU.")
    print("Lưu ý: Đây là bước khám phá dữ liệu, chưa làm sạch dữ liệu.")


if __name__ == "__main__":
    main()
