from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. KHAI BAO DUONG DAN
# ==========================================
ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "raw" / "LD2011_2014.txt"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("KIEM TRA CHAT LUONG VA SO SANH KHACH HANG")
    print("=" * 70)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Khong tim thay file: {DATA_FILE}")

    print("\nDang doc du lieu lon, vui long cho...")

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

    print(f"So moc thoi gian: {len(df):,}")
    print(f"So khach hang: {len(df.columns):,}")
    print(f"Thoi diem dau: {df.index.min()}")
    print(f"Thoi diem cuoi: {df.index.max()}")

    # ==========================================
    # 2. THONG KE CHAT LUONG VA DAC DIEM
    # ==========================================
    print("\nDang tinh thong ke cho tung khach hang...")

    report = pd.DataFrame(index=df.columns)

    report["records_15min"] = len(df)
    report["missing_count"] = df.isna().sum()
    report["missing_percent"] = df.isna().mean() * 100

    report["zero_count"] = df.eq(0).sum()
    report["zero_percent"] = df.eq(0).mean() * 100
    report["negative_count"] = df.lt(0).sum()

    report["mean_15min"] = df.mean()
    report["std_15min"] = df.std()
    report["min_15min"] = df.min()
    report["max_15min"] = df.max()
    report["median_15min"] = df.median()

    # He so bien dong: do muc bien dong tuong doi cua chuoi
    report["cv"] = (
        report["std_15min"] / report["mean_15min"].replace(0, pd.NA)
    )

    # Dieu kien chat luong co ban, khong phai ket luan cuoi cung
    report["quality_pass"] = (
        (report["missing_percent"] == 0)
        & (report["zero_percent"] == 0)
        & (report["negative_count"] == 0)
        & (report["mean_15min"] > 0)
    )

    report.index.name = "customer"
    report = report.sort_values(
        ["quality_pass", "missing_percent", "zero_percent"],
        ascending=[False, True, True],
    )

    quality_file = TABLE_DIR / "customer_quality.csv"
    report.to_csv(quality_file, encoding="utf-8-sig")

    # ==========================================
    # 3. DANH SACH 10 KHACH HANG THAM KHAO
    # ==========================================
    # Uu tien du lieu day du, khong co zero va khong co gia tri am.
    top10 = report.sort_values(
        ["missing_percent", "zero_percent", "negative_count"],
        ascending=[True, True, True],
    ).head(10)

    top_file = TABLE_DIR / "top10_customers.csv"
    top10.to_csv(top_file, encoding="utf-8-sig")

    print("\n--- 10 KHACH HANG THAM KHAO ---")
    print(
        top10[
            [
                "missing_percent",
                "zero_percent",
                "mean_15min",
                "std_15min",
                "cv",
                "quality_pass",
            ]
        ].round(3).to_string()
    )

    # ==========================================
    # 4. SO SANH 5 KHACH HANG DANG DUOC CHON
    # ==========================================
    selected = ["MT_156", "MT_158", "MT_159", "MT_168", "MT_169"]
    existing = [c for c in selected if c in df.columns]

    if existing:
        selected_report = report.loc[existing].copy()
        selected_file = TABLE_DIR / "selected_customers_quality.csv"
        selected_report.to_csv(selected_file, encoding="utf-8-sig")

        print("\n--- SO SANH 5 KHACH HANG DANG DUOC CHON ---")
        print(
            selected_report[
                [
                    "missing_percent",
                    "zero_percent",
                    "mean_15min",
                    "std_15min",
                    "min_15min",
                    "max_15min",
                    "cv",
                    "quality_pass",
                ]
            ].round(3).to_string()
        )

        # Bieu do so sanh luong dien trung binh va do bien dong
        fig, axes = plt.subplots(2, 1, figsize=(12, 9))

        selected_report["mean_15min"].plot(
            kind="bar", ax=axes[0], title="Luong dien trung binh theo khach hang"
        )
        axes[0].set_ylabel("Gia tri trung binh (kW)")
        axes[0].set_xlabel("Khach hang")
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].grid(axis="y", alpha=0.3)

        selected_report["std_15min"].plot(
            kind="bar", ax=axes[1], title="Do bien dong luong dien theo khach hang"
        )
        axes[1].set_ylabel("Do lech chuan (kW)")
        axes[1].set_xlabel("Khach hang")
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        comparison_plot = FIGURE_DIR / "selected_customers_comparison.png"
        plt.savefig(comparison_plot, dpi=150, bbox_inches="tight")
        plt.close()

        # Bieu do chuoi dien nang trong 7 ngay dau cua du lieu
        first_week = df[existing].iloc[:7 * 24 * 4]
        fig, ax = plt.subplots(figsize=(14, 6))

        for customer in existing:
            ax.plot(
                first_week.index,
                first_week[customer],
                label=customer,
                linewidth=1,
            )

        ax.set_title("So sanh 5 chuoi dien nang trong 7 ngay dau")
        ax.set_xlabel("Thoi gian")
        ax.set_ylabel("Cong suat (kW)")
        ax.legend(ncol=3)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()

        series_plot = FIGURE_DIR / "selected_customers_first_week.png"
        plt.savefig(series_plot, dpi=150, bbox_inches="tight")
        plt.close()

    # ==========================================
    # 5. BIEU DO CHAT LUONG TOAN BO KHACH HANG
    # ==========================================
    # Hien thi 20 khach hang co ty le zero thap nhat.
    plot_data = report.sort_values(
        ["zero_percent", "missing_percent"],
        ascending=[True, True],
    ).head(20).sort_values("zero_percent", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_data.index, plot_data["zero_percent"])
    ax.set_title("Ty le gia tri 0 cua 20 khach hang co ty le thap nhat")
    ax.set_xlabel("Ty le gia tri 0 (%)")
    ax.set_ylabel("Khach hang")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    quality_plot = FIGURE_DIR / "customer_quality_top20.png"
    plt.savefig(quality_plot, dpi=150, bbox_inches="tight")
    plt.close()

    # ==========================================
    # 6. TONG KET
    # ==========================================
    print("\n--- TONG KET CHAT LUONG DU LIEU ---")
    print(f"Tong gia tri thieu: {int(df.isna().sum().sum()):,}")
    print(f"Tong gia tri bang 0: {int(df.eq(0).sum().sum()):,}")
    print(f"Tong gia tri am: {int(df.lt(0).sum().sum()):,}")
    print(f"So khach hang dat dieu kien co ban: {int(report['quality_pass'].sum())}")

    print("\nDa cap nhat cac file:")
    print(quality_file)
    print(top_file)
    print(quality_plot)

    if existing:
        print(selected_file)
        print(comparison_plot)
        print(series_plot)

    print("\nHOAN THANH!")


if __name__ == "__main__":
    main()
