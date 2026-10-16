# BTL Học sâu (12INF028) - Chủ đề 7: Dự báo nhu cầu điện năng theo chuỗi thời gian

Bài tập lớn học phần Học sâu, Trường Đại học Thủ đô Hà Nội. Nhóm 5 sinh viên.

## 1. Bài toán

- **Đầu vào:** 24 giá trị kW trung bình theo giờ gần nhất của một khách hàng.
- **Đầu ra:** giá trị kW trung bình của **giờ kế tiếp** (t+1), bài toán hồi quy.
- **Ý nghĩa:** hỗ trợ lập kế hoạch mua điện, cân đối phụ tải. Đây là mô hình thử nghiệm phục vụ học tập, không dùng cho vận hành thực tế.
- **Chỉ số:** MAE, RMSE (kèm MASE hoặc % cải thiện so với baseline).
- **Mô hình:** baseline (Last Value, Moving Average, Seasonal Naive lag 24 và 168), MLP, LSTM.

## 2. Dữ liệu

- Nguồn: [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) (370 khách hàng, 2011-2014, mỗi 15 phút, đơn vị kW). Giấy phép và điều kiện sử dụng: xem trang UCI ở trên; nhóm ghi lại trong báo cáo.
- Tệp gốc `LD2011_2014.txt` (~700 MB) **không** nằm trong Git. Tải bằng `python src/download_data.py` hoặc tải thủ công (hướng dẫn ở đầu file đó) và đặt vào `data/raw/`.
- Dữ liệu gốc dùng dấu `;` phân cách cột và dấu phẩy làm dấu thập phân.
- Dữ liệu chỉ gồm mã khách hàng ẩn danh, thời gian và giá trị đo.

**Quy ước tiền xử lý** (`src/data_preprocessing.py`):

- Gộp 15 phút thành 1 giờ bằng **trung bình** (giữ đơn vị kW), nhãn thời gian là **mốc kết thúc** của giờ (mốc 02:00 gồm 01:15, 01:30, 01:45, 02:00).
- Không điền dữ liệu thiếu, không xóa giá trị 0; chỉ kiểm tra và báo lỗi.
- Kết quả: `data/processed/series_hourly.csv` (35.064 giờ, 5 chuỗi ứng viên).

**Chuỗi được dùng (đề xuất, nhóm xác nhận):** chính **MT_158**; phụ (tùy chọn) **MT_168, MT_169**; dự phòng MT_159; loại MT_156 (nhiều đoạn không đổi). Chi tiết và số liệu: `notebooks/01_data_exploration.ipynb`.

**Chia tập theo thời gian (không xáo trộn), cấu hình trong `configs/config.yaml`:**

| Tập | Đến mốc (bao gồm) | Số giờ | Tỷ lệ |
|---|---|---|---|
| Train | 2014-03-01 00:00 | 27.720 | 79,1% |
| Validation | 2014-08-01 00:00 | 3.672 | 10,5% |
| Test | 2015-01-01 00:00 | 3.672 | 10,5% |

Bộ chuẩn hóa (nếu dùng) chỉ được fit trên tập train. Test chỉ đánh giá một lần sau khi chốt cấu hình.

**Lưu ý dữ liệu:** các giờ nhãn 02:00 của ngày chuyển giờ (DST) có thể bất thường (chạy `python src/dst_check.py`, kết quả ở `results/tables/dst_hours.csv`). Dữ liệu được giữ nguyên, các mốc này dùng để giải thích lỗi khi phân tích lỗi.

## 3. Cấu trúc thư mục

```
BTL_DuBaoDienNang/
├── configs/config.yaml        # đường dẫn, chuỗi, mốc chia tập, seed, tham số huấn luyện
├── data/
│   ├── raw/                   # dữ liệu gốc (không đưa lên Git)
│   └── processed/             # series_hourly.csv, train.csv, validation.csv, test.csv
├── src/
│   ├── config.py              # đọc cấu hình, set_seed
│   ├── download_data.py       # tải dữ liệu gốc
│   ├── data_preprocessing.py  # 15 phút -> theo giờ
│   ├── data.py                # đọc dữ liệu giờ, chia train/validation/test
│   ├── dst_check.py           # liệt kê giờ bất thường ngày chuyển giờ
│   ├── baseline.py            # (đang phát triển) baseline
│   ├── metrics.py             # (đang phát triển) MAE, RMSE, MASE
│   └── check_customers.py, compare_customers.py,
│       analyze_selected_customer.py, data_exploration.py   # script khảo sát ban đầu (đọc dữ liệu gốc)
├── notebooks/01_data_exploration.ipynb
├── models/                    # mô hình đã lưu (lớn thì để Drive, ghi link ở mục 6)
├── results/figures/, results/tables/
├── demo/
├── requirements.txt
└── README.md
```

## 4. Cài đặt và chạy lại từ đầu (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/download_data.py          # hoặc đặt thủ công LD2011_2014.txt vào data/raw/
python src/data_preprocessing.py     # tạo data/processed/series_hourly.csv
python src/data.py                   # chia train/validation/test, tạo results/tables/split_summary.csv
python src/dst_check.py              # tạo results/tables/dst_hours.csv
jupyter notebook notebooks/01_data_exploration.ipynb
```

Nhóm cần chốt framework học sâu (TensorFlow hoặc PyTorch) và ghi phiên bản vào `requirements.txt`; trước khi chốt, kiểm tra framework hỗ trợ phiên bản Python đang dùng.

## 5. Tái lập

- Seed mặc định 42 và danh sách seed cho thí nghiệm nhiều lần ở `configs/config.yaml`; dùng `src/config.py: set_seed`.
- Mọi tham số (mốc chia, chuỗi, cửa sổ, learning rate, batch size, ...) đặt trong `configs/config.yaml`.
- Các mô hình học sâu báo cáo trung bình ± độ lệch chuẩn qua 3-5 seed.

## 6. Kết quả

_Chưa có. Cập nhật sau khi chạy baseline và mô hình học sâu (bảng MAE/RMSE trên validation và test, learning curve, phân tích lỗi, link mô hình)._

## 7. Phần sử dụng mã nguồn và mô hình bên ngoài

Dữ liệu: UCI ElectricityLoadDiagrams20112014. Thư viện: pandas, numpy, matplotlib, seaborn, scikit-learn, PyYAML (và framework học sâu sau khi chốt). Chưa dùng mô hình pretrained. Cập nhật mục này nếu có thay đổi.

## 8. Phân công

| Thành viên | Phần việc chính |
|---|---|
| Minh Thư | Dữ liệu, EDA, tiền xử lý, chọn chuỗi |
| Đăng Long | Baseline, chia tập, bộ đánh giá chung |
| Hồng Sơn | Cửa sổ trượt, LSTM |
| Bảo Quốc | MLP, thử nghiệm cải tiến, phân tích lỗi |
| Gia Long | Demo, tích hợp, README, báo cáo, slide |
