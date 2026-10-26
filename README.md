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
- Kết quả: `data/processed/series_hourly.csv` (35.064 giờ, 4 chuỗi được chọn).

**Cách chọn chuỗi** (`src/select_customers.py`, ngưỡng ở mục `selection` trong `configs/config.yaml`):

| Tầng | Lọc | Còn lại |
|---|---|---|
| A | chất lượng dữ liệu thô: thiếu, âm, bắt đầu muộn, giá trị 0 giữa chuỗi | 370 → 90 |
| C | đặc trưng theo giờ: đoạn phẳng, cv, chu kỳ tuần, dịch mức tải | 90 → 78 |
| D1 | chuỗi chính: `ac_lag168` cao nhất trong nhóm `ac_lag24 < 0,90` | 78 → 1 |
| D2 | chuỗi phụ trải đều theo mức tải | 78 → 3 |

**Chuỗi được dùng (4 chuỗi, đề bài cho phép 1–5):**

| Chuỗi | Mức tải | Vai trò |
|---|---|---|
| **MT_158** | 76 kW | **chuỗi chính** — 1 trong 2 chuỗi (trên 78) có `ac_lag24 < 0,90`, tức Seasonal Naive 24h chưa chiếm hết dư địa cải thiện |
| MT_169 | 40 kW | tổng quát quy mô nhỏ + bằng chứng DST |
| MT_168 | 137 kW | tổng quát cùng quy mô + bằng chứng DST rõ nhất |
| MT_294 | 641 kW | tổng quát ở quy mô gấp ~8,5 lần chuỗi chính |

Dự phòng cho chuỗi chính: MT_180 (`ac_lag24` = 0,865). Loại: MT_156 (đoạn không đổi 215 giờ), MT_159 (`year_drift` 0,83). Bằng chứng: `results/tables/customer_main_candidates.csv`, `customer_selection.csv`, `customer_shortlist.csv`. Chi tiết: `notebooks/01_data_exploration.ipynb`.

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
│   ├── select_customers.py    # chọn chuỗi theo 4 tầng tiêu chí (370 -> 4)
│   ├── data_preprocessing.py  # 15 phút -> theo giờ, cho các chuỗi đã chọn
│   ├── data.py                # đọc dữ liệu giờ, chia train/validation/test
│   ├── dst_check.py           # liệt kê giờ bất thường ngày chuyển giờ
│   ├── baseline.py            # Last Value, Moving Average, Seasonal Naive 24/168
│   └── metrics.py             # MAE, RMSE, MASE, gom nhóm sai số theo giờ/thứ/tháng
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
python src/select_customers.py       # chọn chuỗi (370 -> 4), tạo 4 bảng trong results/tables/
python src/data_preprocessing.py     # tạo data/processed/series_hourly.csv cho các chuỗi đã chọn
python src/data.py                   # chia train/validation/test, tạo results/tables/split_summary.csv
python src/dst_check.py              # tạo results/tables/dst_hours.csv
python -m pytest tests -q            # kiểm tra baseline và metric
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_baseline.ipynb
```

`select_customers.py` đọc file gốc 700 MB nên chạy chậm; lần sau chỉ cần đổi ngưỡng rồi chạy
`python src/select_customers.py --cached` (dùng lại `data/processed/candidates_hourly.csv`, vài giây).
Chỉ chạy lại `data_preprocessing.py` khi `customers.hourly_file_columns` trong config thay đổi.

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
