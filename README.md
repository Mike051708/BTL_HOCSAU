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

### Ý nghĩa của giá trị 0 trong dữ liệu

Bộ dữ liệu có hai loại số 0, **ý nghĩa khác hẳn nhau**, và nhóm xử lý khác nhau:

| Loại | Ý nghĩa | Cách xử lý |
|---|---|---|
| Số 0 **liên tiếp ở đầu** chuỗi | Khách hàng chưa tồn tại trong giai đoạn đó (theo mô tả bộ dữ liệu, các khách hàng được đưa vào ở những thời điểm khác nhau) | **Dùng để loại chuỗi** ở tầng A |
| Số 0 **nằm giữa** chuỗi | Tiêu thụ thật bằng 0 - công tơ đã hoạt động và đo được 0. Đây là một quan sát hợp lệ | **Không dùng để loại**, giữ nguyên giá trị |

Lý do loại chuỗi bắt đầu muộn: mốc chia tập là ngày cố định chung cho mọi chuỗi. Giữ một khách hàng chỉ có dữ liệu từ 2012 thì tập train của khách hàng đó ngắn hơn, tỷ lệ 80/10/10 không còn đúng và so sánh giữa các chuỗi mất công bằng.

Lý do **không** loại theo số 0 giữa chuỗi: bỏ một quan sát hợp lệ chỉ vì nó bằng 0 là làm sai bản chất dữ liệu. Rủi ro thật sự cần tránh là chuỗi có đoạn 0 kéo dài (công tơ hỏng, giá trị bị giữ nguyên), và rủi ro đó được tầng C bắt bằng `const_run_max_h` và `cv` - nơi tiêu chí có lý do rõ ràng hơn. Thực tế tầng C loại 19 chuỗi vì đoạn giá trị không đổi quá dài.

Trong 158 chuỗi qua tầng A có 68 chuỗi chứa số 0 giữa chuỗi; cả 68 chuỗi đều được giữ lại để xét tiếp. Chuỗi chính MT_162 là một trong số đó (45 giá trị 0 giữa chuỗi).

**Cách chọn chuỗi** (`src/select_customers.py`, ngưỡng ở mục `selection` trong `configs/config.yaml`):

| Tầng | Lọc | Còn lại |
|---|---|---|
| A | chất lượng dữ liệu thô: thiếu, âm, trung bình dương, **bắt đầu muộn** | 370 → 158 |
| B | gộp theo giờ và tính đặc trưng (không lọc) | 158 |
| C | đặc trưng theo giờ: đoạn phẳng, cv, chu kỳ tuần, dịch mức tải, lệch test/train | 158 → 122 |
| D1 | chuỗi chính: `ac_lag168` cao nhất trong nhóm `ac_lag24 < 0,90` | 122 → 1 |
| D2 | chuỗi phụ trải đều theo mức tải | 122 → 3 |

Tầng A loại 212 chuỗi, tất cả vì **một lý do duy nhất**: `full_from_2011` (bắt đầu muộn). Tầng C loại 36 chuỗi, nhiều nhất là 19 chuỗi vì đoạn giá trị không đổi quá dài.

**Vì sao tiêu chí chuỗi chính là `ac_lag24 < 0,90`:** tiêu chí này được đặt ra **trước khi chạy bất kỳ mô hình nào**. Nếu `ac_lag24` quá cao thì "giá trị cùng giờ hôm qua" đã bám sát chuỗi, baseline Seasonal Naive 24h gần như tối ưu và mô hình học sâu không còn dư địa cải thiện - bài toán mất ý nghĩa thực nghiệm. Chỉ **4 trên 122** chuỗi thỏa điều kiện này; phần lớn khách hàng trong bộ dữ liệu có chu kỳ ngày lặp gần như máy móc.

**Chuỗi được dùng (4 chuỗi, đề bài cho phép 1-5):**

| Chuỗi | Mức tải | `cv` | `ac_lag24` | `ac_lag168` | Vai trò |
|---|---|---|---|---|---|
| MT_234 | 88 kW | 0,338 | 0,939 | 0,925 | tổng quát, quy mô nhỏ |
| MT_217 | 214 kW | 0,407 | 0,968 | 0,954 | tổng quát, quy mô trung bình |
| **MT_162** | **289 kW** | **0,500** | **0,569** | **0,946** | **chuỗi chính** - hạng 1/4 theo `ac_lag168` trong nhóm `ac_lag24 < 0,90` |
| MT_314 | 1.121 kW | 0,288 | 0,967 | 0,947 | tổng quát, quy mô gấp ~3,9 lần chuỗi chính |

Dự phòng cho chuỗi chính: **MT_124** (`ac_lag24` = 0,892). Hai ứng viên còn lại MT_158 và MT_180 xếp sau theo `ac_lag168`.

**Kiểm tra độ nhạy của ngưỡng.** Sáu ngưỡng ở tầng C và D1 là lựa chọn của nhóm, không có cơ sở lý thuyết quy định giá trị cụ thể. Khi không chứng minh được một con số là đúng, nhóm chứng minh **kết luận không phụ thuộc** vào con số đó: chạy lại toàn bộ quy trình với 37 bộ ngưỡng (mỗi lần đổi một ngưỡng trong dải rộng), **36/37 bộ vẫn chọn MT_162**. Trường hợp duy nhất khác là khi nới `main_max_ac_lag24` lên 0,95 - khi đó số chuỗi đủ điều kiện nhảy từ 4 lên 62 trên 122, tức tiêu chí không còn lọc nữa. Chạy bằng `python kiem_tra_do_nhay.py`, kết quả ở `results/tables/do_nhay_nguong.csv`. Hạn chế: đây là phép thử một-lần-một-biến, không bao phủ mọi tổ hợp.

Bằng chứng: `results/tables/customer_main_candidates.csv`, `customer_selection.csv`, `customer_shortlist.csv`, `customer_features_hourly.csv`, `customer_quality_all.csv`. Chi tiết: `notebooks/01_data_exploration.ipynb`.

**Chia tập theo thời gian (không xáo trộn), cấu hình trong `configs/config.yaml`:**

| Tập | Đến mốc (bao gồm) | Số giờ | Tỷ lệ |
|---|---|---|---|
| Train | 2014-03-01 00:00 | 27.720 | 79,1% |
| Validation | 2014-08-01 00:00 | 3.672 | 10,5% |
| Test | 2015-01-01 00:00 | 3.672 | 10,5% |

Bộ chuẩn hóa (nếu dùng) chỉ được fit trên tập train, **mỗi chuỗi một scaler riêng** để còn `inverse_transform` đúng về thang đo gốc. Test chỉ đánh giá một lần sau khi chốt cấu hình.

**Hạn chế cần nêu trong báo cáo:** mức tải của MT_162 theo tập là train 293,5 kW - validation 255,1 kW - test 285,0 kW. Tập validation thấp hơn train khoảng **13%** trong khi test chỉ thấp hơn 3%, nên mô hình chọn theo validation được tinh chỉnh cho một chế độ tải khác chế độ của test. Đây là hệ quả tất yếu của việc chia theo thời gian; đổi lại, cách chia này mô phỏng đúng tình huống dùng quá khứ dự báo tương lai và loại bỏ rò rỉ dữ liệu.

**Lưu ý dữ liệu:** các giờ nhãn 02:00 của ngày chuyển giờ (DST) có thể bất thường (chạy `python src/dst_check.py`, kết quả ở `results/tables/dst_hours.csv`). Ngưỡng đánh dấu **không đặt cứng**: với mỗi chuỗi, script tính tỷ lệ `giờ 02:00 / trung bình giờ 01:00 và 03:00` cho toàn bộ ~1.453 ngày thường rồi lấy phân vị 1 và 99 của chính chuỗi đó làm cận. Phân vị 1 được chọn vì số lần báo nhầm kỳ vọng mỗi chuỗi là 8 × 0,02 = 0,16 lần, dưới 1 (phân vị 5 cho 0,80 lần, quá lỏng). Cả bốn chuỗi đều có 6 trên 8 ngày chuyển giờ bị đánh dấu. Dữ liệu được giữ nguyên, các mốc này dùng làm giả thuyết giải thích lỗi khi phân tích lỗi.

Mô tả cột của mọi tệp dữ liệu và bảng kết quả: [docs/data_dictionary.md](docs/data_dictionary.md)

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
│   ├── data.py                # đọc dữ liệu giờ, chia train/validation/test, cửa sổ trượt
│   ├── dst_check.py           # liệt kê giờ bất thường ngày chuyển giờ
│   ├── baseline.py            # Last Value, Moving Average, Seasonal Naive 24/168
│   └── metrics.py             # MAE, RMSE, MASE, gom nhóm sai số theo giờ/thứ/tháng
├── notebooks/01_data_exploration.ipynb
├── docs/data_dictionary.md    # mô tả cột của mọi tệp dữ liệu và bảng kết quả
├── kiem_tra_do_nhay.py        # kiểm tra độ nhạy của ngưỡng chọn chuỗi
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
python src/select_customers.py       # chọn chuỗi (370 -> 4), tạo 5 bảng trong results/tables/
python src/data_preprocessing.py     # tạo data/processed/series_hourly.csv cho các chuỗi đã chọn
python src/data.py                   # chia train/validation/test, tạo results/tables/split_summary.csv
python src/dst_check.py              # tạo results/tables/dst_hours.csv
python kiem_tra_do_nhay.py           # kiểm tra độ nhạy ngưỡng (cần cache, xem ghi chú dưới)
python -m pytest tests -q            # kiểm tra baseline và metric
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_baseline.ipynb
```
### Phần học sâu (LSTM / MLP)

Huấn luyện chạy trên **Google Colab, GPU Tesla T4** (Python 3.12, PyTorch 2.11 + CUDA 13).
Máy cá nhân không có GPU vẫn chạy được nhưng chậm hơn khoảng 8-10 lần.

```powershell
python src/train_dl.py --stage check       # kiểm tra GPU, thư viện và dữ liệu
python src/train_dl.py --stage baseline    # chạy mốc: 4 chuỗi, 1 seed, cấu hình mặc định
python src/train_dl.py --stage tune        # dò 12 cấu hình trên chuỗi chính MT_162
python src/train_dl.py --stage final --lookback 24 --hidden 128 --layers 2 --n-seeds 5
python src/train_dl.py --stage final --model MLP --lookback 168 --hidden 64 --dropout 0.1 --n-seeds 5
```

Thêm `--calendar` để bật đặc trưng lịch (giờ, thứ, cuối tuần, ngày lễ Bồ Đào Nha);
thêm `--no-holiday` để tắt riêng cột ngày lễ. Chạy bốn tổ hợp mô hình × có/không
đặc trưng lịch sẽ cho ma trận ablation 2×2.

Mọi lần chạy được ghi nối thêm vào `results/experiments.csv`.

### Biểu đồ và phân tích lỗi (không cần GPU)

```powershell
python src/training_figures.py             # learning curve, dò tham số, hội tụ - LSTM
python src/training_figures.py --model MLP # tương tự cho MLP
python src/error_analysis.py               # so sánh mô hình và phân tích lỗi
```

Kết quả vào `results/analysis_train/<model>/` và `results/analysis/`.
Hai script này chỉ cần `pandas`, `numpy`, `matplotlib`, `pyyaml`, `holidays` - **không cần PyTorch**.
```
---


`select_customers.py` đọc file gốc 700 MB nên chạy chậm; lần sau chỉ cần đổi ngưỡng rồi chạy
`python src/select_customers.py --cached` (dùng lại `data/processed/candidates_hourly.csv`, vài giây).
Chỉ chạy lại `data_preprocessing.py` khi `customers.hourly_file_columns` trong config thay đổi.

**Lưu ý khi sửa file trong `src/`:** phải **Restart Kernel** trong Jupyter trước khi chạy lại notebook. Python nạp một module một lần rồi cache lại, nên notebook sẽ chạy bằng code cũ mà không báo gì.

Nhóm cần chốt framework học sâu (TensorFlow hoặc PyTorch) và ghi phiên bản vào `requirements.txt`; trước khi chốt, kiểm tra framework hỗ trợ phiên bản Python đang dùng.

## 5. Tái lập

- Seed mặc định 42 và danh sách seed cho thí nghiệm nhiều lần ở `configs/config.yaml`; dùng `src/config.py: set_seed`.
- Mọi tham số (mốc chia, chuỗi, cửa sổ, learning rate, batch size, ...) đặt trong `configs/config.yaml`.
- Các mô hình học sâu báo cáo trung bình ± độ lệch chuẩn qua 3-5 seed.
- Dữ liệu thô được đọc ở **float64**. Không dùng float32: `const_run_max_h` so sánh giá trị bằng tuyệt đối, nên sai số float32 (~2×10⁻⁷ tương đối) làm hai giá trị khác nhau bị coi là giống nhau và sinh ra "đoạn không đổi" không có trong dữ liệu thật.

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
