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
│   ├── metrics.py             # MAE, RMSE, MASE, gom nhóm sai số theo giờ/thứ/tháng
│   ├── models_dl.py           # kiến trúc LSTM, MLP, CNN-LSTM (cùng chữ ký forward)
│   ├── dl_utils.py            # seed, thiết bị, cửa sổ, đặc trưng lịch, vòng huấn luyện
│   ├── train_dl.py            # CLI: check / baseline / tune / final / single
│   ├── training_figures.py    # learning curve, mức ảnh hưởng tham số, hội tụ
│   └── error_analysis.py      # 23 hình + 19 bảng phân tích lỗi cho báo cáo
├── tests/                     # test_baseline.py, test_dl_utils.py (pytest)
├── notebooks/                 # 01_data_exploration, 02_baseline, 03_mlp_baseline_4_series
├── docs/
│   ├── data_dictionary.md     # mô tả cột của mọi tệp dữ liệu và bảng kết quả
│   └── *.docx                 # báo cáo Word (KHÔNG đưa lên Git, nộp/chia sẻ riêng)
├── kiem_tra_do_nhay.py        # kiểm tra độ nhạy của ngưỡng chọn chuỗi
├── models/                    # scaler_<chuỗi>.pkl (lên Git) và dl/*.pt (không lên Git)
├── results/
│   ├── experiments.csv        # SỔ CÁI: một dòng một lần chạy, chỉ thêm không ghi đè
│   ├── <chuỗi>/               # baseline của từng chuỗi
│   ├── dl/<model>/            # tables, predictions, figures của mô hình học sâu
│   ├── analysis/              # phân tích lỗi (error_analysis.py)
│   ├── analysis_train/<model>/# phân tích quá trình huấn luyện (training_figures.py)
│   └── figures/, tables/      # EDA và baseline giai đoạn đầu
├── requirements.txt
└── README.md
```

Phân biệt hai thư mục dễ lẫn: `results/analysis/` trả lời **mô hình sai ở đâu, khi nào**
(theo giờ, theo thứ, ngày lễ, top-10 lỗi); `results/analysis_train/` trả lời **mô hình học
thế nào** (learning curve, hội tụ, mức ảnh hưởng tham số).

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

LSTM huấn luyện trên **Google Colab, GPU Tesla T4** (Python 3.12, PyTorch 2.11 + CUDA 13).
Máy cá nhân không có GPU vẫn chạy được nhưng chậm hơn khoảng 8-10 lần.
Riêng **MLP đủ nhẹ để chạy trên CPU**: khoảng 77 giây một lần chạy trên 12 lõi
(PyTorch 2.15 bản CPU), nên cả giai đoạn `tune` (~10 phút) và `final` (~40 phút)
không cần GPU.

```powershell
python src/train_dl.py --stage check       # kiểm tra GPU, thư viện và dữ liệu
python src/train_dl.py --stage baseline    # chạy mốc: 4 chuỗi, 1 seed, cấu hình mặc định

# Dò tham số: lưới 12 điểm theo tọa độ, sinh ra 8 lần huấn luyện khác nhau
# (giá trị tốt nhất hiện tại được giữ lại nên các điểm trùng lấy từ cache)
python src/train_dl.py --stage tune
python src/train_dl.py --stage tune --model MLP --max-epochs 120 --patience 20

# Chạy chính thức với cấu hình đã dò (xem results/dl/<model>/tables/best_config.json)
python src/train_dl.py --stage final --lookback 24 --hidden 128 --layers 2 --n-seeds 5
python src/train_dl.py --stage final --model MLP --lookback 168 --hidden 128 --dropout 0.2 --lr 0.001 --max-epochs 120 --patience 20 --n-seeds 5
```

> **Cảnh báo:** tên tệp kết quả chỉ gồm `{chuỗi}_s{seed}`, không mang dấu hiệu giai đoạn.
> Vì vậy `--stage baseline` và `--stage single` **ghi đè** tệp dự báo, history và biểu đồ
> của `--stage final` ở cùng seed. Sao lưu `results/dl/<model>/` trước khi chạy hai giai
> đoạn đó nếu đang giữ kết quả chính thức.

> **Lưu ý:** `--layers` không có tác dụng với MLP - `build_model()` không truyền tham số
> này cho `MLPForecast`, vốn có thân mạng cố định hai lớp ẩn. Giá trị `layers` trong
> `best_config.json` của MLP vì thế không mang ý nghĩa.

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

Thiếu gói `holidays` thì cột `is_holiday` bằng 0 và phần so sánh ngày lễ bị bỏ qua (có cảnh báo).
Trên Windows cần `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy, nếu không `error_analysis.py`
dừng với `UnicodeEncodeError` vì console mặc định dùng `cp1252`.

> **Cảnh báo:** `doc_du_bao_dl()` quét **mọi** thư mục con của `results/dl/` và đánh khóa theo
> cột `model`, nên thư mục nào xếp sau theo thứ tự chữ cái sẽ thắng. Không đặt bản sao lưu
> bên trong `results/dl/` (ví dụ `results/dl/mlp_backup/`) - mọi hình và bảng MLP sẽ lặng lẽ
> lấy số từ bản sao thay vì từ `results/dl/mlp/`. Để sao lưu ở ngoài, như `results/luu_tam/`.

Hai script đọc cấu hình đã dò từ `results/dl/<model>/tables/best_config.json` và chỉ giữ
những lần chạy khớp cấu hình đó. Nhờ vậy `results/experiments.csv` chứa nhiều cấu hình của
cùng một mô hình mà bảng tổng hợp vẫn không trộn chúng lại.
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

Mô hình học sâu báo cáo trung bình trên 5 seed (42-46); baseline tiền định nên chỉ chạy một lần.
Cấu hình đã dò: **LSTM** `lookback=24, hidden=128, layers=2, dropout=0.2` - **MLP** `lookback=168, hidden=128, dropout=0.2`.

**test MAE (kW)**

| Mô hình | MT_234 | MT_217 | MT_162 (chính) | MT_314 |
|---|---|---|---|---|
| **MLP** | **3,614** | **8,120** | **20,638** | **35,442** |
| LSTM | 4,649 | 9,265 | 26,632 | 41,067 |
| Seasonal Naive 24h | 6,402 | 13,271 | 96,487 | 63,266 |
| Seasonal Naive 168h | 7,006 | 17,363 | 34,226 | 81,320 |
| Last Value | 8,178 | 19,939 | 43,554 | 91,852 |
| Moving Average 3h | 12,834 | 35,108 | 58,217 | 150,938 |

**test MASE** (chuẩn hóa theo dự báo naive một bước trên train; < 1 là tốt hơn naive)

| Mô hình | MT_234 | MT_217 | MT_162 (chính) | MT_314 |
|---|---|---|---|---|
| **MLP** | **0,4122** | **0,4055** | **0,4372** | **0,4446** |
| LSTM | 0,5303 | 0,4626 | 0,5642 | 0,5152 |
| Seasonal Naive 24h | 0,7303 | 0,6627 | **2,0440** | 0,7937 |
| Seasonal Naive 168h | 0,7993 | 0,8670 | 0,7250 | 1,0202 |

Bảng đầy đủ kèm RMSE và sMAPE: `results/analysis/tables/bang_tong_hop.csv`.

**Nhận xét chính**

- Cả hai mô hình học sâu vượt mọi baseline trên cả bốn chuỗi.
- **MLP tốt hơn LSTM trên cả bốn chuỗi**, chênh lớn nhất ở chuỗi chính (20,64 so với 26,63 kW). Nguyên nhân **không** đơn thuần là độ dài cửa sổ: khi cấp cho LSTM đúng cửa sổ 168 giờ ở cùng điều kiện (`hidden=64`, seed 42), test MAE là **29,090** — tệ hơn bản 24 giờ (28,248) và vẫn kém MLP 41,0 %. Vậy vấn đề là LSTM *không khai thác được* cửa sổ dài, chứ không phải chưa được cấp cửa sổ dài. Với horizon một bước, ánh xạ trực tiếp từng lag phù hợp hơn cơ chế nén tuần tự qua hidden state — kiến trúc phức tạp hơn không tự động cho kết quả tốt hơn.
- Seasonal Naive 24h thất bại trên chuỗi chính (MASE 2,044 - tệ hơn dự báo naive), đúng như dự đoán từ tiêu chí chọn chuỗi `ac_lag24 = 0,569`. Điều này xác nhận chuỗi chính có dư địa thật cho mô hình học sâu.
- **Ma trận ablation 2×2** (80 lần chạy): đặc trưng lịch giúp LSTM 7,2 % trên chuỗi chính (4,95 lần độ lệch chuẩn giữa các seed) nhưng **không giúp MLP** (mọi chênh lệch dưới 0,5 lần độ lệch chuẩn). Lý do: cửa sổ 168 giờ của MLP đã chứa trọn một tuần nên thông tin "thứ mấy" là dư thừa, còn cửa sổ 24 giờ của LSTM thì không. Bảng: `results/analysis/tables/ablation_dac_trung_lich.csv`.
- **Dò siêu tham số** cải thiện MLP 1,85-7,55 % (có ý nghĩa trên 3/4 chuỗi) và giảm độ lệch chuẩn giữa các seed trên MT_162 từ 0,700 xuống 0,240.
- **Phân tích lỗi:** sai số tập trung ở các giờ đổi chế độ quanh ngày lễ. Trên MT_162, MAE trong 73 giờ lễ cao hơn ngày thường 94,3 %; và ưu thế của MLP so với LSTM biến mất hoàn toàn ở những giờ này (39,37 so với 37,52 kW).

Sổ cái đầy đủ 120 lần chạy ở giai đoạn chính thức: `results/experiments.csv`.
Báo cáo chi tiết phần MLP và phân tích lỗi được soạn thành tệp Word `docs/baocao_mlp_phan_tich_loi.docx`
(17 bảng, 11 hình). Tệp Word **không** được đưa lên Git - xem `.gitignore` - nên nhóm chia sẻ riêng
khi nộp. Mọi số liệu trong đó đều sinh ra từ các bảng trong `results/analysis/tables/` và
`results/experiments.csv`, nên có thể kiểm tra lại trực tiếp từ repo mà không cần tệp Word.

### Checkpoint mô hình đã huấn luyện

Checkpoint không nằm trong Git (`.gitignore` chặn `models/*`, chỉ chừa `scaler_*.pkl`),
nên được chia sẻ qua thư mục Drive của nhóm:

**https://drive.google.com/drive/folders/1ERuelBv0EZRyBdjFLVPWUQ_4Hr-0vD5a**

| Gói trong thư mục | Nội dung |
|---|---|
| `MLP_hidden128_checkpoints.zip` | 20 checkpoint MLP (4 chuỗi × 5 seed), cấu hình `hidden=128` - 5,7 MB |
| `MLP_MT_162_hidden128_5seed.zip` | 5 checkpoint chuỗi chính MT_162, kèm `MANIFEST`, script xác minh và hướng dẫn - 1,4 MB |

Giải nén các tệp `.pt` vào `models/dl/`. Quyền truy cập theo cài đặt chia sẻ của thư mục;
ai không mở được thì liên hệ thành viên phụ trách phần MLP.

Mỗi checkpoint **tự chứa scaler** (`scaler_mean`, `scaler_scale`) cùng toàn bộ siêu tham số,
nên nạp và dùng được ngay, không cần tệp scaler riêng.

**Phân biệt hai cấu hình MLP** - hai bên dùng cùng quy ước tên tệp `MLP_<chuỗi>_s<seed>.pt`
nên rất dễ lẫn:

| | Cấu hình đã dò (dùng trong báo cáo) | Cấu hình cũ, đặt tay |
|---|---|---|
| `hidden` | 128 | 64 |
| `body.0.weight` | `(256, 168)` | `(128, 168)` |
| Số tham số | 76.289 | 29.953 |

Xác minh checkpoint MT_162 bằng `python kiem_tra_checkpoint_mt162.py` (đặt 5 tệp `.pt` vào
`models/dl/` trước). Script nạp lại checkpoint và suy luận trên tập test, **không** huấn luyện
lại; kết quả đúng phải là MAE 20,6384 và RMSE 31,2840 trung bình 5 seed. Nếu nạp phải
checkpoint cũ, script dừng với thông báo rõ thay vì cho ra số sai.

## 7. Phần sử dụng mã nguồn và mô hình bên ngoài

Dữ liệu: UCI ElectricityLoadDiagrams20112014. Thư viện: pandas, numpy, matplotlib, seaborn, scikit-learn, PyYAML, **PyTorch** (framework học sâu đã chốt) và **holidays** (lịch nghỉ công Bồ Đào Nha cho cột `is_holiday`). Mọi kiến trúc mô hình do nhóm tự viết trong `src/models_dl.py`; không dùng mô hình pretrained. Cập nhật mục này nếu có thay đổi.

## 8. Phân công

| Thành viên | Phần việc chính |
|---|---|
| Minh Thư | Dữ liệu, EDA, tiền xử lý, chọn chuỗi |
| Đăng Long | Baseline, chia tập, bộ đánh giá chung |
| Hồng Sơn | Cửa sổ trượt, LSTM |
| Bảo Quốc | MLP, thử nghiệm cải tiến, phân tích lỗi |
| Gia Long | Demo, tích hợp, README, báo cáo, slide |
