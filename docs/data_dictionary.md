# Từ điển dữ liệu

Mô tả cột, đơn vị và nguồn sinh ra của mọi tệp dữ liệu và bảng kết quả trong kho mã nguồn.
Phục vụ yêu cầu "tài liệu mô tả cột, đơn vị, cách xử lý thiếu, thời gian bắt đầu/kết thúc" của Nhiệm vụ 1.

## Quy ước chung

| Khái niệm | Quy ước |
|---|---|
| Đơn vị đo | **kW** (công suất). Dữ liệu gốc là kW mỗi 15 phút; dữ liệu theo giờ là kW trung bình mỗi giờ. Muốn đổi sang kWh thì chia 4. |
| Cột `timestamp` | Kiểu `datetime`, không mang múi giờ. Là **mốc kết thúc** của khoảng đo: nhãn `02:00` gồm bốn mốc `01:15`, `01:30`, `01:45`, `02:00`. |
| Múi giờ | Giờ địa phương của nguồn dữ liệu (Bồ Đào Nha), **không** phải UTC. Đây là lý do có các giờ bất thường vào ngày chuyển giờ — xem `dst_hours.csv`. |
| Khoảng thời gian | Dữ liệu theo giờ: `2011-01-01 01:00` đến `2015-01-01 00:00`, đúng 35.064 giờ, cách đều 1 giờ, không trùng, không thiếu. |
| Mã khách hàng | `MT_001` đến `MT_370`, ẩn danh. Không có thông tin định danh. |
| Dữ liệu thiếu | **Không điền, không xóa.** Mọi hàm chỉ kiểm tra và báo lỗi. Trong các tệp dưới đây, mọi cột số đều không có giá trị thiếu. |
| Mã hóa tệp | CSV, UTF-8 có BOM (`utf-8-sig`), dấu phẩy phân cách. Riêng dữ liệu **gốc** của UCI dùng dấu chấm phẩy và dấu phẩy thập phân. |
| Hậu tố `_15min` / `_hourly` / `_kW` | Chỉ rõ chỉ số được tính trên dữ liệu 15 phút hay dữ liệu theo giờ. |

---

## 1. Dữ liệu (`data/`)

### `data/raw/LD2011_2014.txt` — dữ liệu gốc

Không nằm trong Git (~700 MB). Tải bằng `python src/download_data.py`.

| Cột | Kiểu | Mô tả |
|---|---|---|
| (cột đầu, không tên) | datetime | Mốc thời gian 15 phút, mốc kết thúc khoảng đo |
| `MT_001` … `MT_370` | số thực | Công suất tiêu thụ, đơn vị kW, của 370 khách hàng |

Dấu phân cách `;`, dấu thập phân `,`. 140.256 dòng.

### `data/processed/series_hourly.csv` — dữ liệu dùng chung của nhóm

Sinh bởi `src/data_preprocessing.py`. 35.064 dòng.

| Cột | Kiểu | Đơn vị | Mô tả |
|---|---|---|---|
| `timestamp` | datetime | — | Mốc kết thúc của giờ |
| `MT_158` | số thực | kW | Chuỗi **chính**, trung bình 4 mốc 15 phút |
| `MT_168` | số thực | kW | Chuỗi phụ, mức tải trung bình |
| `MT_169` | số thực | kW | Chuỗi phụ, mức tải nhỏ |
| `MT_294` | số thực | kW | Chuỗi phụ, mức tải lớn |

Danh sách cột lấy từ `customers.hourly_file_columns` trong `configs/config.yaml`; đổi danh sách thì chạy lại `data_preprocessing.py`.

### `data/processed/train.csv`, `validation.csv`, `test.csv`

Sinh bởi `src/data.py`. Cột **giống hệt** `series_hourly.csv`, chỉ khác khoảng thời gian:

| Tệp | Khoảng (bao gồm mốc cuối) | Số dòng |
|---|---|---|
| `train.csv` | `2011-01-01 01:00` → `2014-03-01 00:00` | 27.720 |
| `validation.csv` | `2014-03-01 01:00` → `2014-08-01 00:00` | 3.672 |
| `test.csv` | `2014-08-01 01:00` → `2015-01-01 00:00` | 3.672 |

### `data/processed/candidates_hourly.csv` — bộ nhớ đệm

Không nằm trong Git (~30 MB). `timestamp` + một cột kW mỗi chuỗi qua tầng A (90 cột).
Chỉ dùng cho `python src/select_customers.py --cached`; tái tạo được bằng một lệnh.

---

## 2. Bảng chọn chuỗi (`results/tables/`)

Năm bảng dưới đây do `src/select_customers.py` sinh ra. Ngưỡng ở mục `selection` trong `configs/config.yaml`.

### `customer_quality_all.csv` — tầng A, cả 370 chuỗi

| Cột | Kiểu | Mô tả |
|---|---|---|
| `customer` | chuỗi | Mã khách hàng |
| `records_15min` | số nguyên | Số mốc 15 phút (luôn 140.256) |
| `missing_count` | số nguyên | Số giá trị thiếu |
| `negative_count` | số nguyên | Số giá trị âm |
| `leading_zero_steps` | số nguyên | Số mốc bằng 0 **liên tiếp ở đầu** chuỗi — giai đoạn khách hàng chưa có dữ liệu |
| `leading_zero_days` | số thực | `leading_zero_steps / 96`, quy đổi ra ngày cho dễ đọc |
| `zeros_after_start` | số nguyên | Số giá trị 0 **nằm giữa** chuỗi — ý nghĩa khác hẳn cột trên |
| `mean_15min` | số thực (kW) | Trung bình toàn chuỗi |
| `pass_no_missing` | luận lý | `missing_count == 0` |
| `pass_no_negative` | luận lý | `negative_count == 0` |
| `pass_full_from_2011` | luận lý | `leading_zero_steps <= max_leading_zero_steps` |
| `pass_no_inner_zero` | luận lý | `zeros_after_start <= max_zeros_after_start` |
| `pass_positive_mean` | luận lý | `mean_15min > 0` |
| `stage_a_pass` | luận lý | Đạt **mọi** tiêu chí `pass_*` ở trên |
| `stage_a_reason` | chuỗi | Rỗng nếu đạt; nếu trượt thì liệt kê tên tiêu chí trượt, cách nhau bằng dấu phẩy |

### `customer_features_hourly.csv` — tầng B, các chuỗi qua tầng A

| Cột | Kiểu | Mô tả |
|---|---|---|
| `customer` | chuỗi | Mã khách hàng |
| `mean_kW`, `std_kW`, `max_kW` | số thực (kW) | Thống kê trên chuỗi theo giờ |
| `cv` | số thực | Hệ số biến thiên `std_kW / mean_kW`. Thấp = chuỗi phẳng, cao = chuỗi nhiễu |
| `const_run_max_h` | số nguyên (giờ) | Đoạn giá trị **không đổi** dài nhất. Dài = dấu hiệu đồng hồ hỏng hoặc giá trị bị giữ nguyên |
| `const_runs_ge_6h` | số nguyên | Số đoạn không đổi từ 6 giờ trở lên |
| `ac_lag1` | số thực −1…1 | Tự tương quan độ trễ 1 giờ. Cao = baseline Last Value mạnh |
| `ac_lag24` | số thực −1…1 | Tự tương quan độ trễ 24 giờ (chu kỳ ngày). Cao = Seasonal Naive 24h mạnh |
| `ac_lag168` | số thực −1…1 | Tự tương quan độ trễ 168 giờ (chu kỳ tuần) |
| `year_drift` | số thực ≥ 0 | `(trung bình năm cao nhất − thấp nhất) / trung bình toàn chuỗi`. Cao = đổi chế độ vận hành giữa các năm |
| `shift_val_vs_train` | số thực ≥ 0 | `|trung bình validation − trung bình train| / trung bình train` |
| `shift_test_vs_train` | số thực ≥ 0 | Như trên, cho tập test. Cao = lệch phân phối giữa huấn luyện và kiểm thử |
| `weekday_weekend_ratio` | số thực | Trung bình ngày thường chia trung bình cuối tuần. Cao = chu kỳ tuần rõ |

### `customer_shortlist.csv` — tầng C, các chuỗi đạt

Mọi cột của `customer_features_hourly.csv`, **cộng thêm**:

| Cột | Kiểu | Mô tả |
|---|---|---|
| `pass_no_flat` | luận lý | `const_run_max_h <= max_const_run_h` |
| `pass_cv` | luận lý | `cv` nằm trong `[cv_min, cv_max]` |
| `pass_weekly_cycle` | luận lý | `ac_lag168 >= min_ac_lag168` |
| `pass_stable_level` | luận lý | `year_drift <= max_year_drift` |
| `pass_small_shift` | luận lý | `shift_test_vs_train <= max_split_shift` |
| `stage_c_pass` | luận lý | Đạt mọi tiêu chí `pass_*` của tầng C |
| `stage_c_reason` | chuỗi | Tên các tiêu chí trượt, rỗng nếu đạt |

Bảng đã sắp xếp giảm dần theo `ac_lag168`.

### `customer_main_candidates.csv` — tầng D1, ứng viên chuỗi chính

Các chuỗi có `ac_lag24 < main_max_ac_lag24`, sắp xếp giảm dần theo `ac_lag168`. **Dòng đầu là chuỗi chính được chọn.**

Cột: `customer`, `mean_kW`, `cv`, `ac_lag24`, `ac_lag168`, `year_drift`, `shift_test_vs_train` — ý nghĩa như bảng tầng B.

### `customer_selection.csv` — kết quả cuối, các chuỗi được dùng

| Cột | Kiểu | Mô tả |
|---|---|---|
| `customer` | chuỗi | Mã khách hàng |
| `role` | chuỗi | `CHÍNH` hoặc `phụ` |
| `mean_kW`, `cv`, `const_run_max_h`, `ac_lag24`, `ac_lag168`, `year_drift`, `shift_test_vs_train` | | Ý nghĩa như bảng tầng B |

Sắp xếp tăng dần theo `mean_kW`.

---

## 3. Bảng tiền xử lý và chia tập

### `results/tables/selected_customers_quality.csv`

Sinh bởi `src/data_preprocessing.py`, trên dữ liệu **15 phút** của các chuỗi đã chọn.

| Cột | Kiểu | Mô tả |
|---|---|---|
| `customer` | chuỗi | Mã khách hàng |
| `records_15min` | số nguyên | Số bản ghi 15 phút (140.256) |
| `missing_count` | số nguyên | Số giá trị thiếu — phải bằng 0, khác 0 là script dừng |
| `zero_count` | số nguyên | Số giá trị bằng 0 |
| `negative_count` | số nguyên | Số giá trị âm — phải bằng 0, khác 0 là script dừng |
| `mean_15min`, `std_15min`, `min_15min`, `max_15min` | số thực (kW) | Thống kê mô tả trên dữ liệu 15 phút |

### `results/tables/hourly_preprocessing_report.csv`

Sinh bởi `src/data_preprocessing.py`, trên dữ liệu **theo giờ**.

| Cột | Kiểu | Mô tả |
|---|---|---|
| `customer` | chuỗi | Mã khách hàng |
| `records_hourly` | số nguyên | Số giờ sau khi gộp (35.064) |
| `missing_hourly` | số nguyên | Số giờ thiếu giá trị |
| `zero_hourly` | số nguyên | Số giờ có giá trị bằng 0 |
| `mean_hourly`, `std_hourly`, `min_hourly`, `max_hourly` | số thực (kW) | Thống kê mô tả theo giờ |
| `hours_with_not_4_records` | số nguyên | Số giờ **không** đủ 4 bản ghi 15 phút. Bằng 0 nghĩa là phép gộp không mất dữ liệu |

> Kiểm chứng đáng chú ý: `mean_hourly` trùng khít `mean_15min` ở mọi chuỗi khi cột cuối bằng 0.

### `results/tables/split_summary.csv`

Sinh bởi `src/data.py`. Ba dòng, mỗi dòng một tập.

| Cột | Kiểu | Mô tả |
|---|---|---|
| `split` | chuỗi | `train`, `validation` hoặc `test` |
| `rows` | số nguyên | Số giờ trong tập |
| `percent` | số thực | Tỷ lệ phần trăm trên tổng số giờ |
| `start`, `end` | datetime | Mốc đầu và mốc cuối của tập (bao gồm) |
| `mean_MT_158`, `mean_MT_168`, `mean_MT_169`, `mean_MT_294` | số thực (kW) | Trung bình của từng chuỗi **trong tập đó** — dùng để thấy lệch phân phối giữa các tập |

Số cột `mean_*` thay đổi theo `customers.hourly_file_columns` trong config.

### `results/tables/dst_hours.csv`

Sinh bởi `src/dst_check.py`. Một dòng cho mỗi cặp (ngày chuyển giờ × chuỗi).

| Cột | Kiểu | Mô tả |
|---|---|---|
| `date` | ngày | Ngày chuyển giờ (Chủ nhật cuối tháng 3 hoặc tháng 10) |
| `kind` | chuỗi | `march` (thiếu một giờ) hoặc `october` (gộp hai giờ) |
| `hour_label` | datetime | Mốc giờ nghi ngờ, luôn là nhãn `02:00` của ngày đó |
| `split` | chuỗi | Mốc này rơi vào tập `train`, `validation` hay `test` |
| `customer` | chuỗi | Mã khách hàng |
| `value_01h`, `value_02h`, `value_03h` | số thực (kW) | Giá trị tại nhãn 01:00, 02:00 và 03:00 |
| `ratio_vs_neighbors` | số thực | `value_02h` chia trung bình hai giờ lân cận. Nhỏ hơn 1 nhiều = giờ bị thiếu; lớn hơn 1 nhiều = hai giờ bị gộp |
| `suggest_check` | luận lý | **Gợi ý**, không phải kết luận. Với chuỗi biến động mạnh, ngày thường cũng có thể đạt ngưỡng |

### `results/tables/customer_comparison_hourly.csv`

Sinh bởi `notebooks/01_data_exploration.ipynb` (mục 3). Xuất hiện sau khi chạy notebook.
Cùng ý nghĩa với bảng tầng B nhưng đặt tên tiếng Việt, chỉ cho các chuỗi đang dùng:
`customer`, `mean_kW`, `cv(std/mean)`, `max_kW`, `giờ_không_đổi_dài_nhất`, `số_đoạn_không_đổi_>=6h`,
`autocorr_lag1`, `autocorr_lag24`, `autocorr_lag168`.

---

## 4. Bảng baseline (Nhiệm vụ 2)

Sinh bởi `notebooks/02_baseline.ipynb`, liệt kê ở đây để tài liệu trọn vẹn.

### Bốn phương pháp baseline

| Tên cột | Phương pháp | Công thức dự báo cho giờ `t` |
|---|---|---|
| `y_last_value` | Last Value | `x(t−1)` |
| `y_ma3` | Moving Average 3 giờ | trung bình của `x(t−3)`, `x(t−2)`, `x(t−1)` |
| `y_snaive24` | Seasonal Naive 24 giờ | `x(t−24)` — cùng giờ hôm qua |
| `y_snaive168` | Seasonal Naive 168 giờ | `x(t−168)` — cùng giờ tuần trước |

### `results/baseline_predictions_validation.csv`, `baseline_predictions_test.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `timestamp` | datetime | Mốc được dự báo (nhãn mục tiêu) |
| `y_true` | số thực (kW) | Giá trị thực tế tại mốc đó |
| `y_last_value`, `y_ma3`, `y_snaive24`, `y_snaive168` | số thực (kW) | Dự báo của bốn baseline, **thang đo gốc** |

### `results/baseline_predictions.csv`

Như trên, gộp validation và test, thêm cột `split` (`validation` hoặc `test`) ở đầu.

### `results/eval_timestamps.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `split` | chuỗi | `validation` hoặc `test` |
| `timestamp` | datetime | Mốc đánh giá. **Mọi mô hình phải chấm trên đúng tập mốc này** để so sánh công bằng |

### `results/tables/baseline_results.csv`, `baseline_test_results.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `model` | chuỗi | Tên cột dự báo tương ứng |
| `MAE` | số thực (kW) | Sai số tuyệt đối trung bình |
| `RMSE` | số thực (kW) | Căn bậc hai sai số bình phương trung bình, nhạy hơn với lỗi lớn |
| `MASE` | số thực | `MAE` chia `d_train`, trong đó `d_train` là MAE của dự báo naive-1 **trên tập train**. Nhỏ hơn 1 nghĩa là tốt hơn mức naive trung bình của giai đoạn huấn luyện |
| `N_samples` | số nguyên | Số mốc dùng để tính (phải bằng 3.672) |

### `results/tables/error_by_hour.csv`, `error_by_dayofweek.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `hour` / `dayofweek` | số nguyên | Giờ trong ngày 0–23 (nhãn mốc kết thúc) / thứ trong tuần, **0 = thứ Hai, 6 = Chủ nhật** |
| `MAE`, `RMSE` | số thực (kW) | Sai số trong nhóm đó |
| `N_samples` | số nguyên | Số mốc trong nhóm. Với `dayofweek`, số mốc **không đều** giữa các thứ vì hai ngày biên của tập test bị cắt |

### `results/tables/top10_worst_errors.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `timestamp` | datetime | Mốc có sai số lớn |
| `y_true` | số thực (kW) | Giá trị thực tế |
| `y_pred` | số thực (kW) | Dự báo của mô hình tốt nhất theo validation |
| `residual` | số thực (kW) | `y_true − y_pred`, giữ dấu để biết dự báo thừa hay thiếu |
| `abs_error` | số thực (kW) | Trị tuyệt đối của `residual`, dùng để sắp xếp |

> **Lưu ý:** tệp này hiện trên đĩa có cấu trúc khác với cấu trúc mà notebook 02 sinh ra. Chạy lại notebook để hai bên khớp nhau.

---

## 5. Biểu đồ (`results/figures/`)

Tất cả do `notebooks/01_data_exploration.ipynb` sinh ra, trừ ba tệp `baseline_*` của notebook 02.

| Tệp | Nội dung |
|---|---|
| `eda_main_full_series.png` | Toàn kỳ chuỗi chính, có vạch mốc chia tập |
| `eda_main_distribution.png` | Biểu đồ tần suất và biểu đồ hộp theo thứ trong tuần |
| `eda_main_week_and_weekday.png` | Một tuần mẫu và trung bình theo thứ |
| `eda_main_hour_profile.png` | Trung bình theo nhãn giờ, tách ngày thường và cuối tuần |
| `eda_main_acf.png` | Tự tương quan theo độ trễ 1–192 giờ |
| `eda_series_comparison.png` | Bốn chuỗi chuẩn hóa z-score trên cùng một tuần |
| `eda_rolling_mean_all.png` | Trung bình trượt 30 ngày của bốn chuỗi |
| `baseline_val_comparison.png` | So sánh MAE và RMSE của bốn baseline trên validation |
| `baseline_test_timeseries.png` | Thực tế và dự báo trên một tuần đầu tập test |
| `baseline_error_by_hour.png` | MAE theo giờ trong ngày |
