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
| Giá trị 0 | **Không xóa, không thay thế.** Số 0 ở **đầu** chuỗi = khách hàng chưa tồn tại; số 0 **giữa** chuỗi = tiêu thụ thật bằng 0, là quan sát hợp lệ. Hai loại này được phân biệt bằng hai cột riêng trong `customer_quality_all.csv`. |
| Độ chính xác số | Dữ liệu thô được đọc ở **float64**. Không dùng float32 vì `const_run_max_h` so sánh bằng tuyệt đối, sai số float32 sinh ra đoạn không đổi giả. |
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
| `MT_234` | số thực | kW | Chuỗi phụ, mức tải nhỏ (88 kW) |
| `MT_217` | số thực | kW | Chuỗi phụ, mức tải trung bình (214 kW) |
| `MT_162` | số thực | kW | Chuỗi **chính** (289 kW), trung bình 4 mốc 15 phút |
| `MT_314` | số thực | kW | Chuỗi phụ, mức tải lớn (1.121 kW) |

Thứ tự cột trong tệp theo `customers.hourly_file_columns` trong `configs/config.yaml`; đổi danh sách thì chạy lại `data_preprocessing.py`.

Lưu ý: `MT_162` có **4 giờ bằng 0** (2012-03-25 02:00, 2012-10-28 02:00, 2013-05-06 10:00, 2013-10-07 10:00). Hai mốc đầu là ngày chuyển giờ; hai mốc sau là tiêu thụ thật bằng 0. Cả bốn đều trong tập train. Ba chuỗi còn lại không có giờ nào bằng 0.

### `data/processed/train.csv`, `validation.csv`, `test.csv`

Sinh bởi `src/data.py`. Cột **giống hệt** `series_hourly.csv`, chỉ khác khoảng thời gian:

| Tệp | Khoảng (bao gồm mốc cuối) | Số dòng |
|---|---|---|
| `train.csv` | `2011-01-01 01:00` → `2014-03-01 00:00` | 27.720 |
| `validation.csv` | `2014-03-01 01:00` → `2014-08-01 00:00` | 3.672 |
| `test.csv` | `2014-08-01 01:00` → `2015-01-01 00:00` | 3.672 |

### `data/processed/candidates_hourly.csv` — bộ nhớ đệm

Không nằm trong Git (~50 MB). `timestamp` + một cột kW mỗi chuỗi qua tầng A (**158 cột**).
Chỉ dùng cho `python src/select_customers.py --cached`; tái tạo được bằng một lệnh.

Giá trị trong tệp này **phải trùng khít** `series_hourly.csv` ở bốn cột dùng chung, vì cả hai dùng cùng quy ước gộp giờ (`closed="right", label="right"`). Nếu lệch, kiểm tra lại kiểu dữ liệu khi đọc file thô.

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
| `zeros_after_start` | số nguyên | Số giá trị 0 **nằm giữa** chuỗi — tiêu thụ thật bằng 0, ý nghĩa khác hẳn cột trên |
| `mean_15min` | số thực (kW) | Trung bình toàn chuỗi |
| `pass_no_missing` | luận lý | `missing_count == 0` |
| `pass_no_negative` | luận lý | `negative_count == 0` |
| `pass_full_from_2011` | luận lý | `leading_zero_steps <= max_leading_zero_steps` |
| `pass_no_inner_zero` | luận lý | `zeros_after_start <= max_zeros_after_start`. **Tiêu chí này đã được vô hiệu hóa có chủ đích:** `max_zeros_after_start` đặt rất lớn trong config nên cột luôn `True`. Giữ cột lại để người đọc thấy nhóm đã cân nhắc và quyết định không loại theo số 0 giữa chuỗi |
| `pass_positive_mean` | luận lý | `mean_15min > 0` |
| `stage_a_pass` | luận lý | Đạt **mọi** tiêu chí `pass_*` ở trên |
| `stage_a_reason` | chuỗi | Rỗng nếu đạt; nếu trượt thì liệt kê tên tiêu chí trượt, cách nhau bằng dấu phẩy |

Kết quả: **370 → 158** chuỗi. Cả 212 chuỗi bị loại đều chỉ vì một lý do: `full_from_2011`.

### `customer_features_hourly.csv` — tầng B, các chuỗi qua tầng A (158 dòng)

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

**Cột nào được dùng làm tiêu chí lọc:** chỉ `const_run_max_h`, `cv`, `ac_lag168`, `year_drift`, `shift_test_vs_train` (tầng C), cộng `ac_lag24` (tầng D1) và `mean_kW` (tầng D2). Các cột còn lại (`std_kW`, `max_kW`, `const_runs_ge_6h`, `ac_lag1`, `shift_val_vs_train`, `weekday_weekend_ratio`) **chỉ để tham khảo**, không tham gia vào việc chọn chuỗi.

### `customer_shortlist.csv` — tầng C, các chuỗi đạt (122 dòng)

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

Bảng đã sắp xếp giảm dần theo `ac_lag168`. Kết quả: **158 → 122**; 36 chuỗi bị loại, nhiều nhất là 19 chuỗi vì `no_flat`.

### `customer_main_candidates.csv` — tầng D1, ứng viên chuỗi chính

Các chuỗi có `ac_lag24 < main_max_ac_lag24` (= 0,90), sắp xếp giảm dần theo `ac_lag168`. **Dòng đầu là chuỗi chính được chọn.**

Cột: `customer`, `mean_kW`, `cv`, `ac_lag24`, `ac_lag168`, `year_drift`, `shift_test_vs_train` — ý nghĩa như bảng tầng B.

Chỉ **4 trên 122** chuỗi thỏa điều kiện: MT_162 (0,569 / 0,946), MT_124 (0,892 / 0,930), MT_158 (0,560 / 0,891), MT_180 (0,865 / 0,833). Chọn MT_162; dự phòng MT_124.

### `customer_selection.csv` — kết quả cuối, các chuỗi được dùng

| Cột | Kiểu | Mô tả |
|---|---|---|
| `customer` | chuỗi | Mã khách hàng |
| `role` | chuỗi | `CHÍNH` hoặc `phụ` |
| `mean_kW`, `cv`, `const_run_max_h`, `ac_lag24`, `ac_lag168`, `year_drift`, `shift_test_vs_train` | | Ý nghĩa như bảng tầng B |

Sắp xếp tăng dần theo `mean_kW`.

### `do_nhay_nguong.csv` — kiểm tra độ nhạy của ngưỡng

Sinh bởi `python kiem_tra_do_nhay.py` (chạy ở thư mục gốc, cần có `candidates_hourly.csv`).
Một dòng cho mỗi bộ ngưỡng thử; mỗi lần đổi **một** ngưỡng, giữ nguyên các ngưỡng còn lại.

| Cột | Kiểu | Mô tả |
|---|---|---|
| `ngưỡng` | chuỗi | Tên ngưỡng được đổi trong lần thử này |
| `giá trị` | số | Giá trị thử |
| `đang dùng` | chuỗi | `<--` nếu đây là giá trị đang dùng trong config |
| `qua tầng C` | số nguyên | Số chuỗi qua tầng C với bộ ngưỡng này |
| `ứng viên chính` | số nguyên | Số chuỗi thỏa `ac_lag24 < main_max_ac_lag24` |
| `chuỗi chính` | chuỗi | Chuỗi chính được chọn, hoặc `(rỗng)` nếu không chuỗi nào qua tầng C |
| `khớp gốc` | chuỗi | `có` nếu trùng chuỗi chính của bộ ngưỡng đang dùng, `KHÁC` nếu không |

Kết quả: **36/37** bộ ngưỡng chọn MT_162.

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
| `mean_MT_234`, `mean_MT_217`, `mean_MT_162`, `mean_MT_314` | số thực (kW) | Trung bình của từng chuỗi **trong tập đó** — dùng để thấy lệch phân phối giữa các tập |

Số cột `mean_*` thay đổi theo `customers.hourly_file_columns` trong config.

Giá trị của chuỗi chính MT_162: train 293,5 — validation 255,1 — test 285,0 kW. Validation thấp hơn train ~13%, test chỉ thấp hơn 3%: cần nêu trong phần hạn chế.

### `results/tables/dst_hours.csv`

Sinh bởi `src/dst_check.py`. Một dòng cho mỗi cặp (ngày chuyển giờ × chuỗi) — 8 ngày × 4 chuỗi = 32 dòng.

| Cột | Kiểu | Mô tả |
|---|---|---|
| `date` | ngày | Ngày chuyển giờ (Chủ nhật cuối tháng 3 hoặc tháng 10) |
| `kind` | chuỗi | `march` (thiếu một giờ) hoặc `october` (gộp hai giờ) |
| `hour_label` | datetime | Mốc giờ nghi ngờ, luôn là nhãn `02:00` của ngày đó |
| `split` | chuỗi | Mốc này rơi vào tập `train`, `validation` hay `test` |
| `customer` | chuỗi | Mã khách hàng |
| `value_01h`, `value_02h`, `value_03h` | số thực (kW) | Giá trị tại nhãn 01:00, 02:00 và 03:00 |
| `ratio_vs_neighbors` | số thực | `value_02h` chia trung bình hai giờ lân cận. Nhỏ hơn 1 nhiều = giờ bị thiếu; lớn hơn 1 nhiều = hai giờ bị gộp |
| `ref_low`, `ref_high` | số thực | Cận dưới và cận trên của khoảng bình thường, **lấy từ chính chuỗi đó**: phân vị `dst.ref_percentile` và `100 − ref_percentile` của phân bố `ratio_vs_neighbors` trên mọi ngày thường. In kèm để người đọc tự kiểm chứng |
| `n_ngay_tham_chieu` | số nguyên | Số ngày thường dùng để tính hai cận (≈1.453). Dưới 100 ngày thì không đánh dấu theo tỷ lệ vì phân vị không đáng tin |
| `suggest_check` | luận lý | **Gợi ý**, không phải kết luận. `True` khi `reason` không rỗng |
| `reason` | chuỗi | Lý do đánh dấu: `giá trị bằng 0`, `thấp bất thường so với chính chuỗi này`, `cao bất thường so với chính chuỗi này`, hoặc rỗng |

**Ngưỡng không đặt cứng.** Phiên bản trước dùng một cặp ngưỡng chung cho mọi chuỗi (`ratio < 0,5` hoặc `> 1,3`); hai con số đó không có cơ sở, và áp một ngưỡng chung là sai nguyên tắc — chuỗi biến động mạnh thì ngày thường cũng vượt ngưỡng (báo nhầm), chuỗi rất đều thì bất thường thật lại không vượt (bỏ sót). Phiên bản hiện tại lấy ngưỡng từ phân bố của chính chuỗi, nên câu trả lời cho "tại sao đánh dấu giờ này" là *vì nó bất thường so với chính chuỗi này*.

**Vì sao phân vị 1** (`dst.ref_percentile: 1.0` trong config): theo định nghĩa phân vị, 2p% số ngày thường cũng nằm ngoài khoảng — đó là tỷ lệ báo nhầm chấp nhận trước. Mỗi chuỗi có 8 ngày chuyển giờ, nên số lần báo nhầm kỳ vọng là `8 × 2p/100`: với p = 5 là 0,80 lần (gần 1, quá lỏng), với p = 1 là 0,16 lần (đủ chặt). Ngưỡng suy ra từ số ngày có trong dữ liệu, không chọn theo cảm tính.

Kết quả: cả bốn chuỗi đều có **6 trên 8** ngày bị đánh dấu. Riêng MT_162 (khoảng 0,96–1,50): hai ngày năm 2011 bình thường; 2012-03-25 và 2012-10-28 giá trị bằng 0; 2013-03-31 và 2014-03-30 thấp bất thường; 2013-10-27 và 2014-10-26 cao bất thường. Mốc tháng 3/2014 nằm trong **validation**, mốc tháng 10/2014 nằm trong **test**.

Một ngoại lệ đáng ghi: **2012-10-28** là ngày tháng 10 (đáng ra giá trị *tăng* vì gộp hai giờ) nhưng giá trị lại **bằng 0** — tài liệu và dữ liệu không khớp hoàn toàn ở mốc này.

### `results/tables/customer_comparison_hourly.csv`

Sinh bởi `notebooks/01_data_exploration.ipynb` (mục 3). Xuất hiện sau khi chạy notebook.
Cùng ý nghĩa với bảng tầng B nhưng đặt tên tiếng Việt, chỉ cho các chuỗi đang dùng:
`customer`, `mean_kW`, `cv(std/mean)`, `max_kW`, `giờ_không_đổi_dài_nhất`, `số_đoạn_không_đổi_>=6h`,
`autocorr_lag1`, `autocorr_lag24`, `autocorr_lag168`.

Cột `giờ_không_đổi_dài_nhất` dùng **cùng một hàm** `longest_constant_run` mà tầng C dùng để lọc (notebook import từ `src.select_customers`), nên nó phải trùng `const_run_max_h` trong `customer_selection.csv`: MT_234 = 2, MT_217 = 1, MT_162 = 1, MT_314 = 3. Lệch nhau là dấu hiệu hai pipeline đọc dữ liệu khác nhau.

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

Với chuỗi chính MT_162, `ac_lag168` (0,946) cao hơn hẳn `ac_lag24` (0,569), nên Seasonal Naive 168 được kỳ vọng tốt hơn Seasonal Naive 24. `ac_lag1` = 0,848 nên Last Value cũng là đối thủ mạnh.

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

Với MT_162, hai mốc giá trị đổi nhanh nhất trong ngày là **nhãn 03** (424 → 232 kW) và **nhãn 09** (211 → 387 kW); dự đoán sai số sẽ lớn nhất ở hai giờ này. Đây là **dự đoán cần kiểm chứng** bằng chính bảng này, không phải kết luận.

### `results/tables/baseline_test_residuals.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `timestamp` | datetime | Mốc được dự báo trong tập test |
| `y_true` | số thực (kW) | Giá trị thực tế |
| `y_pred` | số thực (kW) | Dự báo của baseline tốt nhất theo validation |
| `residual` | số thực (kW) | `y_true − y_pred`, giữ dấu |

Khác `top10_worst_errors.csv` ở chỗ giữ **toàn bộ** 3.672 mốc, không chỉ 10 mốc tệ nhất — dùng để vẽ phân phối sai số và kiểm tra sai số có lệch về một phía không.

### `results/tables/top10_worst_errors.csv`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `timestamp` | datetime | Mốc có sai số lớn |
| `y_true` | số thực (kW) | Giá trị thực tế |
| `y_pred` | số thực (kW) | Dự báo của mô hình tốt nhất theo validation |
| `residual` | số thực (kW) | `y_true − y_pred`, giữ dấu để biết dự báo thừa hay thiếu |
| `abs_error` | số thực (kW) | Trị tuyệt đối của `residual`, dùng để sắp xếp |

> **Lưu ý:** tệp này hiện trên đĩa có cấu trúc khác với cấu trúc mà notebook 02 sinh ra. Chạy lại notebook để hai bên khớp nhau.

## N. Phần học sâu (`results/dl/`, `results/analysis_train/`, `models/dl/`)

Sinh bởi `src/train_dl.py`, `src/training_figures.py`, `src/error_analysis.py`.
Đơn vị của mọi cột sai số là **kW** (trừ các cột ghi rõ `_scaled` hoặc `_%`).

### `results/experiments.csv` — nhật ký mọi lần chạy

Mỗi dòng là một lần huấn luyện (một mô hình × một chuỗi × một seed × một cấu hình).
Ghi nối thêm sau mỗi lần chạy, không ghi đè.

| Cột | Kiểu | Mô tả |
|---|---|---|
| `exp_id` | chuỗi | Mã định danh duy nhất, ghép từ mô hình + chuỗi + tham số + seed |
| `stage` | chuỗi | `baseline` (chạy mốc), `tune` (dò tham số), `final` (chính thức), `single` |
| `model` | chuỗi | `LSTM`, `MLP` hoặc `CNNLSTM` |
| `customer` | chuỗi | Mã chuỗi khách hàng |
| `lookback` | số nguyên | Số giờ lịch sử làm đầu vào |
| `hidden` | số nguyên | Kích thước hidden state (LSTM) hoặc lớp ẩn (MLP) |
| `layers` | số nguyên | Số lớp |
| `dropout` | số thực | Tỉ lệ dropout |
| `lr` | số thực | Learning rate ban đầu |
| `batch_size` | số nguyên | Kích thước lô |
| `calendar` | 0/1 | Có dùng đặc trưng lịch hay không |
| `holiday` | 0/1 | Trong đặc trưng lịch có cột `is_holiday` hay không |
| `seed` | số nguyên | Seed ngẫu nhiên |
| `n_params` | số nguyên | Số tham số huấn luyện được |
| `best_epoch` | số nguyên | Epoch có validation loss thấp nhất; trọng số lấy tại đây |
| `epochs_run` | số nguyên | Tổng số epoch đã chạy trước khi early stopping dừng |
| `val_MAE`, `val_RMSE`, `val_MASE` | số thực | Chỉ số trên tập validation, **thang gốc** (đã `inverse_transform`) |
| `test_MAE`, `test_RMSE`, `test_MASE` | số thực | Chỉ số trên tập test, thang gốc |

Lưu ý: `epochs_run − best_epoch` luôn bằng `patience` (mặc định 30) khi early
stopping kích hoạt. Dùng để kiểm tra mô hình đã được chạy đủ hay chưa.

### `results/dl/<model>/tables/history_<chuoi>_s<seed>.csv` — learning curve

| Cột | Kiểu | Mô tả |
|---|---|---|
| `epoch` | số nguyên | Số thứ tự epoch, bắt đầu từ 1 |
| `train_mse_scaled` | số thực | **Train loss**: MSE trung bình trên tập train, **thang đã chuẩn hoá** |
| `validation_mse_scaled` | số thực | **Validation loss**: MSE trên toàn tập validation, thang đã chuẩn hoá |
| `lr` | số thực | Learning rate tại epoch đó (giảm dần do `ReduceLROnPlateau`) |

Loss tính trên thang chuẩn hoá nên **không so sánh trực tiếp được giữa các chuỗi**.
Mọi chỉ số báo cáo (MAE/RMSE/MASE) đều được tính lại trên thang gốc.

### `results/dl/<model>/predictions/<tap>_<chuoi>_s<seed>.csv` — dự báo từng điểm

`<tap>` là `validation` hoặc `test`. Mỗi tệp có đúng 3.672 dòng.

| Cột | Kiểu | Đơn vị | Mô tả |
|---|---|---|---|
| `timestamp` | datetime | — | Mốc kết thúc của giờ cần dự báo |
| `customer` | chuỗi | — | Mã chuỗi |
| `model` | chuỗi | — | Tên mô hình |
| `seed` | số nguyên | — | Seed của lần chạy |
| `y_true` | số thực | kW | Giá trị thực tế |
| `y_pred` | số thực | kW | Giá trị dự báo, đã `inverse_transform` về thang gốc |
| `error` | số thực | kW | `y_true − y_pred`. Dương = mô hình dự báo THẤP hơn thực tế |

### `results/dl/<model>/tables/final_summary.csv`

Tổng hợp của lần chạy `--stage final` gần nhất: trung bình và độ lệch chuẩn
qua các seed cho từng chuỗi (`*_mean`, `*_std`), kèm `best_epoch_mean` và `n_seeds`.

### `results/dl/<model>/tables/tuning_log.csv`

Mỗi dòng là một cấu hình đã thử trong giai đoạn dò tham số (chỉ trên chuỗi chính).
Cột: `lookback`, `hidden`, `lr`, `layers`, `dropout`, `val_MAE`, `val_MASE`, `best_epoch`.
Cấu hình được chọn ghi ở `best_config.json` cùng thư mục.

### `results/analysis_train/<model>/tables/`

| Tệp | Nội dung |
|---|---|
| `muc_anh_huong_tham_so.csv` | Mỗi tham số ảnh hưởng bao nhiêu % tới validation MAE, **so sánh có kiểm soát** (chỉ so các cấu hình khác nhau đúng ở tham số đó) |
| `hoi_tu_va_on_dinh.csv` | `best_epoch`, `epochs_run`, test MAE ± std và `std_tren_TB_%` cho từng chuỗi |
| `truoc_sau_do_tham_so.csv` | MAE lần chạy mốc so với sau khi dò tham số |

### `models/dl/<model>_<chuoi>_s<seed>.pt` — checkpoint

**Không nằm trong Git** (`.gitignore` chặn `models/*`); xem link Drive ở README mục 6.
Mỗi tệp là một `dict` PyTorch chứa `model_state_dict`, toàn bộ siêu tham số
(`lookback`, `hidden`, `layers`, `dropout`, `lr`, `calendar`, `holiday`),
`best_epoch`, `n_params`, `seed`, cùng `scaler_mean` và `scaler_scale` để
`inverse_transform` đúng về thang gốc khi nạp lại.
```

---

## 5. Biểu đồ (`results/figures/`)

Các tệp `eda_*` do `notebooks/01_data_exploration.ipynb` sinh ra, các tệp `baseline_*` do `notebooks/02_baseline.ipynb`.

| Tệp | Nội dung |
|---|---|
| `eda_main_full_series.png` | Toàn kỳ chuỗi chính, có vạch mốc chia tập |
| `eda_main_distribution.png` | Biểu đồ tần suất và biểu đồ hộp theo thứ trong tuần |
| `eda_main_week_and_weekday.png` | Một tuần mẫu và trung bình theo thứ |
| `eda_main_hour_profile.png` | Trung bình theo nhãn giờ, tách ngày thường và cuối tuần |
| `eda_main_acf.png` | Tự tương quan theo độ trễ 1–192 giờ |
| `eda_series_comparison.png` | Bốn chuỗi chuẩn hóa z-score trên cùng một tuần |
| `eda_rolling_mean_all.png` | Trung bình trượt 30 ngày của bốn chuỗi |
| `baseline_acf.png` | Tự tương quan dùng để chọn độ trễ cho Seasonal Naive |
| `baseline_val_comparison.png` | So sánh MAE và RMSE của bốn baseline trên validation |
| `baseline_val_mae.png` | MAE của bốn baseline trên validation |
| `baseline_val_rmse.png` | RMSE của bốn baseline trên validation |
| `baseline_test_timeseries.png` | Thực tế và dự báo trên một tuần đầu tập test |
| `baseline_residuals_timeseries.png` | Sai số theo thời gian trên tập test |
| `baseline_error_distribution.png` | Phân phối sai số |
| `baseline_error_by_hour.png` | MAE theo giờ trong ngày |
| `baseline_error_by_dayofweek.png` | MAE theo thứ trong tuần |

> Ba hình `baseline_val_comparison`, `baseline_val_mae` và `baseline_val_rmse` trình bày cùng một thông tin ở ba dạng. Danh sách hình của notebook 02 còn đang được Nhiệm vụ 2 chỉnh lại, nên mục này cập nhật sau khi notebook 02 chốt.

Tuần mẫu dùng ở `eda_main_week_and_weekday.png` và `eda_series_comparison.png` được chọn **bằng quy tắc**, không chọn tùy ý: tuần đủ 168 giờ, **nằm trong tập train**, không chứa ngày chuyển giờ, không chứa giờ bằng 0, và có trung bình gần trung bình toàn chuỗi nhất. Kết quả: `2012-11-12 → 2012-11-18` (288,3 kW so với toàn chuỗi 288,6 kW). Điều kiện "nằm trong tập train" là bắt buộc — khảo sát dữ liệu không được nhìn vào validation hay test, kể cả chỉ để vẽ minh họa.
