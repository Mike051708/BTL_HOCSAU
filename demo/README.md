# Hướng dẫn chạy demo dự báo điện năng

Demo dự báo công suất (kW) của **giờ kế tiếp** cho khách hàng **MT_162**. Có hai cách dùng, cùng một lõi dự báo nên cho kết quả giống nhau:

- **Trang web cục bộ** (`app.py` + `index.html`): chọn mô hình, dán hoặc tải dữ liệu, xem số dự báo, biểu đồ và bảng đánh giá.
- **Dòng lệnh** (`predict.py`): nạp checkpoint `.pt`, in dự báo; có chế độ chấm cả tập test.

Checkpoint đã chứa kiến trúc, siêu tham số, `scaler_mean` và `scaler_scale`, nên kết quả được đưa thẳng về kW mà không cần dữ liệu huấn luyện.

| Tệp | Vai trò |
|---|---|
| `predict.py` | Lõi dự báo và công cụ dòng lệnh |
| `app.py`, `index.html` | Máy chủ web và giao diện (chỉ dùng thư viện chuẩn của Python) |
| `checkpoints/` | Ba checkpoint đi kèm (mục 5) |
| `sample_input.csv` | Dữ liệu mẫu: 168 giờ cuối tập validation |
| `../tests/test_demo_predict.py` | 4 ca kiểm thử |

Mọi lệnh dưới đây chạy trong **PowerShell** tại **thư mục gốc dự án** (thư mục chứa `README.md`, `src/`, `demo/`).

---

## 1. Chuẩn bị (làm một lần)

**Bước 1 — Mở PowerShell tại thư mục gốc dự án**

```powershell
cd <đường dẫn tới thư mục dự án>
```

**Bước 2 — Bật môi trường Python**

```powershell
python -m venv .venv            # chỉ cần nếu chưa có .venv
.venv\Scripts\Activate.ps1
```

Nếu PowerShell không cho chạy script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Bước 3 — Cài thư viện**

```powershell
pip install -r requirements.txt
```

Cần Python 3.11 hoặc 3.12 (PyTorch chưa hỗ trợ 3.14). Gói `holidays` bắt buộc nếu dùng LSTM vì mô hình này dùng cột ngày lễ.

**Bước 4 — Kiểm tra checkpoint**

```powershell
dir demo\checkpoints
```

Phải thấy `MLP_MT_162_hidden128_khong_lich.pt`, `MLP_MT_162_khong_lich.pt` và `LSTM_MT_162_co_lich.pt`. Thiếu thì máy chủ web không khởi động được.

---

## 2. Cách A — Chạy trang web

**Bước 1 — Khởi động máy chủ**

```powershell
python demo/app.py
```

Trình duyệt tự mở `http://127.0.0.1:8000`; nếu không, gõ địa chỉ đó vào trình duyệt. Giữ cửa sổ PowerShell mở trong lúc dùng.

**Bước 2 — Chọn mô hình** ở mục *Mô hình* (mặc định MLP 168 giờ, không đặc trưng lịch).

**Bước 3 — Chọn dữ liệu** ở mục *Dữ liệu đầu vào*:

- **Dữ liệu mẫu:** dùng 168 giờ cuối tập validation, dự báo giờ 01:00 ngày 01/08/2014 và so với giá trị thực (451,48 kW).
- **Dán / tải CSV:** dán vào ô văn bản hoặc chọn tệp `.csv` (định dạng ở mục 4).

**Bước 4 — Bấm "Dự báo giờ kế tiếp".** Hiện số dự báo (kW), giờ được dự báo, biểu đồ 168 giờ lịch sử và các dòng chi tiết; với dữ liệu mẫu còn có giá trị thực và sai lệch.

**Bước 5 (tùy chọn) — Bấm "Chấm cả tập test (3.672 giờ)"** để xem MAE, RMSE, MASE của mô hình đang chọn so với hai baseline Seasonal Naive.

**Bước 6 — Dừng máy chủ:** quay lại cửa sổ PowerShell, bấm `Ctrl + C`.

| Mở địa chỉ | Kết quả |
|---|---|
| `http://127.0.0.1:8000/#run` | Tự dự báo, hiện sẵn kết quả (tiện khi trình diễn) |
| `http://127.0.0.1:8000/#eval` | Tự chấm tập test, hiện sẵn bảng đánh giá |

Đổi cổng nếu 8000 bị chiếm: `python demo/app.py --port 8080`. Thêm `--no-browser` để không tự mở trình duyệt.

Máy chủ chỉ lắng nghe `127.0.0.1` (máy đang chạy), chỉ nạp checkpoint trong `demo/checkpoints/` và giới hạn dữ liệu gửi lên 2 MB.

---

## 3. Cách B — Chạy bằng dòng lệnh

| Mục đích | Lệnh |
|---|---|
| Dự báo nhanh (MLP hidden 128, dữ liệu validation) | `python demo/predict.py` |
| Dự báo từ tệp CSV | `python demo/predict.py --csv demo/sample_input.csv` |
| Dự báo từ danh sách số kW | `python demo/predict.py --values 300 310 320 ...` (đủ 168 giá trị) |
| Dùng LSTM có đặc trưng lịch | `python demo/predict.py --checkpoint demo/checkpoints/LSTM_MT_162_co_lich.pt --csv demo/sample_input.csv` |
| Chấm cả tập test, so với baseline | `python demo/predict.py --evaluate-test` |

Kết quả mẫu của lệnh đầu tiên:

```
Mô hình : MLP | chuỗi huấn luyện MT_162 | lookback 168 | đặc trưng lịch: không
Đầu vào : 3672 giờ, dùng 168 giờ cuối (giờ cuối = 447.51 kW)
Dự báo  : 448.86 kW cho giờ nhãn 2014-08-01 01:00
```

Giá trị thực của giờ đó là 451,48 kW.

---

## 4. Định dạng dữ liệu đầu vào

- **CSV hai cột** `timestamp,kW`, mỗi giờ một dòng, có hoặc không có dòng tiêu đề; phân cách bằng `,` `;` hoặc Tab (xem `sample_input.csv`):

  ```
  timestamp,kW
  2014-07-25 02:00:00,312.4
  2014-07-25 03:00:00,298.1
  ...
  ```

- **`timestamp`** là **mốc kết thúc của giờ** (nhãn `02:00` gồm `01:15…02:00`), tăng dần và cách đều đúng 1 giờ.
- **Số giờ cần có:** MLP **168 giờ**, LSTM **24 giờ**. Đưa dư thì lấy các giờ cuối.
- **Đơn vị:** kW trung bình mỗi giờ; phải là dữ liệu của khách hàng **MT_162** (cùng thang đo lúc huấn luyện). Dữ liệu thiếu, vô cực hoặc âm bị từ chối.
- **Một cột số** (không có timestamp) chỉ dùng được với MLP. LSTM bắt buộc có timestamp vì dùng đặc trưng lịch (giờ, thứ, cuối tuần, ngày lễ Bồ Đào Nha).

---

## 5. Ba checkpoint đi kèm (`demo/checkpoints/`)

| Tệp | Mô hình | Cấu hình | Đặc trưng lịch |
|---|---|---|---|
| `MLP_MT_162_hidden128_khong_lich.pt` (**mặc định**) | MLP | lookback 168, **hidden 128, dropout 0,2** (đã dò tham số), seed 42 | không |
| `MLP_MT_162_khong_lich.pt` | MLP | lookback 168, hidden 64, dropout 0,1 (cấu hình khởi điểm, trước khi dò), seed 42 | không |
| `LSTM_MT_162_co_lich.pt` | LSTM | lookback 24, hidden 128, 2 lớp, dropout 0,2, seed 42 | có |

**Đối chiếu với báo cáo MLP của Bảo Quốc** (test MAE trên MT_162):

| | Báo cáo Bảo Quốc | Demo | Khớp |
|---|---|---|---|
| Cấu hình MLP | hidden 128, dropout 0,2 (đã dò tham số) | MLP mặc định: hidden 128, dropout 0,2 | Có |
| MAE test của MLP | 20,638 (trung bình 5 seed) | 20,398 (seed 42) | Cùng mô hình; chỉ khác 1 seed so với trung bình 5 seed |
| Cấu hình LSTM | lookback 24, hidden 128, 2 lớp, có lịch | giống hệt | Có |
| MAE test của LSTM | 26,632 (trung bình 5 seed) | 25,987 (seed 42) | Cùng mô hình; chỉ khác 1 seed so với trung bình 5 seed |

Checkpoint MLP mặc định do Bảo Quốc gửi (`MLP_MT_162_s42.pt` trong gói `MLP_MT_162_hidden128_5seed`). Theo gói đó, 5 seed (42–46) cho MAE test 20,3977 / 20,3768 / 20,9114 / 20,7949 / 20,7113, trung bình **20,6384**, đúng số trong báo cáo. Demo chỉ mang seed 42. MLP `hidden 64` được giữ lại để so sánh "trước / sau khi dò tham số" (báo cáo, Bảng 5: 21,434 ± 0,700 so với 20,638 ± 0,240).

### Kiểm tra checkpoint: `--evaluate-test`

Chấm toàn bộ tập test (3.672 giờ, 01/08/2014 → 01/01/2015) bằng đúng quy ước huấn luyện (mặc định MLP hidden 128):

| | MAE (kW) | RMSE (kW) | MASE |
|---|---|---|---|
| MLP hidden 128 (seed 42) | 20,398 | 30,979 | 0,432 |
| LSTM có lịch (seed 42) | 25,987 | 39,451 | 0,551 |
| Seasonal Naive 168h | 34,226 | 57,327 | 0,725 |
| Seasonal Naive 24h | 96,487 | 146,194 | 2,044 |

MAE và RMSE của MLP khớp số Bảo Quốc ghi cho seed 42 (20,3977 và 30,9795) và sổ cái `results/experiments.csv`; LSTM khớp sổ cái (25,9871). Riêng LSTM cần cài `holidays`; thiếu gói này cột ngày lễ bằng 0 và MAE lệch khoảng 0,07 kW.

---

## 6. Xử lý lỗi thường gặp

| Thông báo / hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| `Cần ít nhất 168 giờ lịch sử, chỉ có N` | Đầu vào thiếu giờ; đưa đủ số giờ theo mô hình (mục 4) |
| `Cột timestamp phải cách đều đúng 1 giờ` | Dữ liệu nhảy giờ hoặc không tăng dần; kiểm tra lại CSV |
| `Checkpoint dùng đặc trưng lịch: cần dữ liệu CSV có cột timestamp` | Đang dùng LSTM với dữ liệu một cột; thêm timestamp hoặc đổi sang MLP |
| `Không đọc được dữ liệu. Mỗi dòng cần dạng: timestamp,kW` | CSV sai dạng; so với `sample_input.csv` |
| `Dữ liệu có giá trị thiếu, vô cực hoặc âm` | Xóa hoặc sửa các dòng lỗi |
| Cổng 8000 bị chiếm, không mở được trang | `python demo/app.py --port 8080` rồi mở cổng mới |
| `ModuleNotFoundError: torch` (hoặc gói khác) | Chưa bật `.venv` hoặc chưa cài: làm lại mục 1, bước 2–3 |
| LSTM lệch khoảng 0,07 kW so với báo cáo | Chưa cài `holidays`: `pip install holidays==0.107` |
| Không mở được trang từ máy khác | Bình thường: máy chủ chỉ phục vụ `127.0.0.1` vì lý do an toàn |

---

## 7. Kiểm thử tự động (tùy chọn)

```powershell
python -m pytest tests/test_demo_predict.py -q
```

Kết quả mong đợi: `4 passed`.

---

## 8. Hạn chế

Đây là mô hình thử nghiệm phục vụ học tập, **không dùng cho vận hành thực tế**. Chỉ dự báo một bước (t+1), một khách hàng, và checkpoint chỉ đúng với dữ liệu cùng thang đo với MT_162.
