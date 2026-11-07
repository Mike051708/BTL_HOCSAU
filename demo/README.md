# Demo dự báo điện năng giờ kế tiếp

`predict.py` nạp một checkpoint PyTorch (`.pt`), nhận một đoạn dữ liệu kW theo giờ và in ra **dự báo của giờ kế tiếp**. Checkpoint đã chứa kiến trúc, siêu tham số, `scaler_mean` và `scaler_scale`, nên kết quả được đưa thẳng về kW mà không cần dữ liệu huấn luyện.

## Chạy nhanh

Cần Python có `torch`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`; cài thêm `holidays==0.107` nếu dùng checkpoint có đặc trưng lịch (`pip install -r requirements.txt`). Chạy từ thư mục gốc dự án:

```powershell
python demo/predict.py                                    # trình diễn nhanh (MLP, MT_162)
python demo/predict.py --csv demo/sample_input.csv        # dữ liệu mới dạng CSV
python demo/predict.py --values 280 290 ...               # danh sách kW (168 giá trị cho MLP)
python demo/predict.py --evaluate-test                    # chấm cả tập test, so với baseline
python demo/predict.py --checkpoint demo/checkpoints/LSTM_MT_162_co_lich.pt --csv demo/sample_input.csv
```

Kết quả mẫu:

```
Mô hình : MLP | chuỗi huấn luyện MT_162 | lookback 168 | đặc trưng lịch: không
Dự báo  : 452.61 kW cho giờ nhãn 2014-08-01 01:00
```

Giá trị thực tế của giờ đó là 451,48 kW.

## Định dạng đầu vào

- **CSV:** hai cột, `timestamp` rồi `kW`. `timestamp` là **mốc kết thúc của giờ** (nhãn `02:00` gồm `01:15…02:00`), cách đều đúng 1 giờ, tăng dần. Xem `sample_input.csv`.
- **Số giờ cần có:** bằng `lookback` của checkpoint — MLP 168 giờ, LSTM 24 giờ. Nếu đưa dư thì lấy các giờ cuối.
- **Đơn vị:** kW trung bình mỗi giờ. Dữ liệu thiếu, vô cực hoặc âm bị từ chối.
- **Checkpoint có đặc trưng lịch** (LSTM trong thư mục này) bắt buộc dùng `--csv` vì cần timestamp để tính giờ, thứ, cuối tuần, ngày lễ Bồ Đào Nha. `--values` chỉ chạy với checkpoint không dùng lịch.

## Checkpoint đi kèm (`demo/checkpoints/`)

| Tệp | Mô hình | Cấu hình | Đặc trưng lịch |
|---|---|---|---|
| `MLP_MT_162_khong_lich.pt` (mặc định) | MLP | lookback 168, hidden 64, dropout 0,1, seed 42 | không |
| `LSTM_MT_162_co_lich.pt` | LSTM | lookback 24, hidden 128, 2 lớp, dropout 0,2, seed 42 | có |

Cả hai huấn luyện trên chuỗi chính **MT_162** (scaler của chuỗi này). Đưa dữ liệu của khách hàng khác vào sẽ cho kết quả sai thang đo.

**Lưu ý:** MLP ở đây là cấu hình khởi điểm (`hidden 64`), chưa phải cấu hình tinh chỉnh cuối (`hidden 128`, `dropout 0,2`) nêu trong README chính. Hai checkpoint là một seed duy nhất; số trong báo cáo là trung bình 5 seed.

## Kiểm tra checkpoint: `--evaluate-test`

Chấm toàn bộ tập test (3.672 giờ, 01/08/2014 → 01/01/2015) bằng đúng quy ước huấn luyện và so với hai Seasonal Naive:

| | MAE (kW) | RMSE (kW) | MASE |
|---|---|---|---|
| MLP (seed 42) | 20,406 | 31,103 | 0,432 |
| LSTM có lịch (seed 42) | 25,987 | 39,451 | 0,551 |
| Seasonal Naive 168h | 34,226 | 57,327 | 0,725 |
| Seasonal Naive 24h | 96,487 | 146,194 | 2,044 |

Hai dòng mô hình khớp sổ cái `results/experiments.csv` (20,4057 và 25,9871). Riêng LSTM cần cài `holidays`; thiếu gói này cột ngày lễ bằng 0 và MAE lệch khoảng 0,07 kW.

## Hạn chế

Đây là mô hình thử nghiệm phục vụ học tập, không dùng cho vận hành thực tế. Chỉ dự báo một bước (t+1), một khách hàng, và checkpoint chỉ đúng cho dữ liệu cùng thang đo với MT_162.
