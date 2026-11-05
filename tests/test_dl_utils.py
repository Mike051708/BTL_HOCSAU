"""Kiểm tra các hàm dùng chung của phần học sâu (src/dl_utils.py).

Cùng tinh thần với tests/test_baseline.py: tập trung vào những chỗ dễ sai
nhất của bài toán chuỗi thời gian, đặc biệt là RÒ RỈ DỮ LIỆU TƯƠNG LAI.

Chạy:
    python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

# dl_utils cần torch; nếu máy chưa cài thì bỏ qua cả file thay vì báo lỗi
# thu thập (lỗi thu thập sẽ làm hỏng toàn bộ phiên pytest).
pytest.importorskip("torch", reason="Cần PyTorch để kiểm tra src/dl_utils.py")

from src.dl_utils import (          # noqa: E402
    CAL_COLUMNS,
    calendar_feats,
    make_eval_windows,
    make_windows,
    set_seed_all,
)


# --------------------------------------------------
# 1. Cửa sổ trượt: hình dạng và căn nhãn
# --------------------------------------------------

def test_make_windows_hinh_dang_va_can_nhan():
    """X[i] phải là lookback giá trị NGAY TRƯỚC y[i], không lệch một bước."""
    gia_tri = np.arange(100, dtype=np.float32)
    X, y = make_windows(gia_tri, lookback=24)

    assert X.shape == (76, 24, 1)          # 100 - 24 = 76 mẫu
    assert y.shape == (76, 1)

    # Mẫu đầu: X = 0..23, nhãn = 24
    np.testing.assert_array_equal(X[0].reshape(-1), gia_tri[0:24])
    assert y[0, 0] == pytest.approx(24.0)

    # Mẫu cuối: X = 75..98, nhãn = 99
    np.testing.assert_array_equal(X[-1].reshape(-1), gia_tri[75:99])
    assert y[-1, 0] == pytest.approx(99.0)


def test_make_windows_khong_ro_ri_tuong_lai():
    """Nhãn y[i] KHÔNG được xuất hiện bên trong cửa sổ X[i]."""
    gia_tri = np.arange(200, dtype=np.float32)
    X, y = make_windows(gia_tri, lookback=48)
    for i in range(0, len(X), 17):          # lấy mẫu thưa cho nhanh
        assert y[i, 0] > X[i].max(), "Nhãn phải nằm SAU mọi giá trị trong cửa sổ"


def test_make_windows_bao_loi_khi_chuoi_qua_ngan():
    with pytest.raises(ValueError):
        make_windows(np.arange(10, dtype=np.float32), lookback=24)


# --------------------------------------------------
# 2. Cửa sổ đánh giá: ngữ cảnh lấy từ tập TRƯỚC
# --------------------------------------------------

@pytest.mark.parametrize("lookback", [24, 48, 168])
def test_make_eval_windows_so_nhan_khong_doi(lookback):
    """Mọi lookback phải cho ra ĐÚNG len(eval) nhãn.

    Đây là điều kiện để so sánh các cấu hình lookback khác nhau một cách công
    bằng: cùng tập nhãn, cùng số mẫu. Trong dự án, validation và test đều có
    3.672 giờ nên mọi cấu hình phải cho đúng 3.672 nhãn.
    """
    train = np.arange(1000, dtype=np.float32)
    val = np.arange(1000, 1000 + 3672, dtype=np.float32)

    X, y = make_eval_windows(train, val, lookback)

    assert len(X) == len(val) == 3672
    assert X.shape == (3672, lookback, 1)


def test_make_eval_windows_ngu_canh_tu_tap_truoc():
    """Cửa sổ đầu tiên phải ghép lookback giá trị CUỐI của tập train."""
    train = np.arange(100, dtype=np.float32)
    val = np.arange(100, 110, dtype=np.float32)
    lookback = 24

    X, y = make_eval_windows(train, val, lookback)

    # X[0] = 24 giá trị cuối của train (76..99), nhãn = val[0] = 100
    np.testing.assert_array_equal(X[0].reshape(-1), train[-lookback:])
    assert y[0, 0] == pytest.approx(val[0])

    # X[1] trượt đi một bước: 23 giá trị cuối train + val[0]
    np.testing.assert_array_equal(X[1].reshape(-1)[:-1], train[-lookback + 1:])
    assert X[1].reshape(-1)[-1] == pytest.approx(val[0])


def test_make_eval_windows_khong_dung_nhan_de_du_bao():
    """Giá trị cần dự báo không được nằm trong cửa sổ đầu vào của chính nó."""
    train = np.arange(500, dtype=np.float32)
    val = np.arange(500, 600, dtype=np.float32)
    X, y = make_eval_windows(train, val, lookback=48)
    for i in range(len(X)):
        assert y[i, 0] not in set(X[i].reshape(-1).tolist())


def test_make_eval_windows_bao_loi_khi_thieu_ngu_canh():
    with pytest.raises(ValueError):
        make_eval_windows(np.arange(5, dtype=np.float32),
                          np.arange(10, dtype=np.float32), lookback=24)


# --------------------------------------------------
# 3. Đặc trưng lịch
# --------------------------------------------------

def test_calendar_feats_hinh_dang():
    idx = pd.date_range("2014-08-01", periods=72, freq="h")
    cal = calendar_feats(idx, use_holiday=False)
    assert cal.shape == (72, len(CAL_COLUMNS))
    assert cal.dtype == np.float32


def test_calendar_feats_gio_dung_chu_ky():
    """sin/cos của giờ phải lặp lại sau đúng 24 giờ."""
    idx = pd.date_range("2014-08-01 00:00", periods=48, freq="h")
    cal = calendar_feats(idx, use_holiday=False)
    np.testing.assert_allclose(cal[0, :2], cal[24, :2], atol=1e-6)
    # 0h và 23h phải GẦN nhau (lý do dùng sin/cos thay cho số nguyên 0-23)
    assert np.linalg.norm(cal[0, :2] - cal[23, :2]) < \
           np.linalg.norm(cal[0, :2] - cal[12, :2])


def test_calendar_feats_cuoi_tuan():
    """Cột is_weekend phải bật đúng thứ Bảy và Chủ nhật."""
    idx = pd.date_range("2014-08-04", periods=7, freq="D")   # 04/08/2014 là thứ Hai
    cal = calendar_feats(idx, use_holiday=False)
    cot = CAL_COLUMNS.index("is_weekend")
    np.testing.assert_array_equal(cal[:, cot], [0, 0, 0, 0, 0, 1, 1])


def test_calendar_feats_khong_ngay_le_khi_tat():
    idx = pd.date_range("2014-12-24", periods=5, freq="D")   # có 25/12
    cal = calendar_feats(idx, use_holiday=False)
    assert cal[:, CAL_COLUMNS.index("is_holiday")].sum() == 0


def test_calendar_feats_nhan_dung_giang_sinh():
    """25/12 phải được đánh dấu là ngày lễ (lịch Bồ Đào Nha)."""
    pytest.importorskip("holidays", reason="Cần gói holidays")
    idx = pd.date_range("2014-12-24", periods=4, freq="D")
    cal = calendar_feats(idx, use_holiday=True)
    le = cal[:, CAL_COLUMNS.index("is_holiday")]
    assert le[1] == 1.0, "25/12/2014 phải là ngày lễ"
    assert le[0] == 0.0, "24/12/2014 không phải ngày lễ chính thức"


# --------------------------------------------------
# 4. Tái lập
# --------------------------------------------------

def test_set_seed_all_tai_lap_duoc():
    """Cùng seed phải cho cùng kết quả ngẫu nhiên ở numpy và torch."""
    import torch

    set_seed_all(42)
    a_np, a_torch = np.random.rand(5), torch.rand(5)
    set_seed_all(42)
    b_np, b_torch = np.random.rand(5), torch.rand(5)

    np.testing.assert_array_equal(a_np, b_np)
    assert torch.equal(a_torch, b_torch)


def test_set_seed_all_seed_khac_cho_ket_qua_khac():
    set_seed_all(42)
    a = np.random.rand(5)
    set_seed_all(43)
    b = np.random.rand(5)
    assert not np.allclose(a, b)
