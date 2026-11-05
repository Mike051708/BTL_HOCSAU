"""Các kiến trúc học sâu dùng cho bài toán dự báo t+1.

Mọi mô hình có CÙNG chữ ký forward(x, cal):
    x   : (batch, lookback, 1)  - lịch sử đã chuẩn hóa
    cal : (batch, n_cal)        - đặc trưng lịch CỦA GIỜ CẦN DỰ BÁO
    ra  : (batch, 1)

Nhờ cùng chữ ký, cả ba mô hình chạy qua đúng một vòng lặp huấn luyện
trong src/dl_utils.py -> so sánh giữa chúng là so sánh công bằng.

Khi n_cal = 0 thì phần đặc trưng lịch bị bỏ qua hoàn toàn (dùng để chạy
ablation "có / không có đặc trưng lịch").
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecast(nn.Module):
    """LSTM nhiều lớp, lấy hidden state của lớp cuối tại bước thời gian cuối.

    Vì sao lấy h_n[-1] chứ không phải out[:, -1, :]: với LSTM một chiều hai
    giá trị này bằng nhau, nhưng h_n[-1] viết rõ ý "trạng thái tổng kết của
    lớp trên cùng" và không phụ thuộc vào chiều dài chuỗi.

    Đặc trưng lịch được nối vào SAU khi LSTM đã cuộn xong, thay vì lặp lại ở
    mọi bước thời gian: lịch của giờ cần dự báo là một thông tin tĩnh, lặp nó
    168 lần chỉ làm loãng tín hiệu chuỗi.
    """

    def __init__(self, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, n_cal: int = 0, n_feat: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            # PyTorch bo qua dropout khi num_layers = 1 va canh bao; tranh bang 0.0
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + n_cal, 1)
        self.n_cal = n_cal

    def forward(self, x, cal=None):
        _, (h_n, _) = self.lstm(x)
        z = self.drop(h_n[-1])                      # (batch, hidden)
        if self.n_cal:
            z = torch.cat([z, cal[:, :self.n_cal]], dim=1)
        return self.head(z)


class MLPForecast(nn.Module):
    """MLP tham chiếu: xem cả cửa sổ như một vector phẳng.

    Giữ lại ở đây để chạy lại MLP bằng ĐÚNG vòng lặp của LSTM. So sánh
    "LSTM có scheduler" với "MLP không scheduler" là so sánh không hợp lệ.
    """

    def __init__(self, lookback: int = 168, h1: int = 128, h2: int = 64,
                 dropout: float = 0.1, n_cal: int = 0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(lookback + n_cal, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.head = nn.Linear(h2, 1)
        self.n_cal = n_cal

    def forward(self, x, cal=None):
        z = x.squeeze(-1)                           # (batch, lookback)
        if self.n_cal:
            z = torch.cat([z, cal[:, :self.n_cal]], dim=1)
        return self.head(self.body(z))


class CNNLSTMForecast(nn.Module):
    """Conv1D trích mẫu cục bộ rồi mới đưa vào LSTM.

    Lý do thử: LSTM phải cuộn qua lookback bước, chuỗi càng dài tín hiệu từ
    bước đầu càng mờ (vanishing gradient). Conv1D + MaxPool rút ngắn chuỗi
    một nửa trước khi LSTM xử lý, nên LSTM phải nhớ ít bước hơn.
    """

    def __init__(self, hidden: int = 64, layers: int = 1, dropout: float = 0.2,
                 n_cal: int = 0, n_filters: int = 64, kernel: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_filters, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=n_filters, hidden_size=hidden, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0, batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + n_cal, 1)
        self.n_cal = n_cal

    def forward(self, x, cal=None):
        z = self.conv(x.transpose(1, 2))            # (batch, filters, lookback/2)
        _, (h_n, _) = self.lstm(z.transpose(1, 2))
        z = self.drop(h_n[-1])
        if self.n_cal:
            z = torch.cat([z, cal[:, :self.n_cal]], dim=1)
        return self.head(z)


MODEL_REGISTRY = {
    "LSTM": LSTMForecast,
    "MLP": MLPForecast,
    "CNNLSTM": CNNLSTMForecast,
}


def build_model(name: str, *, lookback: int, n_cal: int, hidden: int = 64,
                layers: int = 2, dropout: float = 0.2):
    """Tạo mô hình theo tên, truyền đúng tham số mà kiến trúc đó cần."""
    name = name.upper()
    if name == "LSTM":
        return LSTMForecast(hidden=hidden, layers=layers, dropout=dropout, n_cal=n_cal)
    if name == "MLP":
        return MLPForecast(lookback=lookback, h1=max(hidden * 2, 32), h2=hidden,
                           dropout=dropout, n_cal=n_cal)
    if name == "CNNLSTM":
        return CNNLSTMForecast(hidden=hidden, layers=layers, dropout=dropout, n_cal=n_cal)
    raise ValueError(f"Khong biet mo hinh '{name}'. Chon: {list(MODEL_REGISTRY)}")


def count_params(model) -> int:
    """Đếm số tham số huấn luyện được (đưa vào báo cáo khi so sánh mô hình)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
