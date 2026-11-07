"""Trang web demo dự báo điện năng giờ kế tiếp, chạy cục bộ trên máy đã clone.

Chạy từ thư mục gốc dự án:
    python demo/app.py            # rồi mở http://127.0.0.1:8000
    python demo/app.py --port 8080

Không cần cài thêm gói nào ngoài requirements.txt (dùng http.server của thư viện chuẩn).
Máy chủ chỉ lắng nghe ở 127.0.0.1 nên người khác trong mạng không truy cập được.
Dùng lại hàm của demo/predict.py, nên kết quả giống hệt bản dòng lệnh.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import predict  # noqa: E402  (demo/predict.py)

CKPT_DIR = HERE / "checkpoints"
MAX_BODY = 2 * 1024 * 1024            # 2 MB là quá đủ cho 168 giờ dữ liệu
LABELS = {
    "MLP_MT_162_hidden128_khong_lich.pt": "MLP · hidden 128 · như báo cáo (mặc định)",
    "MLP_MT_162_khong_lich.pt": "MLP · hidden 64 · khởi điểm",
    "LSTM_MT_162_co_lich.pt": "LSTM · 24 giờ · có đặc trưng lịch",
}

_lock = threading.Lock()
_models: dict = {}
_eval_cache: dict = {}


def list_checkpoints():
    out = []
    for p in sorted(CKPT_DIR.glob("*.pt"), key=lambda q: (not q.name.startswith("MLP"), q.name)):  # MLP lên đầu = mặc định
        _, ck = get_model(p.name)
        out.append({"name": p.name, "label": LABELS.get(p.name, p.name),
                    "model": ck["model_name"], "lookback": ck["lookback"],
                    "calendar": bool(ck["calendar"]), "customer": ck.get("customer", "?")})
    return out


def get_model(name: str):
    """Chỉ nạp tệp nằm trong demo/checkpoints/ (không nhận đường dẫn tùy ý)."""
    path = (CKPT_DIR / Path(name).name)
    if path.suffix != ".pt" or not path.is_file():
        raise ValueError(f"Không có checkpoint '{name}'.")
    with _lock:
        if path.name not in _models:
            _models[path.name] = predict.load_checkpoint(path)
        return _models[path.name]


def parse_csv(text: str):
    """CSV 'timestamp,kW' (có hoặc không có dòng tiêu đề) hoặc một cột số."""
    text = text.strip()
    if not text:
        raise ValueError("Chưa có dữ liệu.")
    head = text.splitlines()[0]
    sep = next((s for s in (",", ";", "\t") if s in head), ",")   # không tự đoán: "300" sẽ bị tách ở chữ 0
    df = pd.read_csv(io.StringIO(text), header=None, sep=sep)
    try:                                   # dòng đầu là tiêu đề nếu cột giá trị không phải số
        float(df.iloc[0, df.shape[1] - 1])
    except ValueError:
        df = df.iloc[1:].reset_index(drop=True)
    if df.shape[1] == 1:
        return pd.to_numeric(df.iloc[:, 0]).to_numpy(dtype=float), None
    try:
        ts = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]))
        vals = pd.to_numeric(df.iloc[:, 1]).to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise ValueError("Không đọc được dữ liệu. Mỗi dòng cần dạng: timestamp,kW "
                         "(ví dụ 2014-08-01 01:00:00,451.5).") from None
    if len(ts) > 1 and not (ts.to_series().diff().dropna() == pd.Timedelta("1h")).all():
        raise ValueError("Cột timestamp phải cách đều đúng 1 giờ và tăng dần.")
    return vals, ts


def sample_series(ck):
    """168 giờ cuối của validation (cột theo checkpoint) và giá trị thực của giờ kế tiếp."""
    col = ck.get("customer", "MT_162")
    val = pd.read_csv(ROOT / "data" / "processed" / "validation.csv",
                      encoding="utf-8-sig", parse_dates=["timestamp"])
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv", encoding="utf-8-sig")
    return (val[col].to_numpy(dtype=float)[-168:], pd.DatetimeIndex(val["timestamp"])[-168:],
            float(test[col].iloc[0]))


def do_predict(body: dict):
    model, ck = get_model(body.get("checkpoint", ""))
    actual = None
    mode = body.get("mode")
    if mode == "sample":
        values, ts, actual = sample_series(ck)
    elif mode == "csv":
        values, ts = parse_csv(body.get("text", ""))
    else:
        raise ValueError("Chế độ không hợp lệ.")
    with _lock:
        y, target = predict.predict_next(model, ck, values, ts)
    L = ck["lookback"]
    shown = values[-168:]
    shown_ts = [t.strftime("%d/%m %H:%M") for t in ts[-168:]] if ts is not None else None
    return {"prediction": y, "target_time": target.strftime("%d/%m/%Y %H:%M") if target is not None else None,
            "lookback": L, "n_input": int(len(values)), "last_value": float(values[-1]),
            "history": [float(v) for v in shown], "history_time": shown_ts,
            "actual": actual, "model": ck["model_name"], "calendar": bool(ck["calendar"])}


def do_evaluate(name: str):
    if name not in _eval_cache:
        model, ck = get_model(name)
        with _lock:
            col, m, base = predict.evaluate_test(model, ck)
        rows = [{"name": ck["model_name"], **{k: m[k] for k in ("MAE", "RMSE", "MASE")}}]
        rows += [{"name": n, **{k: r[k] for k in ("MAE", "RMSE", "MASE")}} for n, r in base.items()]
        _eval_cache[name] = {"customer": col, "n": int(m["N_samples"]), "rows": rows}
    return _eval_cache[name]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json; charset=utf-8"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        try:
            if path == "/":
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/checkpoints":
                self._send(200, list_checkpoints())
            elif path == "/api/evaluate":
                name = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("checkpoint", "")
                self._send(200, do_evaluate(name))
            else:
                self._send(404, {"error": "Không tìm thấy."})
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": str(e)})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                raise ValueError("Dữ liệu quá lớn.")
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path != "/api/predict":
                return self._send(404, {"error": "Không tìm thấy."})
            self._send(200, do_predict(body))
        except Exception as e:  # noqa: BLE001  (ValueError, lỗi CSV...) -> báo cho người dùng
            self._send(400, {"error": str(e)})

    def log_message(self, fmt, *args):      # bớt ồn trên terminal
        pass


def main():
    ap = argparse.ArgumentParser(description="Web demo dự báo điện năng (chạy cục bộ).")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    a = ap.parse_args()
    if not list(CKPT_DIR.glob("*.pt")):
        sys.exit(f"Không thấy checkpoint nào trong {CKPT_DIR}.")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Demo đang chạy tại {url}   (Ctrl+C để dừng)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng.")


if __name__ == "__main__":
    main()
