"""Tải dữ liệu gốc ElectricityLoadDiagrams20112014 (UCI) vào data/raw/.

Chạy: python src/download_data.py

Nếu tải tự động thất bại (đổi đường dẫn, mất mạng, bị chặn), tải THỦ CÔNG:
  1. Mở https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
  2. Bấm Download, giải nén, đặt file LD2011_2014.txt vào data/raw/
Tệp ~700 MB, KHÔNG đưa lên Git (đã nằm trong .gitignore).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

try:
    from src.config import load_config, path_from
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import load_config, path_from

# Đường dẫn tải trực tiếp theo quy ước static của UCI. Nếu lỗi, dùng cách thủ công ở trên.
ZIP_URL = ("https://archive.ics.uci.edu/static/public/321/"
           "electricityloaddiagrams20112014.zip")
MIN_SIZE_BYTES = 100 * 1024 * 1024  # file đủ phải lớn hơn ~100 MB


def main():
    cfg = load_config()
    raw_file = path_from(cfg, "raw_file")
    raw_dir = raw_file.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    if raw_file.exists() and raw_file.stat().st_size > MIN_SIZE_BYTES:
        print(f"Đã có dữ liệu: {raw_file}. Bỏ qua bước tải.")
        return

    zip_path = raw_dir / "electricityloaddiagrams20112014.zip"
    try:
        print(f"Đang tải: {ZIP_URL}")
        urllib.request.urlretrieve(ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(raw_file.name)]
            if not names:
                raise RuntimeError(f"Không thấy {raw_file.name} trong tệp zip.")
            zf.extract(names[0], raw_dir)
            extracted = raw_dir / names[0]
            if extracted != raw_file:
                extracted.replace(raw_file)
        zip_path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Tải tự động thất bại: {exc}")
        print("Hãy tải thủ công theo hướng dẫn ở đầu file src/download_data.py.")
        sys.exit(1)

    if raw_file.stat().st_size < MIN_SIZE_BYTES:
        print("Tệp tải về nhỏ bất thường, hãy kiểm tra lại.")
        sys.exit(1)
    print(f"Hoàn tất: {raw_file} ({raw_file.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
