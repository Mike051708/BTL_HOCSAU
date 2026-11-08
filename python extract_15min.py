import pandas as pd

input_file = 'data/raw/LD2011_2014.txt'
output_file = 'data/processed/series_15min.csv'

print("Đang đọc dữ liệu gốc...")
# File txt gốc dùng dấu ';' để tách cột và dấu ',' cho phần thập phân
df = pd.read_csv(input_file, sep=';', decimal=',')

# Đổi tên cột đầu tiên (đang để trống tên trong file gốc) thành 'timestamp'
df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)

# Trích xuất cột thời gian và 4 chuỗi khách hàng cần thiết
cols_to_keep = ['timestamp', 'MT_234', 'MT_217', 'MT_162', 'MT_314']
df_selected = df[cols_to_keep]

# Xuất ra file CSV mới
df_selected.to_csv(output_file, index=False)
print(f"Hoàn tất! Đã lưu dữ liệu theo phút tại: {output_file}")