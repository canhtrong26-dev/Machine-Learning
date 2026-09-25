# Đề kiểm tra môn Trí tuệ nhân tạo

Dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe (file `CSDL_TieuDuong.csv`) bằng Machine Learning (Logistic Regression, Random Forest, SVM). Giao diện web chạy trên localhost (Flask).

## Cách chạy

```bash
cd webapp
pip install -r requirements.txt
python app.py
```
Link demo: https://machine-learning-p49j.vercel.app/

Mở trình duyệt tại `http://127.0.0.1:5000`.

Khi khởi động, server sẽ in ra console kết quả tiền xử lý dữ liệu (Bước 1-6: xác định bài toán, thu thập, làm sạch, biến đổi, phân chia, huấn luyện, đánh giá 3 mô hình). Trang web (Bước 7) cho phép nhập 13 chỉ số sức khỏe, hiển thị % nguy cơ, mức cảnh báo (Thấp/Trung bình/Cao) và giải thích 3 yếu tố ảnh hưởng nhiều nhất.

## Cấu trúc

```
webapp/
├── app.py          # Flask app + Bước 7: route giao diện & API dự đoán
├── ml_logic.py      # Bước 1-6: xử lý dữ liệu, huấn luyện, đánh giá mô hình
├── templates/        # Giao diện HTML
├── static/           # CSS, JavaScript
└── CSDL_TieuDuong.csv
```
