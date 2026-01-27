# 🌫️ AIR GUARD – Dự báo PM2.5 và Phân loại AQI bằng Machine Learning

Mini Project - AIR GUARD – DỰ BÁO PM2.5 VÀ CẢNH BÁO AQI THEO TRẠM
Nhóm: 11
## 📌 Giới thiệu đề tài

Ô nhiễm không khí, đặc biệt là bụi mịn PM2.5, là một trong những vấn đề môi trường nghiêm trọng tại các đô thị lớn.
Chỉ số AQI (Air Quality Index) được sử dụng để đánh giá mức độ ảnh hưởng của chất lượng không khí đến sức khỏe con người.

Trong mini project này, nhóm xây dựng hệ thống **AIR GUARD** nhằm:

- Dự báo nồng độ PM2.5 theo thời gian
- Phân loại mức độ AQI theo từng trạm quan trắc
- Áp dụng các kỹ thuật học máy, bao gồm cả **semi-supervised learning** để tận dụng dữ liệu chưa gán nhãn

Mục tiêu không chỉ là xây dựng mô hình dự đoán, mà còn đánh giá hiệu quả của các phương pháp học khác nhau trong bối cảnh dữ liệu môi trường thực tế.

## ⚙️ Pipeline xử lý dữ liệu và mô hình

Toàn bộ hệ thống được xây dựng theo pipeline tự động gồm các bước:

1. Tiền xử lý và khám phá dữ liệu (Preprocessing & EDA)
2. Chuẩn bị đặc trưng (Feature Engineering)
3. Xây dựng mô hình supervised (Baseline)
4. Áp dụng semi-supervised learning:
   - Self-training
   - Co-training
5. So sánh và đánh giá kết quả các mô hình

Pipeline được tự động hóa bằng `papermill`, cho phép chạy toàn bộ notebook chỉ bằng một lệnh:

```bash
python run_papermill.py
```

---

## 🔷 3. Phần việc cá nhân – Semi-supervised Learning & Baseline

## 🧠 Phần việc thực hiện: Baseline & Semi-supervised Learning

Trong dự án này, em phụ trách các nội dung chính sau:

### ✅ 1. Xây dựng Baseline Supervised Model

- Chia dữ liệu theo thời gian (time-based split)
- Huấn luyện mô hình phân loại AQI
- Đánh giá bằng các chỉ số:
  - Accuracy
  - F1-macro
  - Confusion Matrix

Mục tiêu của baseline là tạo mốc so sánh cho các phương pháp semi-supervised.

---

### ✅ 2. Chuẩn bị dữ liệu cho Semi-supervised Learning

Dữ liệu được chia thành:

- Tập có nhãn (labeled)
- Tập chưa có nhãn (unlabeled)
- Tập validation
- Tập test

Các tập dữ liệu được lưu dưới dạng `.pkl` để tái sử dụng cho các thuật toán semi-supervised.

---

### ✅ 3. Self-training

Quy trình self-training gồm:

1. Huấn luyện model trên dữ liệu có nhãn
2. Dự đoán nhãn cho dữ liệu chưa gán nhãn
3. Chọn các mẫu có độ tin cậy cao
4. Bổ sung vào tập huấn luyện
5. Lặp lại nhiều vòng

Mục tiêu là mở rộng tập huấn luyện mà không cần thêm dữ liệu gán nhãn thủ công.

---

### ✅ 4. Co-training

Trong co-training:

- Tách đặc trưng thành hai view độc lập
- Huấn luyện hai mô hình song song
- Mỗi mô hình gán nhãn cho dữ liệu mới của mô hình còn lại
- Các mẫu tin cậy cao được thêm dần vào tập train

Phương pháp này giúp giảm thiên lệch và cải thiện độ ổn định so với self-training.

---

### ✅ 5. Đánh giá và so sánh

Kết quả của các mô hình được so sánh dựa trên:

- Accuracy
- F1-macro

Qua đó đánh giá mức độ cải thiện của semi-supervised learning so với supervised learning trong bài toán phân loại AQI.

