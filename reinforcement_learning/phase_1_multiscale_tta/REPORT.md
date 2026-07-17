# Kiểm thử multi-scale zoom TTA cho PARSeq

Đây là thử nghiệm chỉ thay đổi luồng inference; checkpoint PARSeq không được fine-tune lại.
Tham số consensus được chọn hoàn toàn trên validation và được khóa trước khi đánh giá test.

## Kết quả

| Tập dữ liệu | Phương pháp | Exact Match | Character Accuracy | Sửa đúng | Làm sai |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Baseline | 92.6952% | 98.6524% | - | - |
| Validation | Consensus đã khóa | 94.7103% | 99.2037% | 8 | 0 |
| Validation | Oracle | 97.7330% | 99.7243% | 20 | 0 |
| Test | Baseline | 91.9708% | 98.8684% | - | - |
| Test | Consensus đã khóa | 93.9173% | 99.1662% | 11 | 3 |
| Test | Oracle | 96.5937% | 99.5831% | 19 | 0 |

Oracle dùng target để kiểm tra liệu có bất kỳ nhánh nào đọc đúng hay không; đây không phải kết quả có thể triển khai.

## Nhóm 21 ảnh trước đây không pipeline nào đọc đúng

- Có ít nhất một nhánh TTA đọc đúng: **7/21**.
- Consensus đã khóa tự chọn đúng: **1/21**.

## Nhận xét

- Consensus tăng Exact Match trên test thêm **1.9465%** và Character Accuracy thêm **0.2978%**.
- Upscale 2× ở zoom 1.00 là view đơn tốt nhất trên validation; upscale trước enhancement có tác dụng rõ hơn thay đổi zoom đơn thuần.
- Unwrap có accuracy độc lập thấp, nhưng tạo được một số dự đoán đúng cho ảnh rất khó. Vì vậy unwrap chỉ nên là nhánh bỏ phiếu, không dùng mặc định.
- Selector vẫn làm sai 3 ảnh test vốn được baseline đọc đúng; cần cải thiện bước chọn ứng viên trước khi dùng production.

## Các view đơn tốt nhất trên validation

| View | Exact Match | Character Accuracy |
| --- | ---: | ---: |
| `full_z1.00_up2_train_baseline` | 93.9547% | 98.8974% |
| `full_z1.00_up3_train_baseline` | 93.7028% | 98.8974% |
| `full_z1.00_up2_clahe_clip1_tile4` | 93.4509% | 98.8361% |
| `full_z1.00_up2_clahe_rl_deblur_bilateral` | 93.4509% | 98.7749% |
| `full_z1.00_up3_clahe_clip1_tile4` | 93.4509% | 98.7749% |
| `full_z1.00_up3_clahe_rl_deblur_bilateral` | 93.1990% | 98.7443% |
| `full_z1.07_up2_clahe_clip1_tile4` | 92.9471% | 98.8668% |
| `full_z1.07_up2_clahe_rl_deblur_bilateral` | 92.9471% | 98.8668% |
| `full_z1.00_up2_adaptive_noise_3way` | 92.9471% | 98.6830% |
| `full_z1.00_up3_adaptive_noise_3way` | 92.9471% | 98.5911% |
| `full_z1.07_up3_train_baseline` | 92.6952% | 98.9280% |
| `full_z1.07_up3_clahe_clip1_tile4` | 92.6952% | 98.8361% |