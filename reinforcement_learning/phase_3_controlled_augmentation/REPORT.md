# Báo cáo thử nghiệm Phase 3

## Kết luận

Pipeline fine-tune augmentation đã được triển khai và chạy thực tế. Tuy nhiên, chưa có epoch fine-tuned nào vượt checkpoint cha trên validation đã khóa. Cơ chế bảo vệ đã giữ checkpoint epoch 0, vì vậy model tốt nhất không bị giảm chất lượng.

Không nên công bố Phase 3 là đã tăng accuracy của trọng số model. Cải thiện trước đó vẫn đến từ inference Phase 1–2: multi-scale TTA, preprocessing và consensus selector.

## Thiết lập đã chạy

- Checkpoint cha: `outputs/testing/refinement_finetune_20260710_142307/best_official_parseq_anpr.pt`.
- Train: 3.270 ảnh.
- Validation: 397 ảnh, khóa theo manifest của checkpoint cha.
- Test: 411 ảnh, khóa theo manifest của checkpoint cha.
- Learning rate: `3e-6`.
- Encoder đóng băng epoch đầu.
- Early stopping: 4 epoch không cải thiện.
- Policy chính: `full`.
- Hai ablation bổ sung: `resolution_only`, `restoration_only`.

## Xu hướng lỗi validation

| Nhóm | Số ảnh | Số lỗi | Exact Match |
|---|---:|---:|---:|
| Tiny | 79 | 16 | 79,75% |
| Small | 209 | 10 | 95,22% |
| Regular | 109 | 3 | 97,25% |
| Có khả năng hai dòng | 260 | 22 | 91,54% |
| Biển ngang | 137 | 7 | 94,89% |

Phân bố này xác nhận việc ưu tiên low-resolution, upscale và unwrap là đúng hướng. Trong 29 lỗi, 11 dự đoán thiếu một ký tự và 4 dự đoán thừa một ký tự; vì vậy chỉ tăng sharpness không đủ giải quyết vấn đề sequence length.

## Kết quả canonical với manifest đã khóa

| Model | Refine | Validation Exact | Validation Character | Test Exact | Test Character |
|---|---:|---:|---:|---:|---:|
| Checkpoint cha | 1 | 92,6952% | 98,6217% | 91,7275% | 98,8386% |
| Checkpoint cha, refine được chọn trên validation | 2 | 92,6952% | 98,6524% | 91,9708% | 98,8684% |
| Phase 3 `full`, checkpoint được giữ | 2 | 92,6952% | 98,6524% | 91,9708% | 98,8684% |

Phase 3 chọn epoch 0 vì các epoch mới đều thấp hơn validation cha:

| Epoch | Encoder | Validation Exact | Validation Character |
|---:|---|---:|---:|
| 0 | Checkpoint cha | 92,6952% | 98,6217% |
| 1 | Đóng băng | 92,4433% | 98,5911% |
| 2 | Mở | 92,1914% | 98,5911% |
| 3 | Mở | 91,9395% | 98,5605% |
| 4 | Mở | 91,9395% | 98,5605% |

Tensor của `best_phase3_parseq_anpr.pt` đã được đối chiếu với checkpoint cha: 175/175 tensor giống hoàn toàn.

## Kiểm soát augmentation thực tế

Trong profile `full`, tỷ lệ trung bình theo epoch xấp xỉ:

| Phép biến đổi | Tỷ lệ mẫu |
|---|---:|
| Ảnh gần nguyên bản | 19,78% |
| Low-resolution | 39,63% |
| Zoom | 33,87% |
| Unwrap hai dòng | 19,47% |
| Blur | 14,19% |
| JPEG | 14,59% |
| Noise | 11,03% |
| Photometric | 21,66% |
| Upscale 2× | 77,16% |
| Upscale 3× | 11,80% |

Mỗi ảnh chỉ nhận tối đa hai degradation. Không có flip, rotation lớn hoặc crop mạnh.

## Phát hiện dataset drift

Dataset hiện tại khác manifest đã dùng trong Phase 1–2:

- 9/397 target validation đã thay đổi.
- 8/411 target test đã thay đổi.

Các run thử đầu dùng nhãn hiện tại có metric cao hơn nhưng không thể so trực tiếp với Phase 1–2. Script đã được sửa để mặc định lấy validation/test manifest cạnh checkpoint cha. Run chuẩn là:

`outputs/phase3_controlled_aug_full_frozen_eval`

## Diễn giải

Checkpoint cha đã gần hội tụ trên 3.270 ảnh train. Augmentation giúp tăng độ khó của train nhưng không thêm thông tin ký tự mới; do đó loss augmentation cao hơn mà validation sạch không tăng. Đặc biệt, nhóm tiny và hai dòng cần thêm mẫu train thật hoặc pseudo-label được kiểm duyệt, không chỉ thêm degradation tổng hợp.

Hướng tiếp theo hợp lý là xây dựng hard-example training set từ ảnh train/validation độc lập với test, tạo nhiều view có cùng nhãn, rồi chọn checkpoint bằng composite validation gồm clean và fixed robust views. Không được đưa 21 ảnh hard test vào train.

