# Phase 2: calibrated candidate selector cho multi-scale PARSeq TTA

Phase 2 chỉ thay đổi cách chọn kết quả từ 65 nhánh inference của Phase 1; không fine-tune và không cập nhật trọng số PARSeq.
Selector được chọn bằng Group K-fold trên validation, sau đó fit lại trên toàn bộ validation và khóa trước khi chạy test.

## Kết quả chính

| Tập dữ liệu | Phương pháp | Exact Match | Character Accuracy |
| --- | --- | ---: | ---: |
| Validation OOF | Baseline | 92.6952% | 98.6524% |
| Validation OOF | Phase 1 consensus | 94.7103% | 99.2037% |
| Validation OOF | Phase 2 selector | 95.2141% | 99.3262% |
| Test | Baseline | 91.9708% | 98.8684% |
| Test | Phase 1 consensus tái tạo | 93.9173% | 99.1662% |
| Test | Phase 2 selector đã khóa | 94.1606% | 99.1662% |

## Selector đã khóa

- Pairwise logistic C: `3.0`.
- Ngưỡng chuyển khỏi kết quả Phase 1: `2.0`.
- Số feature: `57`.
- Phase 2 sửa đúng/làm sai so với baseline trên test: **11/2**.
- Phase 2 sửa đúng/làm sai so với Phase 1 trên test: **1/0**.

## 21 ảnh khó

- Có ít nhất một ứng viên đúng: **7/21**.
- Phase 1 tự chọn đúng: **1/21**.
- Phase 2 tự chọn đúng: **1/21**.

## Cách selector hoạt động

Mỗi chuỗi dự đoán duy nhất được gộp phiếu từ các view. Ranker sử dụng số phiếu, độ tin cậy đã hiệu chỉnh, độ tin cậy lịch sử của view, đồng thuận gần đúng theo edit distance, cấu trúc biển số và tương tác giữa kích thước/tỉ lệ ảnh với upscale hoặc unwrap. Chỉ chuyển khỏi consensus Phase 1 khi chênh lệch điểm vượt ngưỡng đã chọn trên validation.

## Các cấu hình validation tốt nhất

| C | Switch margin | Exact Match OOF | Character Accuracy OOF | Sửa đúng | Làm sai |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | 95.2141% | 99.3262% | 10 | 0 |
| 3 | 1.25 | 95.2141% | 99.3262% | 10 | 0 |
| 10 | 2 | 94.9622% | 99.2956% | 9 | 0 |
| 1 | 2 | 94.7103% | 99.2649% | 8 | 0 |
| 1 | 1.25 | 94.7103% | 99.2649% | 8 | 0 |
| 0.3 | 0.75 | 94.7103% | 99.2649% | 8 | 0 |
| 1 | 0.75 | 94.7103% | 99.2649% | 9 | 1 |
| 1 | 0.4 | 94.7103% | 99.2649% | 9 | 1 |
| 0.3 | 0.1 | 94.7103% | 99.2649% | 9 | 1 |
| 0.3 | 0.05 | 94.7103% | 99.2649% | 9 | 1 |