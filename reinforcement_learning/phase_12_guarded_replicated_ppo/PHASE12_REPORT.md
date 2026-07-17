# Báo cáo Phase 12 — Guarded replicated PPO

Phase 11 seed 728 tăng exact từ `47,20%` lên `47,8667%`, character accuracy từ `82,8013%` lên `83,5529%`, đạt `16 fix / 6 break`. Exact và character paired CI đều hoàn toàn dương, nhưng McNemar hai phía `p=0,05248` vừa vượt ngưỡng `<0,05`; vì vậy Phase 11 không được promote.

Phase 12 khóa guard label-free trước holdout tiếp theo: chỉ commit action PPO seed 728 khi `input_transform=existing_plate_crop` và minimum side `<128`; trường hợp khác rollback baseline. Trên ba external đã mở, rule đạt lần lượt `7/1`, `6/1`, `14/2`, gộp `27 fix / 4 break`, exact delta `+0,8666` điểm %, CI hoàn toàn dương và McNemar `p=3,40e-5`. Đây chỉ là development evidence.

Power contract cần 1.500 group mới, ước tính power McNemar `87,68%`. Dataset hiện chỉ còn 235 candidate unique-label chưa mở trước media SHA/crop validation, thiếu 1.265 so với target và thiếu 265 so với formal minimum 500. Cần bổ sung dữ liệu mới vào `labels.csv`; không cần accept thủ công, vẫn dùng `extracted_character` và loại `review_status=rejected`.

Phạm vi sheet được hiểu rõ là dòng 733 trở xuống: `sheet_row >= 733` (hướng tới số dòng lớn hơn), không phải các dòng 2–733.

## Pipeline đã hoàn thiện trước dữ liệu mới

- `prepare_fresh_holdout.py` kiểm tra đủ 1.500 candidate trước media hashing/render và trước tạo output directory; pool thiếu không để lại partial artifact.
- `evaluate.py` xác minh manifest/candidate/checkpoint/cache/guard hash, claim receipt one-shot rồi mới đọc cache; formal gate tính trên guarded action, không phải raw PPO.
- `promote.py` từ chối mọi summary không qua đủ paired CI, character, net-fix và McNemar hoặc sai receipt/provenance.
- `runtime.py` yêu cầu plate-crop metadata, đọc kích thước thật và short-circuit ảnh guard-disallowed xuống một baseline view; ảnh eligible mới chạy đủ chín view.
- detector-to-crop adapter ghi thêm `input_transform=crop_source_bounding_box` và crop dimensions, nên tự động rollback baseline trong Phase 12 runtime.
- `results/opened_development_guard_audit.json` tái lập trực tiếp từ cache/selections Phase 8/9/11 và từ chối ghi nếu `27 fix / 4 break`, delta hoặc McNemar khác prospective lock; artifact luôn đánh dấu `promotion_eligible=false`.
